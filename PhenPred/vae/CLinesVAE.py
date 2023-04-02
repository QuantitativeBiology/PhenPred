import sys

sys.path.extend(["/home/egoncalves/PhenPred"])

import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import CLinesVAEPlot as ploter
from PhenPred.vae.CLinesDrugResponseBenchmark import DrugResponseBenchmark

# Class variables - paths to csv files
_data_folder = "/data/benchmarks/clines/"
_data_files = dict(
    meth_csv_file=f"{_data_folder}/methylation.csv",
    gexp_csv_file=f"{_data_folder}/transcriptomics.csv",
    prot_csv_file=f"{_data_folder}/proteomics.csv",
    meta_csv_file=f"{_data_folder}/metabolomics.csv",
    dres_csv_file=f"{_data_folder}/drugresponse.csv",
    cris_csv_file=f"{_data_folder}/crisprcas9_22Q2.csv",
)

# Class variables - Hyperparameters
_hyperparameters = dict(
    num_epochs=3,
    learning_rate=1e-5,
    batch_size=32,
    n_folds=3,
    latent_dim=25,
    hidden_dim_1=0.1,
    hidden_dim_2=0.05,
    probability=0.5,
    group=15,
    alpha_kl=0.1,
    alpha_mse=0.9,
    optimizer_type="adam",
    w_decay=1e-5,
    loss_type="mse",
    activation_function=nn.Sigmoid(),
)

# Class variables - Torch
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(32)

# Class variables - Misc
_verbose = True
_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
_dirPlots = "/home/egoncalves/PhenPred/reports/vae/"


class CLinesDataset(Dataset):
    def __init__(self):
        # Read csv files
        self.df_meth = pd.read_csv(_data_files["meth_csv_file"], index_col=0).T
        self.df_gexp = pd.read_csv(_data_files["gexp_csv_file"], index_col=0).T
        self.df_prot = pd.read_csv(_data_files["prot_csv_file"], index_col=0).T
        self.df_meta = pd.read_csv(_data_files["meta_csv_file"], index_col=0).T
        self.df_dres = pd.read_csv(_data_files["dres_csv_file"], index_col=0).T
        self.df_cris = pd.read_csv(_data_files["cris_csv_file"], index_col=0).T

        # Union samples
        self.samples = pd.concat(
            [
                pd.Series(df.index)
                for df in [
                    self.df_meth,
                    self.df_gexp,
                    self.df_prot,
                    self.df_meta,
                    self.df_dres,
                    self.df_cris,
                ]
            ],
            axis=0,
        ).value_counts()
        self.samples = self.samples[
            self.samples > 1
        ]  # Keep only samples that are in at least 2 datasets
        self.samples = set(self.samples.index)
        self.samples -= {"SIDM00189", "SIDM00650"}
        self.samples = list(self.samples)

        if _verbose:
            print(f"[{_timestamp}] Samples = {len(self.samples)}")

        self.df_meth = self.df_meth.reindex(index=self.samples)
        self.df_gexp = self.df_gexp.reindex(index=self.samples)
        self.df_prot = self.df_prot.reindex(index=self.samples)
        self.df_meta = self.df_meta.reindex(index=self.samples)
        self.df_dres = self.df_dres.reindex(index=self.samples)
        self.df_cris = self.df_cris.reindex(index=self.samples)

        # Remove features with more than 50% of missing values
        self.df_prot = self.df_prot.loc[:, self.df_prot.isnull().mean() < 0.5]
        self.df_meta = self.df_meta.loc[:, self.df_meta.isnull().mean() < 0.5]
        self.df_dres = self.df_dres.loc[:, self.df_dres.isnull().mean() < 0.5]

        # Reduce the number of features
        self.df_meth = self.df_meth[
            self.df_meth.std().sort_values(ascending=False).head(3000).index
        ]
        self.df_gexp = self.df_gexp[
            self.df_gexp.std().sort_values(ascending=False).head(3000).index
        ]
        self.df_cris = self.df_cris[
            self.df_cris.std().sort_values(ascending=False).head(3000).index
        ]

        # Standardize the data
        self.x_meth, self.scaler_meth = self.process_df(self.df_meth)
        self.x_gexp, self.scaler_gexp = self.process_df(self.df_gexp)
        self.x_prot, self.scaler_prot = self.process_df(self.df_prot)
        self.x_meta, self.scaler_meta = self.process_df(self.df_meta)
        self.x_dres, self.scaler_dres = self.process_df(self.df_dres)
        self.x_cris, self.scaler_cris = self.process_df(self.df_cris)

        # Datasets list
        self.views = [
            self.x_meth,
            self.x_gexp,
            self.x_prot,
            self.x_meta,
            self.x_dres,
            self.x_cris,
        ]

        self.view_scalers = [
            self.scaler_meth,
            self.scaler_gexp,
            self.scaler_prot,
            self.scaler_meta,
            self.scaler_dres,
            self.scaler_cris,
        ]

        self.view_names = [
            "methylation",
            "transcriptomics",
            "proteomics",
            "metabolomics",
            "drugresponse",
            "crisprcas9",
        ]

        self.view_feature_names = dict(
            zip(
                self.view_names,
                [
                    list(self.df_meth.columns),
                    list(self.df_gexp.columns),
                    list(self.df_prot.columns),
                    list(self.df_meta.columns),
                    list(self.df_dres.columns),
                    list(self.df_cris.columns),
                ],
            )
        )

    def process_df(self, df):
        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(df)
        x = np.nan_to_num(x, copy=False)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float)

        return x, scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (
            self.x_meth[idx],
            self.x_gexp[idx],
            self.x_prot[idx],
            self.x_meta[idx],
            self.x_dres[idx],
            self.x_cris[idx],
        )


class BottleNeck(nn.Module):
    def __init__(self, hidden_dim, group, activation_function):
        super(BottleNeck, self).__init__()

        self.activation_function = activation_function
        self.hidden_dim = hidden_dim
        self.group = group

        self.groups = nn.ModuleList()
        for _ in range(group):
            group_layers = nn.ModuleList()
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim // group))
            )

            group_layers.append(
                nn.Sequential(nn.Linear(hidden_dim // group, hidden_dim // group))
            )
            self.groups.append(group_layers)

    def forward(self, x):
        activation = self.activation_function

        # start with the input, which in this case will be the result of the first
        # fully connected layer
        identity = torch.narrow(x, 1, 0, self.hidden_dim // self.group * self.group)
        out = []

        for group_layers in self.groups:
            group_out = x

            for layer in group_layers:
                group_out = activation(layer(group_out))
            out.append(group_out)

        # concatenate, the size should be equal to the hidden size
        out = torch.cat(out, dim=1)

        # Why do we add here the identity
        out += identity

        return out


class OMIC_VAE(nn.Module):
    def __init__(
        self,
        views,
        hidden_dim_1,
        hidden_dim_2,
        latent_dim,
        probability,
        group,
        activation_function,
    ):
        super(OMIC_VAE, self).__init__()

        self.activation_function = activation_function

        # -- Bottlenecks
        self.omics_bottlenecks = nn.ModuleList()
        for v in views:
            self.omics_bottlenecks.append(
                BottleNeck(
                    hidden_dim=v.shape[1],
                    group=group,
                    activation_function=activation_function,
                )
            )

        # -- Encoders
        self.omics_encoders = nn.ModuleList()
        for v in views:
            self.omics_encoders.append(
                nn.Sequential(
                    nn.Linear(
                        int(v.shape[1] // group * group),
                        int(hidden_dim_1 * v.shape[1] // group * group),
                    ),
                    nn.Dropout(p=probability),
                    activation_function,
                    nn.Linear(
                        int(hidden_dim_1 * v.shape[1] // group * group),
                        int(hidden_dim_2 * v.shape[1] // group * group),
                    ),
                    nn.Dropout(p=probability),
                    activation_function,
                )
            )

        # -- Mean
        self.mus = nn.ModuleList()
        for v in views:
            self.mus.append(
                nn.Sequential(
                    nn.Linear(
                        int(hidden_dim_2 * v.shape[1] // group * group), latent_dim
                    ),
                )
            )

        # -- Log-Var
        self.log_vars = nn.ModuleList()
        for v in views:
            self.log_vars.append(
                nn.Sequential(
                    nn.Linear(
                        int(hidden_dim_2 * v.shape[1] // group * group), latent_dim
                    ),
                )
            )

        # -- Decoders
        self.omics_decoders = nn.ModuleList()
        for v in views:
            self.omics_decoders.append(
                nn.Sequential(
                    nn.Linear(
                        latent_dim, int(hidden_dim_2 * v.shape[1] // group * group)
                    ),
                    nn.Dropout(p=probability),
                    activation_function,
                    nn.Linear(
                        int(hidden_dim_2 * v.shape[1] // group * group),
                        int(hidden_dim_1 * v.shape[1] // group * group),
                    ),
                    nn.Dropout(p=probability),
                    activation_function,
                    nn.Linear(
                        int(hidden_dim_1 * v.shape[1] // group * group), v.shape[1]
                    ),
                )
            )

    def encode(self, views):
        h_bottlenecks = []
        for view, encoder, bottleneck in zip(
            views, self.omics_encoders, self.omics_bottlenecks
        ):
            h_bottleneck_ = bottleneck(view)
            h_bottleneck_ = encoder(h_bottleneck_)
            h_bottlenecks.append(h_bottleneck_)
        return h_bottlenecks

    def mean_variance(self, h_bottlenecks):
        means, log_variances = [], []
        for h_bottleneck, mu, log_var in zip(h_bottlenecks, self.mus, self.log_vars):
            mean = mu(h_bottleneck)
            var = log_var(h_bottleneck)
            means.append(mean)
            log_variances.append(var)
        return means, log_variances

    def product_of_experts(self, means, log_variances):
        # Code taken from Integrating T-cell receptor and transcriptome for 3 large-scale
        # single-cell immune profiling analysis
        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities))
        logvar_joint = torch.sum(
            torch.stack([1.0 / torch.exp(log_var) for log_var in log_variances]),
            dim=0,
        )
        logvar_joint = torch.log(1.0 / logvar_joint)

        # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint,
        # where mu_prior = 0.0
        mu_joint = torch.sum(
            torch.stack(
                [mu / torch.exp(log_var) for mu, log_var in zip(means, log_variances)]
            ),
            dim=0,
        )
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

    def calculate_sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mean + eps * std

    def decode(self, z):
        return [decoder(z) for decoder in self.omics_decoders]

    def forward(self, views):
        h_bottlenecks = self.encode(views)
        means, log_variances = self.mean_variance(h_bottlenecks)
        mu_joint, logvar_joint = self.product_of_experts(means, log_variances)
        z = self.calculate_sample(mu_joint, logvar_joint)
        views_hat = self.decode(z)
        return views_hat


def mse_kl(
    views_hat,
    views,
    means,
    log_variances,
    alpha=0.1,
    lambd=1.0,
):
    n_samples = views[0].shape[0]
    if _hyperparameters["loss_type"] == "mse":
        mse_loss = sum(
            [
                nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1]
                for x, x_hat in zip(views, views_hat)
            ]
        )
    elif _hyperparameters["loss_type"] == "smoothl1":
        mse_loss = sum(
            [
                nn.SmoothL1Loss(reduction="sum")(x, x_hat) / x.shape[1]
                for x, x_hat in zip(views, views_hat)
            ]
        )
    else:
        mse_loss = sum(
            [
                torch.sqrt(nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1])
                for x, x_hat in zip(views, views_hat)
            ]
        )

    # Compute the KL loss
    kl_loss = 0
    for mu_i, logvar_i in zip(means, log_variances):
        kl_loss += (
            -0.5
            * torch.sum(1 + logvar_i - torch.pow(mu_i, 2) - torch.exp(logvar_i))
            / len(mu_i)
        )

    # Compute the total loss
    loss = (lambd * mse_loss) / n_samples + (alpha * kl_loss) / n_samples

    return loss, mse_loss / n_samples, kl_loss / n_samples


def cross_validation(data, model, optimizer):
    # Initiate Cross Validation
    cv = KFold(n_splits=_hyperparameters["n_folds"], shuffle=True)

    # Train Losses
    loss_train, mse_train, kl_train = [], [], []

    # Validation Losses
    loss_val, mse_val, kl_val = [], [], []

    # Train Losses - Dataset Specific
    mse_list = {d: [] for d in data.view_names}

    for train_idx, val_idx in cv.split(data):
        # Train Data
        data_train = torch.utils.data.Subset(data, train_idx)
        dataloader_train = DataLoader(
            data_train, batch_size=_hyperparameters["batch_size"], shuffle=True
        )

        # Validation Data
        data_test = torch.utils.data.Subset(data, val_idx)
        dataloader_test = DataLoader(
            data_test, batch_size=_hyperparameters["batch_size"], shuffle=False
        )

        # --- TRAINING LOOP
        model.train()

        # dataloader train is divided into batches
        for views in dataloader_train:
            n = views[0].size(0)

            views = [view.to(_device) for view in views]

            # Forward pass to get the predictions
            views_hat = model.forward(views)

            # Get last layer of encoder with bottleneck
            h_bottleneck = model.encode(views)

            # Get means and log_vars
            means, log_variances = model.mean_variance(h_bottleneck)
            mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

            # Calculate Losses
            loss, mse, kl = mse_kl(
                views_hat,
                views,
                mu_joint,
                logvar_joint,
            )

            if _hyperparameters["loss_type"] == "mse":
                mse_omics = [
                    nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1]
                    for x, x_hat in zip(views, views_hat)
                ]
            elif _hyperparameters["loss_type"] == "smoothl1":
                mse_omics = [
                    nn.SmoothL1Loss(reduction="sum")(x, x_hat) / x.shape[1]
                    for x, x_hat in zip(views, views_hat)
                ]
            else:
                mse_omics = [
                    torch.sqrt(nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1])
                    for x, x_hat in zip(views, views_hat)
                ]

            for i, d in enumerate(data.view_names):
                mse_list[d].append(mse_omics[i].item() / n)

            loss_train.append(loss.item())
            mse_train.append(mse.item())
            kl_train.append(kl.item())

            with torch.autograd.set_detect_anomaly(True):
                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0)
                optimizer.step()

        # --- VALIDATION LOOP
        model.eval()

        with torch.no_grad():
            for views in dataloader_test:
                views = [view.to(_device) for view in views]
                # Forward pass to get the predictions
                views_hat = model.forward(views)

                # Get last layer of encoder with bottleneck
                h_bottleneck = model.encode(views)

                # Get means and log_vars
                means, log_variances = model.mean_variance(h_bottleneck)
                mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

                # Calculate Losses
                loss, mse, kl = mse_kl(
                    views_hat,
                    views,
                    mu_joint,
                    logvar_joint,
                )

                loss_val.append(loss.item())
                mse_val.append(mse.item())
                kl_val.append(kl.item())

    return loss_train, mse_train, kl_train, loss_val, mse_val, kl_val, mse_list


def epoch(
    data,
):
    model = OMIC_VAE(
        views=data.views,
        hidden_dim_1=_hyperparameters["hidden_dim_1"],
        hidden_dim_2=_hyperparameters["hidden_dim_2"],
        latent_dim=_hyperparameters["latent_dim"],
        probability=_hyperparameters["probability"],
        group=_hyperparameters["group"],
        activation_function=_hyperparameters["activation_function"],
    ).to(_device)

    if _hyperparameters["optimizer_type"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=_hyperparameters["learning_rate"],
            weight_decay=_hyperparameters["w_decay"],
        )
    else:
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=_hyperparameters["learning_rate"],
            weight_decay=_hyperparameters["w_decay"],
        )

    losses_dict = {
        "loss_train": [],
        "mse_train": [],
        "kl_train": [],
        "loss_val": [],
    }

    losses_datasets = {d: [] for d in data.view_names}

    for epoch in range(_hyperparameters["num_epochs"]):
        # -- Cross Validation
        (
            loss_train,
            mse_train,
            kl_train,
            loss_val,
            mse_val,
            kl_val,
            mse_list,
        ) = cross_validation(
            data=data,
            model=model,
            optimizer=optimizer,
        )

        # -- Train Losses (CV + Batch Average)
        losses_dict["loss_train"].append(np.mean(loss_train))
        losses_dict["mse_train"].append(np.mean(mse_train))
        losses_dict["kl_train"].append(np.mean(kl_train))

        # -- Validation Losses (CV + Batch Average)
        losses_dict["loss_val"].append(np.mean(loss_val))

        # -- Train Losses Dataset Specific (CV + Batch Average)
        for v_name in data.view_names:
            losses_datasets[v_name].append(np.mean(mse_list[v_name]))

        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1}"
            + f"| Loss (train): {losses_dict['loss_train'][epoch]:.4f}"
            + f"| Loss (val): {losses_dict['loss_val'][epoch]:.4f}"
        )

    ploter.plot_losses(
        losses_dict,
        losses_datasets,
        _hyperparameters["alpha_kl"],
        _hyperparameters["alpha_mse"],
        timestamp=_timestamp,
    )
    predictions(
        data,
        model,
    )


def predictions(
    data,
    model,
):
    omics_dataloader = DataLoader(data, batch_size=len(data.samples), shuffle=False)

    # Dataframes
    latent_spaces = dict()
    imputed_datasets = dict()

    # Make predictions and latent spaces
    for views in omics_dataloader:
        views = [view.to(_device) for view in views]

        # Forward pass to get the predictions
        views_hat = model.forward(views)
        for i, (name, df) in enumerate(zip(data.view_names, views_hat)):
            imputed_datasets[name] = pd.DataFrame(
                data.view_scalers[i].inverse_transform(df.tolist()),
                index=data.samples,
                columns=data.view_feature_names[name],
            )

        # Get last layer of encoder with bottleneck
        h_bottleneck = model.encode(views)

        # Get means and log_vars
        means, log_variances = model.mean_variance(h_bottleneck)
        mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

        # Create Latent Spaces
        latent_spaces["joint"] = pd.DataFrame(
            model.calculate_sample(mu_joint, logvar_joint).tolist(),
            index=data.samples,
            columns=[f"Latent_{i+1}" for i in range(_hyperparameters["latent_dim"])],
        )

        for name, (mean, log_var) in zip(data.view_names, zip(means, log_variances)):
            latent_spaces[name] = pd.DataFrame(
                model.calculate_sample(mean, log_var).tolist(),
                index=data.samples,
                columns=[
                    f"Latent_{i+1}" for i in range(_hyperparameters["latent_dim"])
                ],
            )

    # Write to file
    for name, df in imputed_datasets.items():
        df.round(5).to_csv(
            f"{_dirPlots}/files/{_timestamp}_imputed_{name}.csv.gz", compression="gzip"
        )

    for name, df in latent_spaces.items():
        df.round(5).to_csv(
            f"{_dirPlots}/files/{_timestamp}_latent_{name}.csv.gz", compression="gzip"
        )


if __name__ == "__main__":
    # Load the first dataset
    clines_db = CLinesDataset()

    # Run the training loop
    epoch(clines_db)

    # Plot latent spaces
    ploter.plot_latent_spaces(
        _timestamp,
        clines_db.view_names,
        {
            k: _hyperparameters[k]
            for k in [
                "hidden_dim_1",
                "hidden_dim_2",
                "latent_dim",
                "probability",
                "group",
                "learning_rate",
                "n_folds",
                "batch_size",
            ]
        },
    )

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(_timestamp)
    dres_benchmark.run()
