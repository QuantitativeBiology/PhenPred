# DATA
import pandas as pd
import numpy as np

# PLOTS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import umap.umap_ as umap

# MODELS
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats

# SEEDS
import random
import warnings

# TIMESTAMP
import calendar
import time

# Class variables
_dirPlots = "/home/egoncalves/PhenPred/reports/vae/"

_dfileTranscriptomics = "/data/benchmarks/clines/transcriptomics.csv"
_dfileProteomics = "/data/benchmarks/clines/proteomics.csv"
_dfileMetabolomics = "/data/benchmarks/clines/metabolomics.csv"
_dfileMethylation = "/data/benchmarks/clines/methylation.csv"
_dfileCrisprCas9 = "/data/benchmarks/clines/crisprcas9_22Q2.csv"
_dfileDrugResponse = "/data/benchmarks/clines/drugresponse.csv"


class OmicsData(Dataset):
    def __init__(
        self,
        cells,
    ):
        # Load the Dataframes
        self.df_tran = pd.read_csv(_dfileTranscriptomics, index_col=0).T
        self.df_prot = pd.read_csv(_dfileProteomics, index_col=0).T
        self.df_drug = pd.read_csv(_dfileDrugResponse, index_col=0).T
        self.df_crispr = pd.read_csv(_dfileCrisprCas9, index_col=0).T
        self.df_meta = pd.read_csv(_dfileMetabolomics, index_col=0).T
        self.df_methy = pd.read_csv(_dfileMethylation, index_col=0).T

        # Remove Features > 95% missing values
        self.df_prot = self.df_prot.T[
            self.df_prot.T.isnull().sum(axis=1) / len(self.df_prot.T.columns) * 100 < 95
        ].T
        self.df_drug = self.df_drug.T[
            self.df_drug.T.isnull().sum(axis=1) / len(self.df_drug.T.columns) * 100 < 95
        ].T

        # Add the selected cell lines
        self.samples = cells
        self.df_tran = self.df_tran.reindex(index=self.samples)
        self.df_prot = self.df_prot.reindex(index=self.samples)
        self.df_drug = self.df_drug.reindex(index=self.samples)
        self.df_crispr = self.df_crispr.reindex(index=self.samples)
        self.df_meta = self.df_meta.reindex(index=self.samples)
        self.df_methy = self.df_methy.reindex(index=self.samples)

        # Normalize
        self.x_tran, self.scaler_tran = self.process_df(self.df_tran)
        self.x_prot, self.scaler_prot = self.process_df(self.df_prot)
        self.x_drug, self.scaler_drug = self.process_df(self.df_drug)
        self.x_crispr, self.scaler_crispr = self.process_df(self.df_crispr)
        self.x_meta, self.scaler_meta = self.process_df(self.df_meta)
        self.x_methy, self.scaler_methy = self.process_df(self.df_methy)

        # Datasets list
        self.views = [
            self.x_tran,
            self.x_prot,
            self.x_drug,
            self.x_crispr,
            self.x_meta,
            self.x_methy,
        ]

    def process_df(self, df):
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
            self.x_tran[idx],
            self.x_prot[idx],
            self.x_drug[idx],
            self.x_crispr[idx],
            self.x_meta[idx],
            self.x_methy[idx],
        )


class BottleNeck(nn.Module):
    def __init__(self, hidden_dim, group, activation_function, probability):
        super(BottleNeck, self).__init__()

        self.activation_function = activation_function
        self.hidden_dim = hidden_dim
        self.group = group

        self.groups = nn.ModuleList()
        for g in range(group):
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

        # start with the input, which in this case will be the result of the first fully connected layer
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
    ) -> None:

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
                    probability=probability,
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

    def integrate_latent_spaces(self, x):
        cov = torch.zeros(
            len(self.encoders) * self.latent_dim, len(self.encoders) * self.latent_dim
        )
        with torch.no_grad():
            for i in range(len(self.encoders)):
                z, _ = self.encoders[i](x)
                z_mean = z.mean(dim=0)
                cov_i = torch.mm((z - z_mean).t(), (z - z_mean))
                cov += cov_i
        cov /= x.size(0)
        self.cov = cov

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
    mu_joint,
    logvar_joint,
    means,
    log_variances,
    loss_type,
    alpha=0.1,
    lambd=1.0,
):
    n_samples = views[0].shape[0]
    if loss_type == "mse":
        mse_loss = sum(
            [
                nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1]
                for x, x_hat in zip(views, views_hat)
            ]
        )
    elif loss_type == "smoothl1":
        mse_loss = sum(
            [
                nn.SmoothL1Loss(reduction="sum")(x, x_hat) / x.shape[1]
                for x, x_hat in zip(views, views_hat)
            ]
        )
    elif loss_type == "rmse":
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


def cross_validation(
    data, n_folds, batch_size, model, optimizer, alpha_KL, alpha_MSE, loss_type
):

    # Initiate Cross Validation
    cv = KFold(n_folds, shuffle=True)

    # Train Losses
    loss_train = []
    mse_train = []
    kl_train = []

    # Validation Losses
    loss_val = []
    mse_val = []
    kl_val = []

    # Train Losses - Dataset Specific
    mse_list = {}

    for i in range(len(data.views)):
        mse_list[f"Dataset {i}"] = []

    for train_idx, val_idx in cv.split(data):

        # Train Data
        data_train = torch.utils.data.Subset(data, train_idx)
        dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)

        # Validation Data
        data_test = torch.utils.data.Subset(data, val_idx)
        dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True)

        # --- TRAINING LOOP
        model.train()

        # dataloader train is divided into batches
        for views in dataloader_train:
            n = views[0].size(0)

            views = [view.to(device) for view in views]

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
                means,
                log_variances,
                loss_type=loss_type,
                alpha=alpha_KL,
                lambd=alpha_MSE,
            )

            if loss_type == "mse":
                mse_omics = [
                    nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1]
                    for x, x_hat in zip(views, views_hat)
                ]
            elif loss_type == "smoothl1":
                mse_omics = [
                    nn.SmoothL1Loss(reduction="sum")(x, x_hat) / x.shape[1]
                    for x, x_hat in zip(views, views_hat)
                ]
            elif loss_type == "rmse":
                mse_omics = [
                    torch.sqrt(nn.MSELoss(reduction="sum")(x, x_hat) / x.shape[1])
                    for x, x_hat in zip(views, views_hat)
                ]

            for i in range(len(mse_omics)):
                mse_list[f"Dataset {i}"].append(mse_omics[i].item() / n)

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
                views = [view.to(device) for view in views]
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
                    means,
                    log_variances,
                    loss_type=loss_type,
                    alpha=alpha_KL,
                    lambd=alpha_MSE,
                )

                loss_val.append(loss.item())
                mse_val.append(mse.item())
                kl_val.append(kl.item())

    return loss_train, mse_train, kl_train, loss_val, mse_val, kl_val, mse_list


def plot_losses(losses_dict, losses_datasets, alpha_KL, alpha_MSE):

    dt = calendar.timegm(time.gmtime())

    losses = pd.DataFrame(losses_dict)
    losses["epoch"] = losses.index

    losses_omics = pd.DataFrame(losses_datasets)
    losses_omics["epoch"] = losses_omics.index

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.lineplot(
        data=pd.melt(losses.loc[:, ["loss_train", "loss_val", "epoch"]], ["epoch"]),
        x="epoch",
        y="value",
        hue="variable",
    )
    ax.set(
        title=f"Train and Validation Loss (lambda = {alpha_MSE} , alpha = {alpha_KL})",
        xlabel="Epoch",
        ylabel="Train Loss and Validation Loss",
    )
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(legend_handles, ["Train Loss", "Validation Loss"], title="Losses")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/losses/{dt}train_validation_loss_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.lineplot(
        data=pd.melt(losses.loc[:, ["mse_train", "kl_train", "epoch"]], ["epoch"]),
        x="epoch",
        y="value",
        hue="variable",
    )
    ax.set(
        title=f"Total loss (lambda = {alpha_MSE} , alpha = {alpha_KL})",
        xlabel="Epoch",
        ylabel="Reconstruction Loss and Regularization Loss",
    )
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(legend_handles, ["Reconstruction Loss", "KL Loss"], title="Losses")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/losses/{dt}reconst_reg_loss_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.lineplot(
        data=pd.melt(losses_omics, ["epoch"]), x="epoch", y="value", hue="variable"
    )
    ax.set(xlabel="Epoch", ylabel="Reconstruction Loss for each dataset")
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        legend_handles,
        [
            "Transcriptomics",
            "Proteomics",
            "Drug Response",
            "CRISPR",
            "Metabolomics",
            "Methylation",
        ],
        title="Reconstruction Losses",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/losses/{dt}reconst_omics_loss_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_latent_spaces(
    latent_space,
    latent_space_transcriptomics,
    latent_space_proteomics,
    latent_space_drugresponse,
    latent_space_crisprcas9,
    latent_space_metabolomics,
    latent_space_methylation,
    hidden_dim_1,
    hidden_dim_2,
    latent_dim,
    probability,
    group,
    learning_rate,
    n_folds,
    batch_size_initial,
):

    dt = calendar.timegm(time.gmtime())
    batch_size = batch_size_initial

    # Get Tissue Types
    samplesheet = pd.read_csv("/data/benchmarks/clines/samplesheet.csv", index_col=0)
    samplesheet = samplesheet.reindex(index=omics_db.samples)
    samplesheet = samplesheet["tissue"].fillna("Other tissue")

    # Get UMAP projections
    latent_space_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space
        )
    )
    latent_space_transcriptomics_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_transcriptomics
        )
    )
    latent_space_proteomics_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_proteomics
        )
    )
    latent_space_drugresponse_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_drugresponse
        )
    )
    latent_space_crisprcas9_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_crisprcas9
        )
    )
    latent_space_metabolomics_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_metabolomics
        )
    )
    latent_space_methylation_umap = pd.DataFrame(
        umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(
            latent_space_methylation
        )
    )

    # Joint
    latent_space_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Joint Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_joint_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_umap,
        x="latent_1",
        y="latent_2",
        c=transcriptomics.T.reindex(index=omics_db.samples)["VIM"].values,
        cmap="viridis",
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Joint Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    norm = plt.Normalize(
        transcriptomics.T.reindex(index=omics_db.samples)["VIM"].values.min(),
        transcriptomics.T.reindex(index=omics_db.samples)["VIM"].values.max(),
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_joint_VIM_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_umap,
        x="latent_1",
        y="latent_2",
        c=metabolomics.T.reindex(index=omics_db.samples)["1-methylnicotinamide"].values,
        cmap="viridis",
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Joint Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    norm = plt.Normalize(
        metabolomics.T.reindex(index=omics_db.samples)[
            "1-methylnicotinamide"
        ].values.min(),
        metabolomics.T.reindex(index=omics_db.samples)[
            "1-methylnicotinamide"
        ].values.max(),
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm)
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_joint_methy_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Transcriptomics
    latent_space_transcriptomics_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_transcriptomics_umap.columns = [
        "latent_1",
        "latent_2",
        "Tissue Type",
    ]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_transcriptomics_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Transcriptomics Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_transcriptomics_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Proteomics
    latent_space_proteomics_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_proteomics_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_proteomics_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Proteomics Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_proteomics_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Drug Response
    latent_space_drugresponse_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_drugresponse_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_drugresponse_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Drug Response Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_drugresponse_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # CRISPR
    latent_space_crisprcas9_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_crisprcas9_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_crisprcas9_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of CRISPR Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_crispr_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # METABOLOMICS
    latent_space_metabolomics_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_metabolomics_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_metabolomics_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Metabolomics Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_metabolomics_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # METHYLATION
    latent_space_methylation_umap["Tissue Type"] = samplesheet.tolist()
    latent_space_methylation_umap.columns = ["latent_1", "latent_2", "Tissue Type"]

    sns.set()
    sns.set_theme()
    sns.set_style("dark")
    ax = sns.scatterplot(
        data=latent_space_methylation_umap,
        x="latent_1",
        y="latent_2",
        hue="Tissue Type",
        palette=PALETTE_TTYPE,
        alpha=0.5,
    )
    ax.set(
        title=f"UMAP projection of Methylation Latent Space \nhidden_dim_1 = {hidden_dim_1}, hidden_dim_2 = {hidden_dim_2}, latent_dim = {latent_dim}, dropout = {probability}, n_groups = {group},\nlr = {learning_rate}, n_folds = {n_folds}, batch_size = {batch_size}",
        xlabel="UMAP_1",
        ylabel="UMAP_2",
        xticklabels=[],
        yticklabels=[],
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(
        f"{_dirPlots}/latent/{dt}umap_methylation_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


##-- Test (Drug Response and Proteomics) --#


def evaluate_drug_response(drugresponse_final):

    dt = calendar.timegm(time.gmtime())
    # Original Dataset
    df_original = omics_db.df_drug.copy()

    # Autoencoder Dataset
    df_imputed = pd.DataFrame(
        omics_db.scaler_drug.inverse_transform(drugresponse_final)
    )
    df_imputed.columns = df_original.columns
    df_imputed.index = drugresponse_final.index
    df_imputed = np.transpose(df_imputed)
    df_imputed["id"] = df_imputed.index
    drugresponse_imputed_table = pd.melt(df_imputed, id_vars=["id"])
    # df_imputed.to_csv(f'/home/sofiaapolinario/test/dataoutputs/{dt}drugresponse_imputed_all.csv', index=True)

    # Mean Dataset
    df_imputed_mean = omics_db.df_drug.copy()
    df_imputed_mean = df_imputed_mean.apply(lambda x: x.fillna(x.mean()), axis=0)
    df_imputed_mean = np.transpose(df_imputed_mean)
    df_imputed_mean["id"] = df_imputed_mean.index
    drugresponse_imputed_mean_table = pd.melt(df_imputed_mean, id_vars=["id"])

    # Mofa Dataset
    df_imputed_mofa = pd.read_csv(
        "/home/sofiaapolinario/test/dataMOFA/drugresponseMOFA.csv"
    )
    df_imputed_mofa = df_imputed_mofa.rename(columns={"Unnamed: 0": "id"})
    drugresponse_imputed_mofa_table = pd.melt(df_imputed_mofa, id_vars=["id"])

    # New Lab Values
    drug_response_new = pd.read_csv(
        "/home/sofiaapolinario/test/dataMOFA/drugresponse_24Jul22.csv"
    )
    drug_response_new["id"] = (
        drug_response_new["DRUG_ID"].astype(str)
        + ";"
        + drug_response_new["DRUG_NAME"]
        + ";"
        + drug_response_new["DATASET"]
    )
    drug_response_new = drug_response_new.drop(
        columns=["DRUG_ID", "DRUG_NAME", "DATASET"]
    )
    drug_response_new.index = drug_response_new["id"]
    drug_response_new_table = pd.melt(drug_response_new, id_vars=["id"])

    df_original = np.transpose(df_original)
    df_original["id"] = df_original.index
    drug_response_table = pd.melt(df_original, id_vars=["id"])

    # Correlation
    drugresponse_corr = pd.merge(
        drug_response_new_table,
        drugresponse_imputed_table,
        how="outer",
        left_on=["id", "variable"],
        right_on=["id", "variable"],
    )
    drugresponse_corr.columns = drugresponse_corr.columns.str.replace("value_x", "NEW")
    drugresponse_corr.columns = drugresponse_corr.columns.str.replace(
        "value_y", "IMPUTED"
    )
    drugresponse_corr = pd.merge(
        drugresponse_corr,
        drug_response_table,
        how="outer",
        left_on=["id", "variable"],
        right_on=["id", "variable"],
    )
    drugresponse_corr.columns = drugresponse_corr.columns.str.replace(
        "value", "ORIGINAL"
    )
    drugresponse_corr = pd.merge(
        drugresponse_corr,
        drugresponse_imputed_mean_table,
        how="outer",
        left_on=["id", "variable"],
        right_on=["id", "variable"],
    )
    drugresponse_corr.columns = drugresponse_corr.columns.str.replace("value", "MEAN")
    drugresponse_corr = pd.merge(
        drugresponse_corr,
        drugresponse_imputed_mofa_table,
        how="outer",
        left_on=["id", "variable"],
        right_on=["id", "variable"],
    )
    drugresponse_corr.columns = drugresponse_corr.columns.str.replace("value", "MOFA")

    corr = drugresponse_corr[
        (~drugresponse_corr["NEW"].isna())
        & (~drugresponse_corr["IMPUTED"].isna())
        & (~drugresponse_corr["MOFA"].isna())
        & (drugresponse_corr["ORIGINAL"].isna())
    ]

    corr_vae = pd.DataFrame(corr[["NEW", "IMPUTED"]].corr(method="pearson")).iloc[0, 1]
    corr_mean = pd.DataFrame(corr[["NEW", "MEAN"]].corr(method="pearson")).iloc[0, 1]
    corr_mofa = pd.DataFrame(corr[["NEW", "MOFA"]].corr(method="pearson")).iloc[0, 1]
    corr_mofa_VAE = pd.DataFrame(corr[["IMPUTED", "MOFA"]].corr(method="pearson")).iloc[
        0, 1
    ]

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(data=corr, x="NEW", y="IMPUTED", alpha=0.5, color="green")
    ax.set(
        title="Correlation between New Measured Values and Autoencoder Predictions",
        xlabel="New Measured Values",
        ylabel="Autoencoder Predictions",
    )
    X_plot = np.linspace(-8, 13, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(corr["NEW"].values) & np.isfinite(corr["IMPUTED"].values)
    )[0]
    r, p = stats.pearsonr(
        x=corr["NEW"].iloc[indices].values, y=corr["IMPUTED"].iloc[indices].values
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/drugresponse/{dt}drugresponse_new_imputed_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.kdeplot(data=corr, x="NEW", y="IMPUTED", fill=True, color="green")
    ax.set(
        title="Correlation between New Measured Values and Autoencoder Predictions",
        xlabel="New Measured Values",
        ylabel="Autoencoder Predictions",
    )
    plt.savefig(
        f"{_dirPlots}/drugresponse/{dt}drugresponse_new_imputed_density_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(data=corr, x="MOFA", y="IMPUTED", alpha=0.5, color="green")
    ax.set(
        title="Correlation between MOFA and Autoencoder Predictions",
        xlabel="MOFA Predictions",
        ylabel="Autoencoder Predictions",
    )
    X_plot = np.linspace(-8, 13, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(corr["MOFA"].values) & np.isfinite(corr["IMPUTED"].values)
    )[0]
    r, p = stats.pearsonr(
        x=corr["MOFA"].iloc[indices].values, y=corr["IMPUTED"].iloc[indices].values
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/drugresponse/{dt}drugresponse_mofa_imputed_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def corr_sample(s, df1, df2, min_samples=300):

    col = df1.columns.intersection(df2.columns)
    array1 = df1.loc[s, col].values
    array2 = df2.loc[s, col].values
    indices = np.where(np.isfinite(array1) & np.isfinite(array2))[0]

    if len(indices) >= min_samples:
        return mean_squared_error(array1[indices], array2[indices])

    else:
        return np.NaN


def evaluate_proteomics(proteomics_final):

    dt = calendar.timegm(time.gmtime())
    scaler = StandardScaler()

    # Original Dataset
    df_original = omics_db.df_prot.copy()

    # Imputed All Dataset
    df_imputed_all = pd.DataFrame(
        omics_db.scaler_prot.inverse_transform(proteomics_final)
    )
    df_imputed_all.columns = (
        df_original.columns
    )  # the columns are the same as original because it is a prediction
    df_imputed_all.index = proteomics_final.index
    # df_imputed_all.to_csv(f'/home/sofiaapolinario/test/dataoutputs/{dt}proteomics_imputed_all.csv', index=True)

    proteomics_imputed_all_table = df_imputed_all.copy()
    proteomics_imputed_all_table["id"] = proteomics_imputed_all_table.index
    proteomics_imputed_all_table = pd.melt(proteomics_imputed_all_table, id_vars=["id"])

    # Imputed only NA Dataset
    df_imputed_na = omics_db.df_prot.copy()
    df_imputed_na[df_imputed_na.isnull()] = df_imputed_all
    # df_imputed_na.to_csv(f'/home/sofiaapolinario/test/dataoutputs/{dt}proteomics_imputed_na.csv', index=True)

    # New Dataset
    df_new = pd.read_csv("/data/benchmarks/clines/proteomics_ccle.csv")
    df_new.index = df_new["Gene_Symbol"]
    df_new = df_new.drop(columns=["Gene_Symbol"]).T
    df_new = df_new.reindex(index=omics_db.samples)

    # Imputed All MOFA
    df_imputed_MOFA_all = pd.read_csv(
        "/home/sofiaapolinario/test/dataMOFA/proteomicsMOFA.csv"
    )
    df_imputed_MOFA_all = df_imputed_MOFA_all.rename(
        columns={"Unnamed: 0": "index_col"}
    )
    index_proteomics = []
    for i in df_imputed_MOFA_all.index_col:
        index_proteomics.append(i.replace("_proteomics", ""))
    df_imputed_MOFA_all.index = index_proteomics
    df_imputed_MOFA_all = df_imputed_MOFA_all.drop(columns=["index_col"])
    df_imputed_MOFA_all = df_imputed_MOFA_all.reindex(index=omics_db.df_prot.columns)
    df_imputed_MOFA_all = df_imputed_MOFA_all.T

    mofa_imputed_all_table = df_imputed_MOFA_all.copy()
    mofa_imputed_all_table["id"] = mofa_imputed_all_table.index
    mofa_imputed_all_table = pd.melt(mofa_imputed_all_table, id_vars=["id"])

    # Imputed only NA MOFA
    df_imputed_MOFA_na = omics_db.df_prot.copy()
    df_imputed_MOFA_na[df_imputed_MOFA_na.isnull()] = df_imputed_MOFA_all

    # Standardize the data
    df_new_scaled = pd.DataFrame(scaler.fit_transform(df_new))
    df_new_scaled.columns = df_new.columns
    df_new_scaled.index = df_new.index

    df_imputed_all_scaled = pd.DataFrame(scaler.fit_transform(df_imputed_all))
    df_imputed_all_scaled.columns = df_imputed_all.columns
    df_imputed_all_scaled.index = df_imputed_all.index

    df_imputed_na_scaled = pd.DataFrame(scaler.fit_transform(df_imputed_na))
    df_imputed_na_scaled.columns = df_imputed_na.columns
    df_imputed_na_scaled.index = df_imputed_na.index

    df_imputed_MOFA_all_scaled = pd.DataFrame(scaler.fit_transform(df_imputed_MOFA_all))
    df_imputed_MOFA_all_scaled.columns = df_imputed_MOFA_all.columns
    df_imputed_MOFA_all_scaled.index = df_imputed_MOFA_all.index

    df_imputed_MOFA_na_scaled = pd.DataFrame(scaler.fit_transform(df_imputed_MOFA_na))
    df_imputed_MOFA_na_scaled.columns = df_imputed_MOFA_na.columns
    df_imputed_MOFA_na_scaled.index = df_imputed_MOFA_na.index

    df_original_scaled = pd.DataFrame(scaler.fit_transform(df_original))
    df_original_scaled.columns = df_original.columns
    df_original_scaled.index = df_original.index

    correlation = pd.DataFrame()

    for sample in df_imputed_MOFA_all.index:
        corr_original = corr_sample(sample, df_new_scaled, df_original_scaled)

        # VAE and New
        corr_imputed = corr_sample(sample, df_new_scaled, df_imputed_all_scaled)
        corr_imputed_NA = corr_sample(sample, df_new_scaled, df_imputed_na_scaled)

        # MOFA and New
        corr_imputed_mofa = corr_sample(
            sample, df_new_scaled, df_imputed_MOFA_all_scaled
        )
        corr_imputed_NA_mofa = corr_sample(
            sample, df_new_scaled, df_imputed_MOFA_na_scaled
        )

        # Append the row
        corr_row = {
            "cell_line": sample,
            "corr_original": corr_original,
            "corr_imputed": corr_imputed,
            "corr_imputed_NA": corr_imputed_NA,
            "corr_imputed_mofa": corr_imputed_mofa,
            "corr_imputed_NA_mofa": corr_imputed_NA_mofa,
        }
        correlation = correlation.append(corr_row, ignore_index=True)

    correlation["index"] = correlation["cell_line"]

    ##-- VAE --##

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=correlation,
        x="corr_original",
        y="corr_imputed",
        alpha=0.5,
        color="orange",
    )
    ax.set(
        title="MSE per sample with New Proteomics Dataset (Autoencoder)",
        xlabel="MSE per sample (original dataset, new dataset)",
        ylabel="MSE per sample (decoder output, new dataset)",
    )
    X_plot = np.linspace(-0.2, 5, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(correlation["corr_original"].values)
        & np.isfinite(correlation["corr_imputed"].values)
    )[0]
    r, p = stats.pearsonr(
        x=correlation["corr_original"].iloc[indices].values,
        y=correlation["corr_imputed"].iloc[indices].values,
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/proteomics/{dt}proteomics_original_imputed_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=correlation,
        x="corr_original",
        y="corr_imputed_NA",
        alpha=0.5,
        color="orange",
    )
    ax.set(
        title="MSE per sample with New Proteomics Dataset (Autoencoder)",
        xlabel="MSE per sample (original dataset, new dataset)",
        ylabel="MSE per sample (imputed dataset, new dataset)",
    )
    X_plot = np.linspace(-0.2, 5, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(correlation["corr_original"].values)
        & np.isfinite(correlation["corr_imputed_NA"].values)
    )[0]
    r, p = stats.pearsonr(
        x=correlation["corr_original"].iloc[indices].values,
        y=correlation["corr_imputed_NA"].iloc[indices].values,
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/proteomics/{dt}proteomics_original_imputed_NA_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    ##-- MOFA --##
    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=correlation,
        x="corr_original",
        y="corr_imputed_mofa",
        alpha=0.5,
        color="orange",
    )
    ax.set(
        title="MSE per sample with New Proteomics Dataset (MOFA)",
        xlabel="MSE per sample (original dataset, new dataset)",
        ylabel="MSE per sample (model output, new dataset)",
    )
    X_plot = np.linspace(-0.2, 5, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(correlation["corr_original"].values)
        & np.isfinite(correlation["corr_imputed_mofa"].values)
    )[0]
    r, p = stats.pearsonr(
        x=correlation["corr_original"].iloc[indices].values,
        y=correlation["corr_imputed_mofa"].iloc[indices].values,
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/proteomics/{dt}proteomics_original_imputed_mofa_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=correlation,
        x="corr_original",
        y="corr_imputed_NA_mofa",
        alpha=0.5,
        color="orange",
    )
    ax.set(
        title="MSE per sample with New Proteomics Dataset (MOFA)",
        xlabel="MSE per sample (original dataset, new dataset)",
        ylabel="MSE per sample (imputed dataset, new dataset)",
    )
    X_plot = np.linspace(-0.2, 5, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(correlation["corr_original"].values)
        & np.isfinite(correlation["corr_imputed_NA_mofa"].values)
    )[0]
    r, p = stats.pearsonr(
        x=correlation["corr_original"].iloc[indices].values,
        y=correlation["corr_imputed_NA_mofa"].iloc[indices].values,
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/proteomics/{dt}proteomics_original_imputed_mofa_NA_.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    ##-- MOFA VS AutoEncoder --##
    proteomics_corr = pd.merge(
        proteomics_imputed_all_table,
        mofa_imputed_all_table,
        how="outer",
        left_on=["id", "GeneSymbol"],
        right_on=["id", "GeneSymbol"],
    )
    proteomics_corr.columns = proteomics_corr.columns.str.replace(
        "value_x", "AUTOENCODER"
    )
    proteomics_corr.columns = proteomics_corr.columns.str.replace("value_y", "MOFA")

    sns.set()
    sns.set_theme()
    sns.set_context("paper")
    ax = sns.scatterplot(
        data=proteomics_corr, x="MOFA", y="AUTOENCODER", alpha=0.5, color="orange"
    )
    X_plot = np.linspace(-6, 18, 20)
    Y_plot = X_plot
    plt.plot(X_plot, Y_plot, color="r")
    indices = np.where(
        np.isfinite(proteomics_corr["MOFA"].values)
        & np.isfinite(proteomics_corr["AUTOENCODER"].values)
    )[0]
    r, p = stats.pearsonr(
        x=proteomics_corr["MOFA"].iloc[indices].values,
        y=proteomics_corr["AUTOENCODER"].iloc[indices].values,
    )
    plt.text(0.05, 0.8, "personr = {:.4f}".format(r), transform=ax.transAxes)
    plt.text(0.05, 0.75, "p-value = {:.4f}".format(p), transform=ax.transAxes)
    plt.savefig(
        f"{_dirPlots}/proteomics/{dt}proteomics_autoencoder_mofa_.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


##-- Predictions --#
def predictions(
    model,
    hidden_dim_1,
    hidden_dim_2,
    latent_dim,
    probability,
    group,
    learning_rate,
    n_folds,
    batch_size,
):
    batch_size_initial = batch_size
    batch_size = omics_db.x_tran.shape[0]
    omics_dataloader = DataLoader(omics_db, batch_size=batch_size, shuffle=False)

    # Dataframes
    latent_space = pd.DataFrame()
    latent_space_transcriptomics = pd.DataFrame()
    latent_space_proteomics = pd.DataFrame()
    latent_space_drugresponse = pd.DataFrame()
    latent_space_crisprcas9 = pd.DataFrame()
    latent_space_metabolomics = pd.DataFrame()
    latent_space_methylation = pd.DataFrame()
    proteomics_final = pd.DataFrame()
    drugresponse_final = pd.DataFrame()

    for views in omics_dataloader:

        views = [view.to(device) for view in views]

        # Forward pass to get the predictions
        views_hat = model.forward(views)

        # Get last layer of encoder with bottleneck
        h_bottleneck = model.encode(views)

        # Get means and log_vars
        means, log_variances = model.mean_variance(h_bottleneck)
        mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

        # Create Latent Spaces
        z = model.calculate_sample(mu_joint, logvar_joint)

        views_latent = []
        for mean, log_var in zip(means, log_variances):
            z_i = model.calculate_sample(mean, log_var)
            views_latent.append(z_i.tolist())

        # Create Dataframes
        latent_space = pd.concat([latent_space, pd.DataFrame(z.tolist())])
        latent_space_transcriptomics = pd.concat(
            [latent_space_transcriptomics, pd.DataFrame(views_latent[0])]
        )
        latent_space_proteomics = pd.concat(
            [latent_space_proteomics, pd.DataFrame(views_latent[1])]
        )
        latent_space_drugresponse = pd.concat(
            [latent_space_drugresponse, pd.DataFrame(views_latent[2])]
        )
        latent_space_crisprcas9 = pd.concat(
            [latent_space_crisprcas9, pd.DataFrame(views_latent[3])]
        )
        latent_space_metabolomics = pd.concat(
            [latent_space_metabolomics, pd.DataFrame(views_latent[4])]
        )
        latent_space_methylation = pd.concat(
            [latent_space_methylation, pd.DataFrame(views_latent[5])]
        )
        proteomics_final = pd.concat(
            [proteomics_final, pd.DataFrame(views_hat[1].tolist())]
        )
        drugresponse_final = pd.concat(
            [drugresponse_final, pd.DataFrame(views_hat[2].tolist())]
        )

    latent_space.index = omics_db.samples
    latent_space_transcriptomics.index = omics_db.samples
    latent_space_proteomics.index = omics_db.samples
    latent_space_drugresponse.index = omics_db.samples
    latent_space_crisprcas9.index = omics_db.samples
    latent_space_metabolomics.index = omics_db.samples
    latent_space_methylation.index = omics_db.samples
    proteomics_final.index = omics_db.samples
    drugresponse_final.index = omics_db.samples

    dt = calendar.timegm(time.gmtime())

    plot_latent_spaces(
        latent_space,
        latent_space_transcriptomics,
        latent_space_proteomics,
        latent_space_drugresponse,
        latent_space_crisprcas9,
        latent_space_metabolomics,
        latent_space_methylation,
        hidden_dim_1,
        hidden_dim_2,
        latent_dim,
        probability,
        group,
        learning_rate,
        n_folds,
        batch_size_initial,
    )
    evaluate_drug_response(drugresponse_final)
    evaluate_proteomics(proteomics_final)


##-- Epochs --#
def epoch(
    data,
    num_epochs,
    hidden_dim_1,
    hidden_dim_2,
    latent_dim,
    probability,
    group,
    learning_rate,
    n_folds,
    batch_size,
    alpha_KL,
    alpha_MSE,
    optimizer_type,
    w_decay,
    loss_type,
    activation_function,
):

    model = OMIC_VAE(
        views=data.views,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        latent_dim=latent_dim,
        probability=probability,
        group=group,
        activation_function=activation_function,
    ).to(device)

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=w_decay
        )
    elif optimizer_type == "radam":
        optimizer = torch.optim.RAdam(
            model.parameters(), lr=learning_rate, weight_decay=w_decay
        )

    losses_dict = {
        "loss_train": [],
        "mse_train": [],
        "kl_train": [],
        "loss_val": [],
    }

    losses_datasets = {}
    for i in range(len(data.views)):
        losses_datasets[f"Dataset {i}"] = []

    for epoch in range(num_epochs):

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
            n_folds=n_folds,
            batch_size=batch_size,
            model=model,
            optimizer=optimizer,
            alpha_KL=alpha_KL,
            alpha_MSE=alpha_MSE,
            loss_type=loss_type,
        )

        # -- Train Losses (CV + Batch Average)
        losses_dict["loss_train"].append(np.mean(loss_train))
        losses_dict["mse_train"].append(np.mean(mse_train))
        losses_dict["kl_train"].append(np.mean(kl_train))

        # -- Validation Losses (CV + Batch Average)
        losses_dict["loss_val"].append(np.mean(loss_val))

        # -- Train Losses Dataset Specific (CV + Batch Average)
        for i in range(len(data.views)):
            losses_datasets[f"Dataset {i}"].append(np.mean(mse_list[f"Dataset {i}"]))

        print(
            f'Epoch {epoch + 1}/{num_epochs} | Loss (train): {losses_dict["loss_train"][epoch]:.4f} | Loss (val): {losses_dict["loss_val"][epoch]:.4f}'
        )

    plot_losses(losses_dict, losses_datasets, alpha_KL, alpha_MSE)
    predictions(
        model,
        hidden_dim_1,
        hidden_dim_2,
        latent_dim,
        probability,
        group,
        learning_rate,
        n_folds,
        batch_size,
    )


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(32)
    print("### Starting the program ###")

    ##-- Choosing the cell lines to be used in the model --#
    transcriptomics = pd.read_csv(_dfileTranscriptomics, index_col=0)
    proteomics = pd.read_csv(_dfileProteomics, index_col=0)
    metabolomics = pd.read_csv(_dfileMetabolomics, index_col=0)
    methylation = pd.read_csv(_dfileMethylation, index_col=0)
    crisprcas9 = pd.read_csv(_dfileCrisprCas9, index_col=0)
    drugresponse = pd.read_csv(_dfileDrugResponse, index_col=0)

    # Remove cell lines that only have data in one dataset
    cell_methylation = pd.DataFrame(methylation.columns, columns=["cell_lines"])
    cell_methylation["dataset"] = "methylation"
    cell_transcriptomics = pd.DataFrame(transcriptomics.columns, columns=["cell_lines"])
    cell_transcriptomics["dataset"] = "transcriptomics"
    cell_proteomics = pd.DataFrame(proteomics.columns, columns=["cell_lines"])
    cell_proteomics["dataset"] = "proteomics"
    cell_metabolomics = pd.DataFrame(metabolomics.columns, columns=["cell_lines"])
    cell_metabolomics["dataset"] = "metabolomics"
    cell_crisprcas9 = pd.DataFrame(crisprcas9.columns, columns=["cell_lines"])
    cell_crisprcas9["dataset"] = "crisprcas9"
    cell_drugresponse = pd.DataFrame(drugresponse.columns, columns=["cell_lines"])
    cell_drugresponse["dataset"] = "drugresponse"

    missing_viz = pd.merge(
        cell_methylation, cell_transcriptomics, on="cell_lines", how="outer"
    )
    missing_viz = pd.merge(missing_viz, cell_proteomics, on="cell_lines", how="outer")
    missing_viz = pd.merge(missing_viz, cell_metabolomics, on="cell_lines", how="outer")
    missing_viz = pd.merge(missing_viz, cell_crisprcas9, on="cell_lines", how="outer")
    missing_viz = pd.merge(missing_viz, cell_drugresponse, on="cell_lines", how="outer")
    missing_viz.index = missing_viz.cell_lines
    missing_viz = missing_viz.drop(columns=["cell_lines"])
    missing_viz.columns = [
        "Methylation",
        "Transcriptomics",
        "Proteomics",
        "Metabolomics",
        "Crispr",
        "DrugResponse",
    ]

    # Remove cell lines which are only in one dataset
    missing_viz = missing_viz[missing_viz.isna().sum(axis=1) < 5]
    cells = list(missing_viz.index)
    cells.remove("SIDM00189")
    cells.remove("SIDM00650")
    cells = sorted(cells)

    del proteomics
    del methylation
    del crisprcas9
    del drugresponse
    del cell_methylation
    del cell_transcriptomics
    del cell_proteomics
    del cell_metabolomics
    del cell_crisprcas9
    del cell_drugresponse
    del missing_viz

    ##-- Load the Data --#
    omics_db = OmicsData(cells)

    epoch(
        data=omics_db,
        num_epochs=25,
        batch_size=30,
        learning_rate=1e-05,
        n_folds=3,
        hidden_dim_1=0.8,
        hidden_dim_2=0.3,
        latent_dim=50,
        probability=0.5,
        group=15,
        alpha_KL=0.1,
        alpha_MSE=0.9,
        optimizer_type="adam",
        w_decay=1e-05,
        loss_type="mse",
        activation_function=nn.Sigmoid(),
    )
