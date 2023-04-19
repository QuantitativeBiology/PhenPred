import sys

sys.path.extend(["/home/egoncalves/PhenPred"])

import json
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
from PhenPred.vae.CLinesModel import OMIC_VAE
from PhenPred.vae.CLinesLosses import CLinesLosses
from PhenPred.vae.CLinesDataset import CLinesDataset
from PhenPred.vae import CLinesVAEPlot as ploter
from PhenPred.vae.CLinesDrugResponseBenchmark import DrugResponseBenchmark
from PhenPred.vae.CLinesProteomicsBenchmark import ProteomicsBenchmark


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
    datasets=dict(
        methylation=_data_files["meth_csv_file"],
        transcriptomics=_data_files["gexp_csv_file"],
        proteomics=_data_files["prot_csv_file"],
        metabolomics=_data_files["meta_csv_file"],
        drugresponse=_data_files["dres_csv_file"],
        crisprcas9=_data_files["cris_csv_file"],
    ),
    num_epochs=25,
    learning_rate=1e-4,
    batch_size=32,
    n_folds=3,
    latent_dim=30,
    hidden_dim_1=0.4,
    # hidden_dim_2=0.3,
    probability=0.4,
    group=15,
    alpha_kl=0.1,
    alpha_mse=0.9,
    alpha_c=1,
    optimizer_type="adam",
    w_decay=1e-5,
    loss_type="mse",
    activation_function=nn.Sigmoid(),
)

# Class variables - Torch
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(28)

# Class variables - Misc
_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
_dirPlots = "/home/egoncalves/PhenPred/reports/vae/"


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
        for views, labels in dataloader_train:
            n = views[0].size(0)

            views = [view.to(_device) for view in views]

            # Conditional
            labels = labels.to(_device)

            # Forward pass to get the predictions
            views_hat = model.forward(views, labels)

            # Get last layer of encoder with bottleneck
            h_bottleneck = model.encode(views, labels)

            # Get means and log_vars
            means, log_variances = model.mean_variance(h_bottleneck)
            mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

            # Calculate Losses
            loss, mse, kl = CLinesLosses.loss_function(
                _hyperparameters,
                views,
                views_hat,
                mu_joint,
                logvar_joint,
            )

            # Calculate MSE for each omic
            for i, d in enumerate(data.view_names):
                mse_list[d].append(
                    CLinesLosses.reconstruction_loss(
                        _hyperparameters, views[i], views_hat[i]
                    )
                    / views[i].shape[1]
                    / n
                )

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
            for views, labels in dataloader_test:
                views = [view.to(_device) for view in views]

                # Conditional
                labels = labels.to(_device)

                # Forward pass to get the predictions
                views_hat = model.forward(views, labels)

                # Get last layer of encoder with bottleneck
                h_bottleneck = model.encode(views, labels)

                # Get means and log_vars
                means, log_variances = model.mean_variance(h_bottleneck)
                mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

                # Calculate Losses
                loss, mse, kl = CLinesLosses.loss_function(
                    _hyperparameters,
                    views,
                    views_hat,
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
        data.views,
        _hyperparameters,
        data.conditional,
    ).to(_device)

    optimizer = CLinesLosses.get_optimizer(_hyperparameters, model)

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
            losses_datasets[v_name].append(
                np.mean([v.detach().numpy() for v in mse_list[v_name]])
            )

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

    return model


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
                data.view_scalers[name].inverse_transform(df.tolist()),
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
    clines_db = CLinesDataset(_hyperparameters["datasets"])

    # Write the hyperparameters to json file
    json.dump(
        _hyperparameters,
        open(f"{_dirPlots}/files/{_timestamp}_hyperparameters.json", "w"),
        default=lambda o: "<not serializable>",
        indent=4,
    )
    print("Hyperparameters:\n")
    print(
        json.dumps(
            _hyperparameters,
            sort_keys=True,
            indent=4,
            default=lambda o: "<not serializable>",
        )
    )

    # Run the training loop
    model = epoch(clines_db)

    # Predictions
    predictions(clines_db, model)

    # _timestamp = "2023-04-13_19:21:53"
    # Plot latent spaces
    ploter.plot_latent_spaces(
        _timestamp,
        clines_db.view_names,
        {
            k: _hyperparameters[k]
            for k in [
                "hidden_dim_1",
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

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(_timestamp)
    proteomics_benchmark.run()
