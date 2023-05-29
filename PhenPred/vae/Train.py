# %load_ext autoreload
# %autoreload 2

import os
import sys

proj_dir = "/home/egoncalves/PhenPred"
if not os.path.exists(proj_dir):
    proj_dir = "/Users/emanuel/Projects/PhenPred"
sys.path.extend([proj_dir])

import json
import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.ModelCVAE import CLinesCVAE
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.Dataset import CLinesDataset
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkGenomics import GenomicsBenchmark
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.Hypers import Hypers


# Class variables - Hyperparameters
_hyperparameters = Hypers.read_hyperparameters()


class CLinesTrain:
    def __init__(self, data, hypers):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(28)

        self.data = data
        self.hypers = hypers

        self.model = CLinesCVAE(
            self.data.views,
            self.hypers,
            self.data.conditional if self.hypers["conditional"] else None,
        ).to(self.device)

        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = CLinesLosses.get_optimizer(self.hypers, self.model)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    def run(self):
        self.epoch()
        self.predictions()

    def epoch(self):
        losses_dict = {
            "train_total": [],
            "train_mse": [],
            "train_kl": [],
            "train_cov": [],
            "train_mse_views": {d: [] for d in self.data.view_names},
            "val_total": [],
            "val_mse": [],
            "val_kl": [],
            "val_cov": [],
            "val_mse_views": {d: [] for d in self.data.view_names},
        }

        for epoch in range(self.hypers["num_epochs"]):
            # -- Cross Validation
            train_loss, val_loss = self.cross_validation()

            # -- Train Losses (CV + Batch Average)
            losses_dict["train_total"].append(np.nanmean(train_loss["total"]))
            losses_dict["train_mse"].append(np.nanmean(train_loss["mse"]))
            losses_dict["train_kl"].append(np.nanmean(train_loss["kl"]))
            losses_dict["train_cov"].append(np.nanmean(train_loss["cov"]))

            # -- Validation Losses (CV + Batch Average)
            losses_dict["val_total"].append(np.nanmean(val_loss["total"]))
            losses_dict["val_mse"].append(np.nanmean(val_loss["mse"]))
            losses_dict["val_kl"].append(np.nanmean(val_loss["kl"]))
            losses_dict["val_cov"].append(np.nanmean(val_loss["cov"]))

            # -- Train Losses Dataset Specific (CV + Batch Average)
            for v in self.data.view_names:
                losses_dict["train_mse_views"][v].append(
                    np.mean(train_loss["mse_views"][v])
                )
                losses_dict["val_mse_views"][v].append(
                    np.mean(val_loss["mse_views"][v])
                )

            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch + 1} Loss (train / val)"
                + f"| Total: {losses_dict['train_total'][epoch]:.3f} / {losses_dict['val_total'][epoch]:.3f}"
                + f"| MSE: {losses_dict['train_mse'][epoch]:.3f} / {losses_dict['val_mse'][epoch]:.3f}"
                + f"| KL: {losses_dict['train_kl'][epoch]:.3f} / {losses_dict['val_kl'][epoch]:.3f}"
                + f"| Cov: {losses_dict['train_cov'][epoch]:.3f} / {losses_dict['val_cov'][epoch]:.3f}"
            )

        CLinesLosses.plot_losses(
            losses_dict,
            self.hypers["beta"],
            self.timestamp,
        )

    def cross_validation(self):
        train_loss = dict(
            total=[],
            mse=[],
            kl=[],
            cov=[],
            mse_views={d: [] for d in self.data.view_names},
        )
        val_loss = dict(
            total=[],
            mse=[],
            kl=[],
            cov=[],
            mse_views={d: [] for d in self.data.view_names},
        )

        cv = KFold(n_splits=self.hypers["n_folds"], shuffle=True)

        for train_idx, val_idx in cv.split(self.data):
            # Train Data
            data_train = torch.utils.data.Subset(self.data, train_idx)
            dataloader_train = DataLoader(
                data_train, batch_size=self.hypers["batch_size"], shuffle=True
            )

            # Validation Data
            data_test = torch.utils.data.Subset(self.data, val_idx)
            dataloader_test = DataLoader(
                data_test, batch_size=self.hypers["batch_size"], shuffle=False
            )

            # --- TRAINING LOOP
            self.model.train()

            # dataloader train is divided into batches
            for views, labels, views_nans in dataloader_train:
                views = [view.to(self.device) for view in views]
                views_nans = [~view.to(self.device) for view in views_nans]

                # Covariates
                labels = labels.to(self.device)

                # Forward pass to get the predictions
                views_hat, mu_joint, logvar_joint = self.model.forward(views, labels)

                # Sample from joint latent space
                z_joint = self.model.module.reparameterize(mu_joint, logvar_joint)

                # Calculate Losses
                loss = CLinesLosses.loss_function(
                    hypers=self.hypers,
                    views=views,
                    views_hat=views_hat,
                    means=mu_joint,
                    log_variances=logvar_joint,
                    z_joint=z_joint,
                    views_nans=views_nans,
                    covariates=None if self.hypers["covariates"] is None else labels,
                )

                # Store values
                for k, v in loss["mse_views"].items():
                    train_loss["mse_views"][k].append(v.cpu().detach().numpy())

                train_loss["total"].append(loss["total"].cpu().detach().numpy())
                train_loss["mse"].append(loss["mse"].cpu().detach().numpy())
                train_loss["kl"].append(loss["kl"].cpu().detach().numpy())
                train_loss["cov"].append(loss["covariate"].cpu().detach().numpy())

                with torch.autograd.set_detect_anomaly(True):
                    self.optimizer.zero_grad()
                    loss["total"].backward()
                    self.optimizer.step()

            # --- VALIDATION LOOP
            self.model.eval()

            with torch.no_grad():
                for views, labels, views_nans in dataloader_test:
                    views = [view.to(self.device) for view in views]
                    views_nans = [~view.to(self.device) for view in views_nans]

                    # covariates
                    labels = labels.to(self.device)

                    # Forward pass to get the predictions
                    views_hat, mu_joint, logvar_joint = self.model.forward(views)

                    # Sample from joint latent space
                    z_joint = self.model.module.reparameterize(mu_joint, logvar_joint)

                    # Calculate Losses
                    loss = CLinesLosses.loss_function(
                        hypers=self.hypers,
                        views=views,
                        views_hat=views_hat,
                        means=mu_joint,
                        log_variances=logvar_joint,
                        z_joint=z_joint,
                        views_nans=views_nans,
                        covariates=None
                        if self.hypers["covariates"] is None
                        else labels,
                    )

                    # Store values
                    for k, v in loss["mse_views"].items():
                        val_loss["mse_views"][k].append(v.cpu().detach().numpy())

                    val_loss["total"].append(loss["total"].cpu().detach().numpy())
                    val_loss["mse"].append(loss["mse"].cpu().detach().numpy())
                    val_loss["kl"].append(loss["kl"].cpu().detach().numpy())
                    val_loss["cov"].append(loss["covariate"].cpu().detach().numpy())

        return train_loss, val_loss

    def predictions(self):
        omics_dataloader = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )

        # Dataframes
        latent_spaces = dict()
        imputed_datasets = dict()

        # Make predictions and latent spaces
        for views, labels, views_nans in omics_dataloader:
            views = [view.to(self.device) for view in views]
            views_nans = [~view.to(self.device) for view in views_nans]

            # Forward pass to get the predictions
            views_hat, mu_joint, logvar_joint = self.model.forward(views, labels)

            for name, df in zip(self.data.view_names, views_hat):
                imputed_datasets[name] = pd.DataFrame(
                    self.data.view_scalers[name].inverse_transform(df.tolist()),
                    index=self.data.samples,
                    columns=self.data.view_feature_names[name],
                )

            # Create Latent Spaces
            latent_spaces["joint"] = pd.DataFrame(
                self.model.module.reparameterize(mu_joint, logvar_joint).tolist(),
                index=self.data.samples,
                columns=[f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])],
            )

        # Write to file
        for name, df in imputed_datasets.items():
            df.round(5).to_csv(
                f"{plot_folder}/files/{self.timestamp}_imputed_{name}.csv.gz",
                compression="gzip",
            )

        for name, df in latent_spaces.items():
            df.round(5).to_csv(
                f"{plot_folder}/files/{self.timestamp}_latent_{name}.csv.gz",
                compression="gzip",
            )


if __name__ == "__main__":
    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        datasets=_hyperparameters["datasets"],
        feature_miss_rate_thres=_hyperparameters["feature_miss_rate_thres"],
        covariates=_hyperparameters["covariates"],
    )
    clines_db.plot_samples_overlap()
    clines_db.plot_datasets_missing_values()

    # Train and predictions
    # train.timestamp = "2023-05-18_19:49:05"
    train = CLinesTrain(clines_db, _hyperparameters)
    train.run()

    # Plot latent spaces
    CLinesLosses.plot_latent_spaces(
        timestamp=train.timestamp,
        view_names=[],
        data=clines_db,
        markers=pd.concat(
            [
                clines_db.dfs["transcriptomics"][["VIM", "CDH1"]],
                clines_db.dfs["metabolomics"][["1-methylnicotinamide"]],
            ],
            axis=1,
        ),
    )

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(train.timestamp)
    dres_benchmark.run()

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(train.timestamp)
    proteomics_benchmark.run()

    # Run CRISPR benchmark
    crispr_benchmark = CRISPRBenchmark(train.timestamp, clines_db)
    crispr_benchmark.run()

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(train.timestamp, clines_db)
    latent_benchmark.run()

    # Write the hyperparameters to json file
    json.dump(
        _hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
