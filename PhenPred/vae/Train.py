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
import torch.nn.functional as F
from datetime import datetime
from PhenPred.vae.Hypers import Hypers
from sklearn.model_selection import KFold
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.ModelCVAE import CLinesCVAE
from torch.utils.data import DataLoader, Dataset
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkGenomics import GenomicsBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


# Class variables - Hyperparameters
_hyperparameters = Hypers.read_hyperparameters()


class CLinesTrain:
    def __init__(self, data, hypers, save_best_model=False):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(28)

        # Data & Hyperparameters
        self.data = data
        self.hypers = hypers

        # Timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Losses
        self.losses = []

        # Best model
        self.best_loss = np.inf
        self.best_model = None
        self.save_best_model = save_best_model
        self.best_model_path = f"{plot_folder}/files/{self.timestamp}_model.model"

    @staticmethod
    def extract_value(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, list):
            return np.array(x)
        else:
            return x

    def register_loss(self, loss, extra_fields=None):
        r = {
            "total": self.extract_value(loss["total"]),
            "mse": self.extract_value(loss["mse"]),
            "kl": self.extract_value(loss["kl"]),
            "covariate": self.extract_value(loss["covariate"]),
        }

        for k, v in loss["mse_views"].items():
            r[f"mse_{k}"] = self.extract_value(v)

        if extra_fields is not None:
            r.update(extra_fields)

        self.losses.append(r)

    def print_losses(self, cv_idx, epoch_idx, train_loss, val_loss):
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] {cv_idx + 1}-fold, Epoch {epoch_idx + 1} Loss (train / val)"
            + f"| Total: {train_loss['total']:.3f} / {val_loss['total']:.3f}"
            + f"| MSE: {train_loss['mse']:.3f} / {val_loss['mse']:.3f}"
            + f"| KL: {train_loss['kl']:.3f} / {val_loss['kl']:.3f}"
            + f"| Cov: {train_loss['covariate']:.3f} / {val_loss['covariate']:.3f}"
        )

    def save_losses(self):
        df = pd.DataFrame(self.losses)
        df.to_csv(
            os.path.join(
                plot_folder,
                f"files/{self.timestamp}_losses.csv",
            ),
            index=False,
        )

    def run(self):
        self.training()
        self.predictions()

    def training(self):
        # Cross Validation
        cv = KFold(n_splits=self.hypers["n_folds"], shuffle=True)

        for cv_idx, (train_idx, val_idx) in enumerate(cv.split(self.data)):
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

            # Initialize Model
            model = CLinesCVAE(
                self.data.views,
                self.hypers,
                self.data.conditional if self.hypers["conditional"] else None,
            ).to(self.device)
            model = nn.DataParallel(model)
            model.to(self.device)

            # Initialize Optimizer
            optimizer = CLinesLosses.get_optimizer(self.hypers, model)

            # Train and Validate Model
            for epoch in range(self.hypers["num_epochs"]):
                # --- TRAINING
                model.train()

                # Dataloader train is divided into batches
                for views, labels, views_nans in dataloader_train:
                    views = [view.to(self.device) for view in views]
                    views_nans = [~view.to(self.device) for view in views_nans]

                    # Covariates
                    labels = labels.to(self.device)

                    # Forward pass to get the predictions
                    views_hat, mu_joint, logvar_joint, _, _ = model.forward(views)

                    # Sample from joint latent space
                    z_joint = model.module.reparameterize(mu_joint, logvar_joint)

                    # Calculate Losses
                    loss_train = CLinesLosses.loss_function(
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

                    # Zero gradients + Backpropagation + Optimize
                    optimizer.zero_grad()
                    loss_train["total"].backward()
                    optimizer.step()

                    # Store losses values
                    self.register_loss(
                        loss_train,
                        extra_fields={
                            "cv": cv_idx,
                            "epoch": epoch,
                            "type": "train",
                        },
                    )

                # --- VALIDATION
                model.eval()

                # Disable gradient computation and reduce memory consumption.
                with torch.no_grad():
                    for views, labels, views_nans in dataloader_test:
                        views = [view.to(self.device) for view in views]
                        views_nans = [~view.to(self.device) for view in views_nans]

                        # covariates
                        labels = labels.to(self.device)

                        # Forward pass to get the predictions
                        views_hat, mu_joint, logvar_joint, _, _ = model.forward(views)

                        # Sample from joint latent space
                        z_joint = model.module.reparameterize(mu_joint, logvar_joint)

                        # Calculate Losses
                        loss_val = CLinesLosses.loss_function(
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
                        self.register_loss(
                            loss_val,
                            extra_fields={
                                "cv": cv_idx,
                                "epoch": epoch,
                                "type": "val",
                            },
                        )

                # --- Print Epoch Losses
                self.print_losses(cv_idx, epoch, loss_train, loss_val)

                # --- Save Best Model
                if loss_val["total"] < self.best_loss:
                    self.best_vloss = loss_val["total"]
                    self.best_model = model.state_dict()

        # --- Save Best Model
        if self.save_best_model:
            torch.save(self.best_model, self.best_model_path)

        self.save_losses()

        CLinesLosses.plot_losses(
            self.losses,
            self.hypers["beta"],
            self.timestamp,
        )

    def predictions(self):
        if self.best_model is None:
            raise Exception("Model not trained. Run training first.")

        # Initialize Model
        model = CLinesCVAE(
            self.data.views,
            self.hypers,
            self.data.conditional if self.hypers["conditional"] else None,
        ).to(self.device)
        model = nn.DataParallel(model)
        model.to(self.device)

        # Load best model
        if self.save_best_model:
            model.load_state_dict(torch.load(self.best_model_path))
        else:
            model.load_state_dict(self.best_model)

        # Data Loader
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
            (
                views_hat,
                mu_joint,
                logvar_joint,
                mu_views,
                logvar_views,
            ) = self.model.forward(views)

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

            for name, mus, logvars in zip(self.data.view_names, mu_views, logvar_views):
                latent_spaces[name] = pd.DataFrame(
                    self.model.module.reparameterize(mus, logvars).tolist(),
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
    latent_benchmark.plot_latent_spaces(
        view_names=[],
        markers=pd.concat(
            [
                clines_db.dfs["transcriptomics"][["VIM", "CDH1"]],
                clines_db.dfs["metabolomics"][["1-methylnicotinamide"]],
                latent_benchmark.covariates["drug_responses"],
            ],
            axis=1,
        ),
    )

    # Write the hyperparameters to json file
    json.dump(
        _hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
