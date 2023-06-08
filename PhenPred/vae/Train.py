import torch
import PhenPred
import contextlib
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PhenPred.vae import plot_folder
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import KFold
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.ModelCVAE import CLinesCVAE


class CLinesTrain:
    def __init__(self, data, hypers, save_best_model=False):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data & Hyperparameters
        self.data = data
        self.hypers = hypers

        # Timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Losses
        self.losses = []

    def run(self):
        self.training()
        self.predictions()

    def register_loss(self, loss, extra_fields=None):
        r = {
            "total": float(loss["total"]),
            "mse": float(loss["mse"]),
            "kl": float(loss["kl"]),
            "covariate": float(loss["covariate"]),
        }

        for k, v in loss["mse_views"].items():
            r[f"mse_{k}"] = float(v)

        for k, v in loss["kl_views"].items():
            r[f"kl_{k}"] = float(v)

        if extra_fields is not None:
            r.update(extra_fields)

        self.losses.append(r)

    def print_losses(self, cv_idx, epoch_idx, pbar=None):
        l = pd.DataFrame(self.losses).query(f"cv == {cv_idx} & epoch == {epoch_idx}")
        l = l.groupby("type").mean()

        ptxt = (
            f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx}, Epoch={epoch_idx} Loss (train/val)"
            + f" | Total={l.loc['train', 'total']:.2f}/{l.loc['val', 'total']:.2f}"
            + f" | MSE={l.loc['train', 'mse']:.2f}/{l.loc['val', 'mse']:.2f}"
            + f" | KL={l.loc['train', 'kl']:.2f}/{l.loc['val', 'kl']:.2f}"
            + f" | Cov={l.loc['train', 'covariate']:.2f}/{l.loc['val', 'covariate']:.2f}"
        )

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

    def initialize_model(self):
        model = CLinesCVAE(
            {n: v.shape[1] for n, v in self.data.views.items()},
            self.hypers,
            self.data.conditional if self.hypers["conditional"] else None,
            device=self.device,
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.to(self.device)

        return model

    def train_epoch(
        self,
        optimizer,
        model,
        datatloader,
        register_losses=True,
        register_losses_fields=dict(),
    ):
        # When evaluating, disable gradient computation to reduce memory usage
        with torch.set_grad_enabled(True) if model.training else torch.no_grad():
            # Dataloader train is divided into batches
            for views, labels, views_nans in datatloader:
                views_nans = [~view for view in views_nans]

                # Clear gradients
                if model.training:
                    optimizer.zero_grad()

                # Forward pass to get the predictions
                views_hat, mu_joint, logvar_joint, _, _ = model.forward(views)

                # Sample from joint latent space
                z_joint = model.module.reparameterize(mu_joint, logvar_joint)

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

                del views_hat, mu_joint, logvar_joint, z_joint

                # Backward pass and optimization
                if model.training:
                    loss["total"].backward()
                    optimizer.step()

                # Register losses
                if register_losses:
                    self.register_loss(
                        loss,
                        register_losses_fields,
                    )

                del loss

    def training(self):
        # Cross Validation
        cv = KFold(n_splits=self.hypers["n_folds"], shuffle=True)

        for cv_idx, (train_idx, val_idx) in enumerate(cv.split(self.data), start=1):
            # Train Data
            data_train = torch.utils.data.Subset(self.data, train_idx)
            dataloader_train = DataLoader(
                data_train, batch_size=self.hypers["batch_size"], shuffle=True
            )

            # Validation Data
            data_val = torch.utils.data.Subset(self.data, val_idx)
            dataloader_val = DataLoader(
                data_val, batch_size=self.hypers["batch_size"], shuffle=False
            )

            # Initialize Model
            model = self.initialize_model()

            # Initialize Optimizer
            optimizer = CLinesLosses.get_optimizer(self.hypers, model)

            # Train and Validate Model
            pbar = tqdm(range(1, self.hypers["num_epochs"] + 1))
            for epoch in pbar:
                # Train
                model.train()

                self.train_epoch(
                    optimizer,
                    model,
                    dataloader_train,
                    True,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="train",
                    ),
                )

                # Validate
                model.eval()

                self.train_epoch(
                    optimizer,
                    model,
                    dataloader_val,
                    True,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="val",
                    ),
                )

                # Print losses
                self.print_losses(cv_idx, epoch, pbar)

        losses_df = self.save_losses()
        self.plot_losses(losses_df, self.timestamp)

    def predictions(self):
        if self.best_model is None:
            raise Exception("Model not trained. Run training first.")

        latent_spaces = dict()
        imputed_datasets = dict()

        # Initialize Best Model
        model = self.initialize_model()

        # Data Loader
        data_all = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )

        # fine-tune model
        if self.hypers["fine_tune"]:
            model.train()
            optimizer = CLinesLosses.get_optimizer(self.hypers, model)
            for _ in range(1, self.hypers["num_epochs"] + 1):
                self.train_epoch(
                    optimizer,
                    model,
                    data_all,
                    False,
                )

        # Make predictions and latent spaces
        model.eval()
        for views, _, _ in data_all:
            views = [view.to(self.device) for view in views]

            (
                views_hat,
                mu_joint,
                logvar_joint,
                mu_views,
                logvar_views,
            ) = model.forward(views)

            for name, df in zip(self.data.view_names, views_hat):
                imputed_datasets[name] = pd.DataFrame(
                    self.data.view_scalers[name].inverse_transform(df.tolist()),
                    index=self.data.samples,
                    columns=self.data.view_feature_names[name],
                )

            latent_spaces["joint"] = pd.DataFrame(
                model.module.reparameterize(mu_joint, logvar_joint).tolist(),
                index=self.data.samples,
                columns=[f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])],
            )

            for name, mus, logvars in zip(self.data.view_names, mu_views, logvar_views):
                latent_spaces[name] = pd.DataFrame(
                    model.module.reparameterize(mus, logvars).tolist(),
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

    @staticmethod
    def plot_losses(losses_df, timestamp=""):
        # Plot total losses
        plot_df = pd.melt(losses_df, id_vars=["epoch", "type"], value_vars="total")

        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="type",
            ax=ax,
        )
        ax.set(
            title=f"Train and Validation Loss",
            xlabel="Epoch",
            ylabel="Loss",
        )
        legend_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            legend_handles,
            ["Train Loss", "Validation Loss"],
            title="Losses",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        PhenPred.save_figure(f"{plot_folder}/losses/{timestamp}_train_validation_loss")

        # Plot loss terms
        plot_df = pd.melt(
            losses_df, id_vars=["epoch", "type"], value_vars=["mse", "kl", "covariate"]
        )

        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="variable",
            style="type",
            ls="--",
            ax=ax,
        )
        ax.set(
            title=f"Total loss",
            xlabel="Epoch",
            ylabel="Loss",
        )
        PhenPred.save_figure(f"{plot_folder}/losses/{timestamp}_reconst_reg_loss")

        # Plot losses views
        plot_df = pd.melt(
            losses_df,
            id_vars=["epoch", "type"],
            value_vars=[c for c in losses_df if c.startswith("mse_")],
        )

        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="variable",
            style="type",
            ax=ax,
        )
        ax.set(xlabel="Epoch", ylabel="Reconstruction Loss")
        PhenPred.save_figure(f"{plot_folder}/losses/{timestamp}_reconst_omics_loss")
