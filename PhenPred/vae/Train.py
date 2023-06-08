from calendar import c
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
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.ModelCVAE import CLinesCVAE
from sklearn.model_selection import KFold, StratifiedKFold


class CLinesTrain:
    def __init__(self, data, hypers, stratify_cv_by=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = data
        self.hypers = hypers

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.losses = []

        self.stratify_cv_by = stratify_cv_by

    def run(self):
        self.training()
        self.predictions()

    def initialize_model(self):
        model = CLinesCVAE(
            views_sizes={n: v.shape[1] for n, v in self.data.views.items()},
            hyper=self.hypers,
            labels_size=self.data.labels_size,
            device=self.device,
        )
        model = nn.DataParallel(model)
        model.to(self.device)
        return model

    def epoch(
        self,
        model,
        optimizer,
        dataloader,
        record_losses=None,
    ):
        for views, classes, views_nans in dataloader:
            views_nans = [~view for view in views_nans]

            covariates, labels = classes
            if self.hypers["covariates"] is None:
                covariates = None

            optimizer.zero_grad()

            with torch.set_grad_enabled(model.training):
                views_hat, mu_joint, logvar_joint, _, _, labels_hat = model(views)

                z_joint = model.module.reparameterize(mu_joint, logvar_joint)

                loss = CLinesLosses.loss_function(
                    hypers=self.hypers,
                    views=views,
                    views_hat=views_hat,
                    means=mu_joint,
                    log_variances=logvar_joint,
                    z_joint=z_joint,
                    views_nans=views_nans,
                    covariates=covariates,
                    labels=labels,
                    labels_hat=labels_hat,
                )

                if model.training:
                    loss["total"].backward()
                    optimizer.step()

            if record_losses is not None:
                self.register_loss(
                    loss,
                    record_losses,
                )

    def cv_strategy(self):
        if self.stratify_cv_by is not None:
            cv = StratifiedKFold(n_splits=self.hypers["n_folds"], shuffle=True).split(
                self.data, self.stratify_cv_by.reindex(self.data.samples)
            )
        else:
            cv = KFold(n_splits=self.hypers["n_folds"], shuffle=True).split(self.data)

        return cv

    def training(self):
        cv = self.cv_strategy()

        for cv_idx, (train_idx, val_idx) in enumerate(cv, start=1):
            # Train and Test Data
            data_train = torch.utils.data.Subset(self.data, train_idx)
            dataloader_train = DataLoader(
                data_train, batch_size=self.hypers["batch_size"], shuffle=True
            )

            data_val = torch.utils.data.Subset(self.data, val_idx)
            dataloader_val = DataLoader(
                data_val, batch_size=len(data_val), shuffle=False
            )

            # Initialize Model and Optimizer
            model = self.initialize_model()
            optimizer = CLinesLosses.get_optimizer(self.hypers, model)

            # Train and Validate Model
            for epoch in range(1, self.hypers["num_epochs"] + 1):
                model.train()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_train,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="train",
                    ),
                )

                model.eval()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_val,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="val",
                    ),
                )

                self.print_losses(cv_idx, epoch)

        losses_df = self.save_losses()
        self.plot_losses(losses_df, self.timestamp)

    def predictions(self):
        latent_spaces = dict()
        imputed_datasets = dict()

        # Data Loader
        data_all = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )

        # Fine-tune model
        model = self.initialize_model()
        optimizer = CLinesLosses.get_optimizer(self.hypers, model)
        for _ in range(1, self.hypers["num_epochs"] + 1):
            model.train()
            self.epoch(
                model,
                optimizer,
                data_all,
            )

        # Make predictions and latent spaces
        model.eval()
        with torch.no_grad():
            for views, _, _ in data_all:
                views = [view.to(self.device) for view in views]

                (
                    views_hat,
                    mu_joint,
                    logvar_joint,
                    mu_views,
                    logvar_views,
                    _,
                ) = model(views)

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

                for name, mus, logvars in zip(
                    self.data.view_names, mu_views, logvar_views
                ):
                    latent_spaces[name] = pd.DataFrame(
                        model.module.reparameterize(mus, logvars).tolist(),
                        index=self.data.samples,
                        columns=[
                            f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])
                        ],
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

    def register_loss(self, loss, extra_fields=None):
        r = {
            "total": float(loss["total"]),
            "mse": float(loss["mse"]),
            "kl": float(loss["kl"]),
            "covariate": float(loss["covariate"]),
            "label": float(loss["label"]),
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
            + f" | Label={l.loc['train', 'label']:.2f}/{l.loc['val', 'label']:.2f}"
        )

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

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
        for ltype in ["mse", "kl"]:
            plot_df = pd.melt(
                losses_df,
                id_vars=["epoch", "type"],
                value_vars=[c for c in losses_df if c.startswith(f"{ltype}_")],
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
            PhenPred.save_figure(f"{plot_folder}/losses/{timestamp}_{ltype}_omics_loss")
