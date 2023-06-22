import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PhenPred.vae import plot_folder
from PhenPred.vae.Model import MOVE
from torch.utils.data import DataLoader
from PhenPred.vae.Losses import CLinesLosses
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    ShuffleSplit,
)


class CLinesTrain:
    def __init__(
        self,
        data,
        hypers,
        stratify_cv_by=None,
    ):
        self.data = data
        self.losses = []
        self.hypers = hypers
        self.stratify_cv_by = stratify_cv_by
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lrs = [(1, hypers["learning_rate"])]

    def run(self):
        self.training()

        losses_df = self.save_losses()
        self.plot_losses(losses_df)

        self.predictions()

    def initialize_model(self):
        model = MOVE(
            hypers=self.hypers,
            views_sizes={n: v.shape[1] for n, v in self.data.views.items()},
            conditional_size=self.data.labels.shape[1],
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
        for x, y, x_nans in dataloader:
            x = [i.to(self.device) for i in x]
            x_nans = [i.to(self.device) for i in x_nans]
            y = y.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(model.training):
                x_hat, _, mu, log_var = model(x, y)
                loss = model.module.loss(x, x_hat, x_nans, mu, log_var)

                if model.training:
                    loss["total"].backward()
                    optimizer.step()

            if record_losses is not None:
                self.register_loss(
                    loss,
                    record_losses,
                )

    def cv_strategy(self, shuffle_split=False):
        if shuffle_split and self.stratify_cv_by is not None:
            cv = StratifiedShuffleSplit(
                n_splits=self.hypers["n_folds"], test_size=0.1
            ).split(self.data, self.stratify_cv_by.reindex(self.data.samples))
        elif shuffle_split:
            cv = ShuffleSplit(n_splits=self.hypers["n_folds"], test_size=0.1).split(
                self.data
            )
        elif self.stratify_cv_by is not None:
            cv = StratifiedKFold(n_splits=self.hypers["n_folds"], shuffle=True).split(
                self.data, self.stratify_cv_by.reindex(self.data.samples)
            )
        else:
            cv = KFold(n_splits=self.hypers["n_folds"], shuffle=True).split(self.data)

        return cv

    def training(self, cv=None):
        cv = self.cv_strategy() if cv is None else cv

        for cv_idx, (train_idx, test_idx) in enumerate(cv, start=1):
            # Train and Test Data
            data_train = torch.utils.data.Subset(self.data, train_idx)
            dataloader_train = DataLoader(
                data_train, batch_size=self.hypers["batch_size"], shuffle=True
            )

            data_test = torch.utils.data.Subset(self.data, test_idx)
            dataloader_test = DataLoader(
                data_test, batch_size=self.hypers["batch_size"], shuffle=False
            )

            # Initialize Model and Optimizer
            model = self.initialize_model()
            optimizer = CLinesLosses.get_optimizer(model, self.hypers)
            scheduler = CLinesLosses.get_scheduler(optimizer, self.hypers)

            # Train and Test Model
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
                    dataloader_test,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="val",
                    ),
                )

                self.print_losses(cv_idx, epoch)

                if scheduler is not None:
                    scheduler.step(
                        self.get_losses(cv_idx, epoch, "type").loc[
                            "val", "reconstruction"
                        ]
                    )
                    current_lr = optimizer.param_groups[0]["lr"]
                    if current_lr != self.lrs[-1][1]:
                        self.lrs.append((epoch, current_lr))

        return self.get_losses(cv_idx, epoch, "type").loc["val", "reconstruction"]

    def predictions(self):
        imputed_datasets = dict()

        # Data Loader
        data_all = DataLoader(
            self.data, batch_size=self.hypers["batch_size"], shuffle=False
        )

        model = self.initialize_model()
        optimizer = CLinesLosses.get_optimizer(model, self.hypers)

        for _ in range(1, self.hypers["num_epochs"] + 1):
            model.train()
            self.epoch(
                model,
                optimizer,
                data_all,
            )

        # Make predictions and latent spaces
        data_all = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )
        model.eval()
        with torch.no_grad():
            for x, y, _ in data_all:
                x_hat, z, _, _ = model(x, y)

                for name, df in zip(self.data.view_names, x_hat):
                    imputed_datasets[name] = pd.DataFrame(
                        self.data.view_scalers[name].inverse_transform(df.tolist()),
                        index=self.data.samples,
                        columns=self.data.view_feature_names[name],
                    )

                z = pd.DataFrame(
                    z.tolist(),
                    index=self.data.samples,
                    columns=[f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])],
                )

            # Write to file
            for name, df in imputed_datasets.items():
                df.round(5).to_csv(
                    f"{plot_folder}/files/{self.timestamp}_imputed_{name}.csv.gz",
                    compression="gzip",
                )

            z.round(5).to_csv(
                f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz",
                compression="gzip",
            )

    def register_loss(self, loss, extra_fields=None):
        r = {
            k: float(v)
            for k, v in loss.items()
            if type(v) == torch.Tensor and v.numel() == 1
        }

        if "reconstruction_views" in loss:
            for i, v in enumerate(loss["reconstruction_views"]):
                r[f"mse_{self.data.view_names[i]}"] = float(v)

        if extra_fields is not None:
            r.update(extra_fields)

        self.losses.append(r)

    def get_losses(self, cv_idx, epoch_idx, groupby=None):
        l = pd.DataFrame(self.losses).query(f"cv == {cv_idx} & epoch == {epoch_idx}")
        if groupby is not None:
            l = l.groupby(groupby).mean()
        return l

    def print_losses(self, cv_idx, epoch_idx, pbar=None):
        l = self.get_losses(cv_idx, epoch_idx, groupby="type")

        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx:02}, Epoch={epoch_idx:03} Loss (train/val)"
        ptxt += f" | Total={l.loc['train', 'total']:.2f}/{l.loc['val', 'total']:.2f}"

        for k in l.columns:
            if k not in ["cv", "epoch", "type", "total"] and "_" not in k:
                ptxt += f" | {k}={l.loc['train', k]:.2f}/{l.loc['val', k]:.2f}"

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

    def _plot_lr_rates(self, ax):
        for e, lr in self.lrs:
            ax.axvline(e, color="black", linestyle="--", alpha=0.5)
            ax.text(
                e,
                ax.get_ylim()[1],
                f"LR={lr:.0e}",
                ha="center",
                va="top",
                rotation=90,
                fontsize=4,
            )

    def plot_losses(self, losses_df, loss_terms=None, figsize=(3, 2)):
        # Plot total losses
        plot_df = pd.melt(losses_df, id_vars=["epoch", "type"], value_vars="total")

        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="type",
            ax=ax,
        )
        self._plot_lr_rates(ax)
        ax.set(
            title=f"Train and Validation Loss",
            xlabel="Epoch",
            ylabel="Loss",
        )
        ax.legend(
            title="Losses",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        PhenPred.save_figure(
            f"{plot_folder}/losses/{self.timestamp}_train_validation_loss"
        )

        # Plot loss terms
        if loss_terms is None:
            loss_terms = [
                c
                for c in losses_df
                if c not in ["cv", "epoch", "type", "total"] and "_" in c
            ]

        unique_prefix = {v.split("_")[0] for v in loss_terms}
        for prefix in unique_prefix:
            plot_df = pd.melt(
                losses_df,
                id_vars=["epoch", "type"],
                value_vars=[c for c in loss_terms if c.startswith(prefix)],
            )

            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
            sns.lineplot(
                data=plot_df,
                x="epoch",
                y="value",
                hue="variable",
                style="type",
                err_kws=dict(alpha=0.2, lw=0),
                ax=ax,
            )
            self._plot_lr_rates(ax)
            ax.legend(
                title="Losses",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
            ax.set(
                title=f"Total loss",
                xlabel="Epoch",
                ylabel="Loss",
            )
            PhenPred.save_figure(
                f"{plot_folder}/losses/{self.timestamp}_{prefix}_losses"
            )
