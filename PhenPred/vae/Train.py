import os
import torch
import PhenPred
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from PhenPred.vae import plot_folder
from PhenPred.vae.Model import MOVE
from PhenPred.vae.ModelGMVAE import GMVAE
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
        early_stop_patience=20,
        gmvae_args_dict=None,
    ):
        self.data = data
        self.losses = []
        self.hypers = hypers
        self.stratify_cv_by = stratify_cv_by
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lrs = [(1, hypers["learning_rate"])]
        self.early_stop_patience = early_stop_patience
        if gmvae_args_dict is not None:
            self.gmvae_args_dict = gmvae_args_dict
            self.gmvae_args_dict["gumbel_temp"] = self.gmvae_args_dict["init_temp"]

    def run(self, run_timestamp=None):
        if run_timestamp is not None:
            self.timestamp = run_timestamp
            return

        self.training()

        losses_df = self.save_losses()
        self.plot_losses(losses_df)

        self.predictions()

        if self.hypers["save_model"]:
            self.save_model()

    def initialize_model(self):
        views_sizes = {n: v.sum() for n, v in self.data.features_mask.items()}
        views_sizes_full = None

        if self.hypers["filtered_encoder_only"]:
            views_sizes_full = {n: v.shape[1] for n, v in self.data.views.items()}

        assert self.hypers["model"] in ["MOVE", "GMVAE"], "Invalid model"
        if self.hypers["model"] == "MOVE":
            model = MOVE(
                hypers=self.hypers,
                views_sizes=views_sizes,
                conditional_size=self.data.labels.shape[1],
                views_sizes_full=views_sizes_full,
            )
        else:
            model = GMVAE(
                views_sizes={n: v.shape[1] for n, v in self.data.views.items()},
                hypers=self.hypers,
                k=self.gmvae_args_dict["k"],
                views_logits=self.hypers["views_logits"],
                conditional_size=self.data.labels.shape[1]
                if self.hypers["use_conditional_gmvae"]
                else 0,
                views_sizes_full=views_sizes_full,
            )

        model = nn.DataParallel(model)

        return model

    def epoch(
        self,
        model,
        optimizer,
        dataloader,
        record_losses=None,
    ):
        for data in dataloader:
            x, y, x_nans, x_mask = data

            x = [m.to(self.device) for m in x]
            x_nans = [m.to(self.device) for m in x_nans]

            x_masked = [m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x)]
            x_nans_masked = [
                m[:, x_mask[i][0]].to(self.device) for i, m in enumerate(x_nans)
            ]

            y = y.to(self.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(model.training):
                if self.hypers["model"] == "MOVE":
                    out_net = model(x_masked, y)
                else:
                    out_net = model(
                        x,
                        self.gmvae_args_dict["gumbel_temp"],
                        self.gmvae_args_dict["hard_gumbel"],
                        y,
                    )

                if self.hypers["filtered_encoder_only"]:
                    # if filtered_encoder_only, use all data for loss
                    loss = model.module.loss(x, x_nans, out_net)
                else:
                    # otherwise, use only filtered data for loss
                    loss = model.module.loss(x_masked, x_nans_masked, out_net)

                if model.training:
                    loss["total"].backward()
                    optimizer.step()

            if record_losses is not None:
                self.register_loss(loss, record_losses)
            else:
                self.print_single_loss(loss)

    def cv_strategy(self, shuffle_split=False):
        if shuffle_split and self.stratify_cv_by is not None:
            cv = StratifiedShuffleSplit(
                n_splits=self.hypers["n_folds"],
                test_size=0.1,
                random_state=42,
            ).split(self.data, self.stratify_cv_by.reindex(self.data.samples))
        elif shuffle_split:
            cv = ShuffleSplit(
                n_splits=self.hypers["n_folds"], test_size=0.1, random_state=42
            ).split(self.data)
        elif self.stratify_cv_by is not None:
            cv = StratifiedKFold(
                n_splits=self.hypers["n_folds"], shuffle=True, random_state=42
            ).split(self.data, self.stratify_cv_by.reindex(self.data.samples))
        else:
            cv = KFold(
                n_splits=self.hypers["n_folds"], shuffle=True, random_state=42
            ).split(self.data)

        return cv

    def training(self, cv=None):
        cv_idx, epoch = 0, 0

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

            model.to(self.device)

            # Train and Test Model
            loss_previous, loss_counter = None, 0
            for epoch in range(1, self.hypers["num_epochs"] + 1):
                # Train
                model.train()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_train,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="train",
                        lr=optimizer.param_groups[0]["lr"],
                    ),
                )

                # Test
                model.eval()
                self.epoch(
                    model,
                    optimizer,
                    dataloader_test,
                    dict(
                        cv=cv_idx,
                        epoch=epoch,
                        type="val",
                        lr=optimizer.param_groups[0]["lr"],
                    ),
                )

                self.print_losses(cv_idx, epoch)

                # Early Stopping
                loss_current = self.get_losses(cv_idx, epoch, "type").loc[
                    "val", "reconstruction"
                ]
                loss_current_total = self.get_losses(cv_idx, epoch, "type").loc[
                    "val", "total"
                ]

                if not (np.isfinite(loss_current) or np.isfinite(loss_current_total)):
                    warnings.warn(f"NaN or Inf loss at cv {cv_idx}, epoch {epoch}.")
                    return np.nan

                elif loss_previous is None:
                    loss_previous = loss_current

                elif round(loss_current, 2) < round(loss_previous, 2):
                    loss_counter = 0
                    loss_previous = loss_current

                else:
                    loss_counter += 1

                if loss_counter >= self.early_stop_patience:
                    warnings.warn(f"Early stopping at cv {cv_idx}, epoch {epoch}.")
                    break

                # Learning rate scheduler
                if scheduler is not None:
                    self.update_learning_rate(scheduler, optimizer, loss_current, epoch)

        return self.get_losses(cv_idx, epoch, "type").loc["val", "reconstruction"]

    def update_learning_rate(self, scheduler, optimizer, loss_current, epoch):
        scheduler.step(loss_current)

        current_lr = optimizer.param_groups[0]["lr"]

        if round(current_lr, 4) < round(self.lrs[-1][1], 4):
            self.lrs.append((epoch, current_lr))

    def predictions(self, n_epochs=None):
        imputed_datasets = dict()

        n_epochs = self.hypers["num_epochs"] if n_epochs is None else n_epochs

        # Data Loader
        data_all = DataLoader(
            self.data, batch_size=len(self.data.samples), shuffle=False
        )

        self.model = self.initialize_model()
        optimizer = CLinesLosses.get_optimizer(self.model, self.hypers)

        for e in range(1, n_epochs + 1):
            self.model.train()
            print(f"Epoch {e:03}")
            self.epoch(
                self.model,
                optimizer,
                data_all,
            )

        # Make predictions and latent spaces
        self.model.eval()
        with torch.no_grad():
            for data in data_all:
                x, y, _, x_mask = data

                x_masked = [m[:, x_mask[i][0]] for i, m in enumerate(x)]

                if self.hypers["model"] == "MOVE":
                    out_net = self.model(x_masked, y)
                else:
                    out_net = self.model(
                        x,
                        self.gmvae_args_dict["gumbel_temp"],
                        self.gmvae_args_dict["hard_gumbel"],
                        y,
                    )

                x_hat = out_net["x_hat"]
                z = out_net["z"]

                for name, df in zip(self.data.view_names, x_hat):
                    imputed_datasets[name] = pd.DataFrame(
                        self.data.view_scalers[name].inverse_transform(df.tolist()),
                        index=self.data.samples,
                        columns=self.data.view_feature_names[name],
                    )

                z = pd.DataFrame(z.tolist(), index=self.data.samples)
                z.columns = [f"Latent_{i+1}" for i in range(z.shape[1])]

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

    def load_vae_reconstructions(self, mode="nans_only", dfs=None):
        """
        Load imputed data and latent space from files. "nans_only" mode, original
        measurements are mantained and only NaNs are imputed. "all" mode all
        data is imputed.

        Parameters
        ----------
        mode : str, optional
            Loading mode of imputed data, by default "nans_only"

        Returns
        -------
        dict
            Dictionary of imputed dataframes
            pandas.DataFrame
                Latent space

        Raises
        ------
        ValueError
            If mode is not "nans_only" or "all"

        """

        if mode not in ["nans_only", "all"]:
            raise ValueError(f"Invalid mode {mode}")

        if dfs is None:
            dfs = self.data.dfs

        dfs_imputed = {}
        for n in dfs:
            df_file = f"{plot_folder}/files/{self.timestamp}_imputed_{n}.csv.gz"

            if not os.path.isfile(df_file):
                continue

            df_imputed = pd.read_csv(df_file, index_col=0)

            if mode == "nans_only":
                df_imputed = self.data.dfs[n].combine_first(df_imputed)

            dfs_imputed[n] = df_imputed

        # Load latent space
        joint_latent = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz", index_col=0
        )

        return dfs_imputed, joint_latent

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

    def print_single_loss(self, loss_dict, pbar=None):
        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] Loss "
        ptxt += f" | Total={loss_dict['total']:.2f}"

        for k in loss_dict:
            if k not in ["cv", "epoch", "type", "total", "lr"] and "_" not in k:
                ptxt += f" | {k}={loss_dict[k]:.2f}"

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def print_losses(self, cv_idx, epoch_idx, pbar=None):
        l = self.get_losses(cv_idx, epoch_idx, groupby="type")

        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx:02}, Epoch={epoch_idx:03} Loss (train/val)"
        ptxt += f" | Total={l.loc['train', 'total']:.2f}/{l.loc['val', 'total']:.2f}"

        for k in l.columns:
            if k not in ["cv", "epoch", "type", "total", "lr"] and "_" not in k:
                ptxt += f" | {k}={l.loc['train', k]:.2f}/{l.loc['val', k]:.2f}"

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

    def save_model(self):
        if self.model is None:
            warnings.warn("No model to save. Run predictions first.")
        else:
            torch.save(
                self.model.state_dict(),
                f"{plot_folder}/files/{self.timestamp}_model.pt",
            )

    def load_model(self):
        model_path = f"{plot_folder}/files/{self.timestamp}_model.pt"

        if not os.path.isfile(model_path):
            warnings.warn("No model to load.")
            return

        self.model = self.initialize_model()
        self.model.load_state_dict(
            torch.load(
                f"{plot_folder}/files/{self.timestamp}_model.pt",
            )
        )

    def _plot_lr_rates(self, ax):
        for e, lr in self.lrs:
            ax.axvline(e, color="black", linestyle="--", alpha=0.5, lw=0.3)
            ax.text(
                e,
                ax.get_ylim()[1],
                f"LR={lr:.0e}",
                ha="left",
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
            errorbar=("ci", 99),
            err_kws=dict(alpha=0.2, lw=0),
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
                if c not in ["cv", "epoch", "type", "total", "lr"] and "_" in c
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
                errorbar=("ci", 99),
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
