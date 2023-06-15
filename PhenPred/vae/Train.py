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
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator
from PhenPred.vae.Losses import CLinesLosses
from PhenPred.vae.ModelCVAE import CLinesCVAE
from PhenPred.vae.ModelGMVAE import CLinesGMVAE
from sklearn.model_selection import KFold, StratifiedKFold


class CLinesTrain:
    def __init__(
        self,
        data,
        hypers,
        stratify_cv_by=None,
        k=2,
        init_temp=1.0,
        decay_temp=1.0,
        hard_gumbel=0,
        min_temp=0.5,
        decay_temp_rate=0.013862944,
    ):
        self.data = data
        self.losses = []
        self.hypers = hypers
        self.stratify_cv_by = stratify_cv_by
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k = k

        # gumbel
        self.init_temp = init_temp
        self.decay_temp = decay_temp
        self.hard_gumbel = hard_gumbel
        self.min_temp = min_temp
        self.decay_temp_rate = decay_temp_rate
        self.gumbel_temp = self.init_temp

    def run(self):
        self.training()
        self.predictions()

    def initialize_model(self):
        model = CLinesGMVAE(
            views_sizes={n: v.shape[1] for n, v in self.data.views.items()},
            hypers=self.hypers,
            k=self.k,
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
            views_nans = [~v for v in views_nans]

            optimizer.zero_grad()

            with torch.set_grad_enabled(model.training):
                out_net = model(views, self.gumbel_temp, self.hard_gumbel)

                loss = CLinesLosses.unlabeled_loss(
                    views=views,
                    out_net=out_net,
                    views_mask=views_nans,
                    rec_type=self.hypers["reconstruction_loss"],
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

                # decay gumbel temperature
                if self.decay_temp == 1:
                    self.gumbel_temp = np.maximum(
                        self.init_temp * np.exp(-self.decay_temp_rate * epoch),
                        self.min_temp,
                    )

        losses_df = self.save_losses()
        self.plot_losses(losses_df)

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
            for views, classes, _ in data_all:
                labels = None if self.hypers["label"] is None else classes[1]

                out_net = model(views)

                for name, df in zip(self.data.view_names, out_net["views_hat"]):
                    imputed_datasets[name] = pd.DataFrame(
                        self.data.view_scalers[name].inverse_transform(df.tolist()),
                        index=self.data.samples,
                        columns=self.data.view_feature_names[name],
                    )

                latent_spaces["joint"] = pd.DataFrame(
                    out_net["z"].tolist(),
                    index=self.data.samples,
                    columns=[f"Latent_{i+1}" for i in range(self.hypers["latent_dim"])],
                )

                for name, view_z in zip(self.data.view_names, out_net["views_z"]):
                    latent_spaces[name] = pd.DataFrame(
                        view_z.tolist(),
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
            k: float(v)
            for k, v in loss.items()
            if type(v) == torch.Tensor and v.numel() == 1
        }

        if "mse_views" in loss:
            for k, v in loss["mse_views"].items():
                r[f"mse_{k}"] = float(v)

        if "kl_views" in loss:
            for k, v in loss["kl_views"].items():
                r[f"kl_{k}"] = float(v)

        if extra_fields is not None:
            r.update(extra_fields)

        self.losses.append(r)

    def print_losses(self, cv_idx, epoch_idx, pbar=None, keys=[]):
        l = pd.DataFrame(self.losses).query(f"cv == {cv_idx} & epoch == {epoch_idx}")
        l = l.groupby("type").mean()

        ptxt = f"[{datetime.now().strftime('%H:%M:%S')}] CV={cv_idx}, Epoch={epoch_idx} Loss (train/val)"
        ptxt += f" | Total={l.loc['train', 'total']:.2f}/{l.loc['val', 'total']:.2f}"

        for k in l.columns:
            if k not in ["cv", "epoch", "type", "total"]:
                ptxt += f" | {k}={l.loc['train', k]:.2f}/{l.loc['val', k]:.2f}"

        if pbar is not None:
            pbar.set_description(ptxt)
        else:
            print(ptxt)

    def save_losses(self):
        l = pd.DataFrame(self.losses)
        l.to_csv(f"{plot_folder}/files/{self.timestamp}_losses.csv", index=False)
        return l

    @staticmethod
    def plot_losses(losses_df, loss_terms=None, figsize=(3, 2)):
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
                c for c in losses_df if c not in ["cv", "epoch", "type", "total"]
            ]

        plot_df = pd.melt(
            losses_df,
            id_vars=["epoch", "type"],
            value_vars=loss_terms,
        )

        _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="variable",
            style="type",
            ax=ax,
        )
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
        PhenPred.save_figure(f"{plot_folder}/losses/{self.timestamp}_reconst_reg_loss")

        # Plot losses views
        for ltype in ["reconstruction_", "kl_"]:
            cols = [c for c in losses_df if c.startswith(f"{ltype}_")]

            if len(cols) > 0:
                plot_df = pd.melt(
                    losses_df,
                    id_vars=["epoch", "type"],
                    value_vars=cols,
                )

                _, ax = plt.subplots(1, 1, figsize=figsize, dpi=600)
                sns.lineplot(
                    data=plot_df,
                    x="epoch",
                    y="value",
                    hue="variable",
                    style="type",
                    ax=ax,
                )
                ax.legend(
                    title="Losses",
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                )
                ax.set(xlabel="Epoch", ylabel=f"{ltype} Loss")
                PhenPred.save_figure(
                    f"{plot_folder}/losses/{self.timestamp}_{ltype}_omics_loss"
                )
