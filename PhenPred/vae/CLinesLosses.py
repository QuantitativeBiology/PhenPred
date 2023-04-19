import umap
import torch
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import scipy.stats as stats
import torch.nn.functional as F
from datetime import datetime
from sklearn.model_selection import KFold
from matplotlib.ticker import MaxNLocator
from PhenPred import PALETTE_TTYPE


class CLinesLosses:
    _dirPlots = "/home/egoncalves/PhenPred/reports/vae/"

    @classmethod
    def loss_function(cls, hypers, views, views_hat, means, log_variances):
        # Compute reconstruction loss across views
        mse_loss = 0
        view_mse_losses = {}
        for i, (n, view) in enumerate(views.items()):
            mse_loss_view = F.mse_loss(views_hat[i], view)
            mse_loss += mse_loss_view
            view_mse_losses[n] = mse_loss_view

        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + log_variances - means.pow(2) - log_variances.exp()
        )

        # Compute total loss
        total_loss = hypers["beta"] * kl_loss + mse_loss

        # Return total loss, total MSE loss, and view specific MSE loss
        return total_loss, mse_loss, kl_loss, view_mse_losses

    @classmethod
    def get_optimizer(cls, hyper, model):
        if hyper["optimizer_type"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=hyper["learning_rate"],
                weight_decay=hyper["w_decay"],
            )
        else:
            return torch.optim.RAdam(
                model.parameters(),
                lr=hyper["learning_rate"],
                weight_decay=hyper["w_decay"],
            )

    @classmethod
    def plot_losses(cls, losses_dict, kl_beta, timestamp=""):
        # Plot dataframes
        losses = pd.DataFrame(losses_dict)
        losses["epoch"] = losses.index

        losses_omics = pd.DataFrame(losses_datasets)
        losses_omics["epoch"] = losses_omics.index

        # Plot train and validation losses
        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=pd.melt(losses.loc[:, ["loss_train", "loss_val", "epoch"]], ["epoch"]),
            x="epoch",
            y="value",
            hue="variable",
            ax=ax,
        )
        ax.set(
            title=f"Train and Validation Loss (KL beta = {kl_beta})",
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

        plt.savefig(
            f"{cls._dirPlots}/losses/{timestamp}_train_validation_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Plot reconstruction and regularization losses
        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=pd.melt(losses.loc[:, ["mse_train", "kl_train", "epoch"]], ["epoch"]),
            x="epoch",
            y="value",
            hue="variable",
            ax=ax,
        )
        ax.set(
            title=f"Total loss (KL beta = {kl_beta})",
            xlabel="Epoch",
            ylabel="Loss",
        )
        legend_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            legend_handles,
            ["Reconstruction Loss", "KL Loss"],
            title="Losses",
            loc="upper left",
            bbox_to_anchor=(1, 1),
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.savefig(
            f"{cls._dirPlots}/losses/{timestamp}_reconst_reg_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Plot omics losses
        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=pd.melt(losses_omics, ["epoch"]),
            x="epoch",
            y="value",
            hue="variable",
            ax=ax,
        )
        ax.set(xlabel="Epoch", ylabel="Reconstruction Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(
            f"{cls._dirPlots}/losses/{timestamp}_reconst_omics_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    @classmethod
    def plot_latent_spaces(
        cls,
        timestamp,
        view_names,
        configs,
        umap_neighbors=25,
        umap_min_dist=0.25,
        umap_metric="euclidean",
        umap_n_components=2,
    ):
        # Get Tissue Types
        samplesheet = pd.read_csv(
            "/data/benchmarks/clines/samplesheet.csv", index_col=0
        )
        samplesheet = samplesheet["tissue"].fillna("Other tissue")

        # Read latent spaces
        latent_spaces = {
            n: pd.read_csv(
                f"{_dirPlots}/files/{timestamp}_latent_{n}.csv.gz", index_col=0
            )
            for n in view_names + ["joint"]
        }

        # Get UMAP projections
        latent_space_umaps = {
            k: pd.DataFrame(
                umap.UMAP(
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    metric=umap_metric,
                    n_components=umap_n_components,
                ).fit_transform(v),
                columns=[f"UMAP_{i+1}" for i in range(umap_n_components)],
                index=v.index,
            )
            for k, v in latent_spaces.items()
        }

        # Configs string
        configs_str = " ".join(
            [
                ("" if i % 4 else "\n") + f"{k}={v}"
                for i, (k, v) in enumerate(configs.items())
            ]
        )

        # Plot projections
        for l_name, l_space in latent_space_umaps.items():
            plot_df = pd.concat([l_space, samplesheet], axis=1)

            _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)
            sns.scatterplot(
                data=plot_df,
                x="UMAP_1",
                y="UMAP_2",
                hue="tissue",
                palette=PALETTE_TTYPE,
                alpha=0.95,
                ax=ax,
            )
            ax.set(
                title=f"UMAP {l_name}" + configs_str,
                xlabel="UMAP_1",
                ylabel="UMAP_2",
                xticklabels=[],
                yticklabels=[],
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.savefig(
                f"{cls._dirPlots}/latent/{timestamp}_umap_{l_name}.pdf",
                bbox_inches="tight",
            )
            plt.close()
