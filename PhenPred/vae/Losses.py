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
from sklearn import metrics
from datetime import datetime
from PhenPred import PALETTE_TTYPE
from sklearn.model_selection import KFold
from matplotlib.ticker import MaxNLocator
from PhenPred.vae.PlotUtils import GIPlot
from PhenPred.vae import data_folder, plot_folder


class CLinesLosses:
    @classmethod
    def loss_function(
        cls,
        hypers,
        views,
        views_hat,
        means,
        log_variances,
        z_joint,
        views_nans=None,
        covariates=None,
    ):
        # Compute reconstruction loss across views
        mse_loss, view_mse_loss = 0, {}
        for i, k in enumerate(hypers["datasets"]):
            X, X_ = views[i], views_hat[i]

            if views_nans is not None:
                X, X_ = X[views_nans[i]], X_[views_nans[i]]

            loss_func = cls.reconstruction_loss(hypers["reconstruction_loss"])
            v_mse_loss = loss_func(X_, X)

            mse_loss += v_mse_loss
            view_mse_loss[k] = v_mse_loss

        # Compute KL divergence loss
        kl_loss = 0
        for mu, log_var in zip(means, log_variances):
            kl_loss += (
                -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / len(mu)
            )

        kl_loss /= hypers["batch_size"]
        kl_loss *= hypers["beta"]

        # Compute batch covariate loss
        covariate_loss = 0 if covariates is None else cls.mmd_loss(z_joint, covariates)

        # Compute total loss
        total_loss = kl_loss + mse_loss + covariate_loss

        # Return total loss, total MSE loss, and view specific MSE loss
        return dict(
            total=total_loss,
            mse=mse_loss,
            kl=kl_loss,
            covariate=covariate_loss,
            mse_views=view_mse_loss,
        )

    @classmethod
    def mmd_loss(cls, means, covariates):
        dist = torch.cdist(means, means, p=2)

        covariate_losses = []
        for i in range(covariates.shape[1]):
            idx_x = torch.nonzero(covariates[:, i]).flatten()
            idx_y = torch.nonzero(covariates[:, i] == 0).flatten()

            if len(idx_x) == 0 or len(idx_y) == 0:
                covariate_losses.append(torch.tensor(0.0))
                continue

            k_xx = cls.gaussian_kernel(dist[idx_x][:, idx_x]).mean()
            k_yy = cls.gaussian_kernel(dist[idx_y][:, idx_y]).mean()
            k_xy = cls.gaussian_kernel(dist[idx_x][:, idx_y]).mean()

            covariate_losses.append(k_xx + k_yy - 2 * k_xy)

        return torch.stack(covariate_losses).mean()

    @classmethod
    def gaussian_kernel(cls, D, gamma=None):
        if gamma is None:
            gamma = [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                100,
                1e3,
                1e4,
                1e5,
                1e6,
            ]

        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K / len(gamma)

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
        # Plot train and validation losses
        losses = pd.DataFrame(
            {k: v for k, v in losses_dict.items() if not k.endswith("_views")}
        )
        losses["epoch"] = losses.index
        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=pd.melt(
                losses.loc[:, ["train_total", "val_total", "epoch"]], ["epoch"]
            ),
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
            f"{plot_folder}/losses/{timestamp}_train_validation_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Plot reconstruction and regularization losses
        plot_df = pd.melt(
            losses[[c for c in losses if not c.endswith("_total")]], ["epoch"]
        )
        plot_df["type"] = plot_df["variable"].apply(lambda v: v.split("_")[0]).values
        plot_df["loss_type"] = (
            plot_df["variable"].apply(lambda v: v.split("_")[1]).values
        )
        _, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        sns.lineplot(
            data=plot_df,
            x="epoch",
            y="value",
            hue="loss_type",
            style="type",
            ls="--",
            ax=ax,
        )
        ax.set(
            title=f"Total loss (KL beta = {kl_beta})",
            xlabel="Epoch",
            ylabel="Loss",
        )
        plt.savefig(
            f"{plot_folder}/losses/{timestamp}_reconst_reg_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

        # Plot losses views
        plot_df = pd.concat(
            [
                pd.DataFrame(losses_dict["train_mse_views"]).assign(type="train"),
                pd.DataFrame(losses_dict["val_mse_views"]).assign(type="val"),
            ]
        )
        plot_df["epoch"] = plot_df.index
        plot_df = pd.melt(plot_df, ["epoch", "type"])
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
        plt.savefig(
            f"{plot_folder}/losses/{timestamp}_reconst_omics_loss.pdf",
            bbox_inches="tight",
        )
        plt.close("all")

    @classmethod
    def plot_latent_spaces(
        cls,
        timestamp,
        view_names,
        umap_neighbors=25,
        umap_min_dist=0.25,
        umap_metric="euclidean",
        umap_n_components=2,
        markers=None,
        data=None,
    ):
        # Get Tissue Types
        if data is None:
            samplesheet = pd.read_csv(f"{data_folder}/samplesheet.csv", index_col=0)
            samplesheet = samplesheet["tissue"].fillna("Other tissue")
        else:
            samplesheet = data.samplesheet.copy()
            samplesheet = samplesheet["tissue"].fillna("Other tissue")

        # Read latent spaces
        latent_spaces = {
            n: pd.read_csv(
                f"{plot_folder}/files/{timestamp}_latent_{n}.csv.gz", index_col=0
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

        # Plot projections by tissue type
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
                xlabel="UMAP_1",
                ylabel="UMAP_2",
                xticklabels=[],
                yticklabels=[],
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.get_legend().get_title().set_fontsize("6")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

            ax.get_legend().get_title().set_fontsize("6")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

            plt.savefig(
                f"{plot_folder}/latent/{timestamp}_umap_{l_name}.pdf",
                bbox_inches="tight",
            )
            plt.close()

        # Plot projections by marker
        if markers is not None:
            for l_name, l_space in latent_space_umaps.items():
                for m in markers:
                    plot_df = pd.concat([l_space, markers[m]], axis=1).dropna()

                    ax = GIPlot.gi_continuous_plot(
                        x="UMAP_1",
                        y="UMAP_2",
                        z=m,
                        plot_df=plot_df,
                        corr_annotation=False,
                        mid_point_norm=False,
                        mid_point=None,
                        cmap="viridis",
                    )

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

                    plt.savefig(
                        f"{plot_folder}/latent/{timestamp}_umap_by_marker_{m}_{l_name}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

        # Plot projections by marker
        if markers is not None:
            for l_name, l_space in latent_space_umaps.items():
                for m in markers:
                    plot_df = pd.concat([l_space, markers[m]], axis=1).dropna()

                    ax = GIPlot.gi_continuous_plot(
                        x="UMAP_1",
                        y="UMAP_2",
                        z=m,
                        plot_df=plot_df,
                        corr_annotation=False,
                        mid_point_norm=False,
                        mid_point=None,
                        cmap="viridis",
                    )

                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

                    plt.savefig(
                        f"{plot_folder}/latent/{timestamp}_umap_by_marker_{m}_{l_name}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

    @classmethod
    def activation_function(cls, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "elu":
            return nn.ELU()
        elif name == "selu":
            return nn.SELU()
        elif name == "softplus":
            return nn.Softplus()
        else:
            return nn.Identity()

    @classmethod
    def reconstruction_loss(cls, name):
        if name == "mse":
            return F.mse_loss
        elif name == "bce":
            return F.binary_cross_entropy
        elif name == "gauss":
            return nn.GaussianNLLLoss
        else:
            return F.mse_loss
