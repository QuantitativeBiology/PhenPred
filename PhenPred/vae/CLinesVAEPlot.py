import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PhenPred import PALETTE_TTYPE

_dirPlots = "/home/egoncalves/PhenPred/reports/vae/"


def plot_losses(losses_dict, losses_datasets, alpha_KL, alpha_MSE, timestamp=""):
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
        title=f"Train and Validation Loss (lambda = {alpha_MSE} , alpha = {alpha_KL})",
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
        f"{_dirPlots}/losses/{timestamp}_train_validation_loss.pdf",
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
        title=f"Total loss (lambda = {alpha_MSE} , alpha = {alpha_KL})",
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
        f"{_dirPlots}/losses/{timestamp}_reconst_reg_loss.pdf",
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
        f"{_dirPlots}/losses/{timestamp}_reconst_omics_loss.pdf",
        bbox_inches="tight",
    )
    plt.close("all")


def plot_latent_spaces(
    timestamp,
    view_names,
    configs,
    umap_neighbors=25,
    umap_min_dist=0.25,
    umap_metric="euclidean",
    umap_n_components=2,
):
    # Get Tissue Types
    samplesheet = pd.read_csv("/data/benchmarks/clines/samplesheet.csv", index_col=0)
    samplesheet = samplesheet["tissue"].fillna("Other tissue")

    # Read latent spaces
    latent_spaces = {
        n: pd.read_csv(f"{_dirPlots}/files/{timestamp}_latent_{n}.csv.gz", index_col=0)
        for n in view_names
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
            f"{_dirPlots}/latent/{timestamp}_umap_{l_name}.pdf",
            bbox_inches="tight",
        )
        plt.close()
