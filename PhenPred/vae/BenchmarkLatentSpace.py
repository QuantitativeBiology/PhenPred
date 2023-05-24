from statistics import variance
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.Utils import two_vars_correlation


class LatentSpaceBenchmark:
    def __init__(self, timestamp, data):
        self.data = data
        self.timestamp = timestamp

        self.mofa_db = CLinesDatasetMOFA()

        self.latent_space = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz", index_col=0
        )

        self.ss = data.samplesheet.copy()

        self.ss_ccell = pd.read_csv(
            f"{data_folder}/samplesheet_cancercell.csv", index_col=0
        )

        covariates_prot = self.data.dfs["proteomics"][["CDH1", "VIM"]].add_suffix(
            "_prot"
        )
        covariates_trans = self.data.dfs["transcriptomics"][["CDH1", "VIM"]].add_suffix(
            "_gexp"
        )

        self.covariates = pd.concat(
            [
                self.ss_ccell["CopyNumberAttenuation"],
                self.ss_ccell["GeneExpressionCorrelation"],
                self.ss_ccell["CopyNumberInstability"],
                self.ss_ccell[["ploidy", "mutational_burden", "growth", "size"]],
                self.ss_ccell["replicates_correlation"].rename("RepsCorrelation"),
                covariates_prot,
                covariates_trans,
                pd.get_dummies(self.ss_ccell["media"]),
                pd.get_dummies(self.ss["growth_properties_sanger"]).add_prefix(
                    "sanger_"
                ),
                pd.get_dummies(self.ss["growth_properties_broad"]).add_prefix("broad_"),
                self.data.dfs["proteomics"].mean(1).rename("MeanProteomics"),
                self.data.dfs["methylation"].mean(1).rename("MeanMethylation"),
                self.data.dfs["drugresponse"].mean(1).rename("MeanDrugResponse"),
            ],
            axis=1,
        )

    def correlate_latents_with_covariates(self):
        latents_corr = {}

        for l in self.latent_space:
            latents_corr[l] = {}

            for c in self.covariates:
                fc_samples = list(
                    self.covariates.reindex(self.latent_space[l].index)[c]
                    .dropna()
                    .index
                )
                latents_corr[l][c] = two_vars_correlation(
                    self.latent_space[l][fc_samples], self.covariates[c][fc_samples]
                )["corr"]

        latents_corr = pd.DataFrame(latents_corr).dropna()

        return latents_corr

    def run(self):
        self.correlation_latents()

        latents_corr = self.correlate_latents_with_covariates()
        latents_corr.to_csv(
            f"{plot_folder}/latent/{self.timestamp}_latents_covariates_corr.csv",
        )

        self.covariates_latents(latents_corr)

        factors_corr = self.correlate_mofa_factors(latents_corr)
        factors_corr.to_csv(
            f"{plot_folder}/latent/{self.timestamp}_factors_mofa_corr.csv",
        )

    def correlate_mofa_factors(self, latents_corr):
        factors = self.mofa_db.factors
        variance = self.mofa_db.rsquare

        latents = self.latent_space
        covs = latents_corr.copy()
        covs.columns = [f"L{l.split('_')[1]}" for l in covs.columns]

        samples = list(set(factors.index).intersection(latents.index))

        corr = pd.DataFrame(
            [
                two_vars_correlation(
                    latents.loc[samples, l],
                    factors.loc[samples, f],
                    extra_fields=dict(factor=f, latent=l),
                )
                for f in factors
                for l in latents
            ]
        )

        # Clustermap
        plot_df = corr.pivot("factor", "latent", "corr")
        plot_df.columns = [f"L{l.split('_')[1]}" for l in plot_df.columns]

        x_order = natsorted(plot_df.columns)
        y_order = natsorted(plot_df.index)

        ticklabelsfs = 4

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(6, 6),
            dpi=600,
            gridspec_kw={"width_ratios": [6, 1], "height_ratios": [6, 2.5]},
            sharey=False,
            sharex=False,
        )

        # Correlation heatmap
        sns.heatmap(
            plot_df.loc[y_order, x_order],
            cmap="RdYlGn",
            center=0,
            xticklabels=False,
            yticklabels=True,
            linewidths=0.0,
            annot=True,
            annot_kws={"fontsize": 3},
            cbar=False,
            fmt=".1f",
            ax=axs[0, 0],
        )

        axs[0, 0].set_xlabel("")
        axs[0, 0].set_ylabel("")
        axs[0, 0].set_title("Correlation (pearson's r)", fontsize=5)

        for tick in axs[0, 0].yaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabelsfs)

        # Variance heatmap
        sns.heatmap(
            variance.T.loc[y_order],
            cmap="Blues",
            xticklabels=True,
            yticklabels=False,
            linewidths=0.0,
            cbar=False,
            annot=True,
            annot_kws={"fontsize": 3},
            fmt=".1f",
            ax=axs[0, 1],
        )

        axs[0, 1].set_ylabel("")
        axs[0, 1].set_xlabel("")
        axs[0, 1].set_title("Variance explained", fontsize=5)

        for tick in axs[0, 1].xaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabelsfs)

        # Covariates heatmap
        sns.heatmap(
            covs[x_order],
            cmap="RdYlGn",
            center=0,
            xticklabels=True,
            yticklabels=True,
            linewidths=0.0,
            annot=True,
            annot_kws={"fontsize": 3},
            cbar=False,
            fmt=".1f",
            ax=axs[1, 0],
        )

        axs[1, 0].set_xlabel("Covariates correlation", fontsize=5)
        axs[1, 0].set_ylabel("")

        for tick in axs[1, 0].xaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabelsfs)

        for tick in axs[1, 0].yaxis.get_major_ticks():
            tick.label.set_fontsize(ticklabelsfs)

        # Change width space
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        # Remove ticks
        for ax in axs.flatten():
            ax.tick_params(axis="both", which="both", length=0)

        # Remove unused axes
        fig.delaxes(axs[1, 1])

        plt.savefig(
            f"{plot_folder}/latent/{self.timestamp}_factors_latents_corr_clustermap.pdf",
            bbox_inches="tight",
        )
        plt.close()

        return corr

    def correlation_latents(self):
        plot_df = self.latent_space.corr()

        g = sns.clustermap(
            plot_df,
            cmap="RdYlGn",
            center=0,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.0,
            cbar_kws={"shrink": 0.5},
            figsize=(4, 4),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        plt.savefig(
            f"{plot_folder}/latent/{self.timestamp}_latents_corr_clustermap.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def covariates_latents(self, latents_corr):
        g = sns.clustermap(
            latents_corr,
            cmap="RdYlGn",
            center=0,
            linewidths=0.0,
            xticklabels=True,
            yticklabels=True,
            col_cluster=False,
            cbar_kws={"shrink": 0.5},
            figsize=(8, 3.5),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        plt.savefig(
            f"{plot_folder}/latent/{self.timestamp}_latents_covariates_clustermap.pdf",
            bbox_inches="tight",
        )
        plt.close()
