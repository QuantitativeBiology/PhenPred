import umap
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from PhenPred import PALETTE_TTYPE
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA


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

        self.df_drug_novel = pd.read_csv(
            f"{data_folder}/GDSC_fitted_dose_response_24Jul22_IC50.csv",
            index_col=0,
        )

        self.df_max_conc = pd.read_csv(
            f"{data_folder}/GDSC_fitted_dose_response_24Jul22_MAX_CONCENTRATION.csv",
            index_col=0,
        ).loc[self.df_drug_novel.index, "MAX_CONC"]

        self.df_drug_novel_bin = pd.DataFrame(
            {
                d: self.df_drug_novel.loc[d].dropna()
                < np.log(self.df_max_conc[d] * 0.5)
                for d in self.df_drug_novel.index
            }
        ).astype(float)

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
                self.df_drug_novel_bin.sum(1).rename("drug_responses").apply(np.log2),
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
            vmin=-1,
            vmax=1,
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
            vmin=-1,
            vmax=1,
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

        PhenPred.save_figure(
            f"{plot_folder}/latent/{self.timestamp}_factors_latents_corr_clustermap",
        )

        return corr

    def correlation_latents(self):
        plot_df = self.latent_space.corr()

        g = sns.clustermap(
            plot_df,
            cmap="RdYlGn",
            center=0,
            xticklabels=False,
            yticklabels=False,
            vmin=-1,
            vmax=1,
            linewidths=0.0,
            cbar_kws={"shrink": 0.5},
            figsize=(4, 4),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        PhenPred.save_figure(
            f"{plot_folder}/latent/{self.timestamp}_latents_corr_clustermap",
        )

    def covariates_latents(self, latents_corr):
        g = sns.clustermap(
            latents_corr,
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.0,
            xticklabels=True,
            yticklabels=True,
            annot=True,
            annot_kws={"fontsize": 3},
            fmt=".1f",
            col_cluster=False,
            cbar_kws={"shrink": 0.5},
            figsize=(8, 3.5),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        PhenPred.save_figure(
            f"{plot_folder}/latent/{self.timestamp}_latents_covariates_clustermap",
        )

    def latent_per_tissue(
        self,
        tissues=["Breast"],
        umap_neighbors=25,
        umap_min_dist=0.25,
        umap_metric="euclidean",
        umap_n_components=2,
    ):
        ss_cmp = pd.read_csv(f"{data_folder}/model_list_20230505.csv", index_col=0)

        t = "Lung"

        samples = ss_cmp.query(f"tissue == '{t}'").index.tolist()

        l_tissue = self.latent_space.reindex(index=samples).dropna()

        #
        l_tissue_umap = pd.DataFrame(
            umap.UMAP(
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                metric=umap_metric,
                n_components=umap_n_components,
            ).fit_transform(l_tissue),
            columns=[f"UMAP_{i+1}" for i in range(umap_n_components)],
            index=l_tissue.index,
        )

        #
        plot_df = pd.concat(
            [
                l_tissue_umap,
                ss_cmp.reindex(l_tissue.index),
                self.data.dfs["transcriptomics"][["VIM"]],
            ],
            axis=1,
        )

        _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)

        GIPlot.gi_continuous_plot(
            x="UMAP_1",
            y="UMAP_2",
            z="VIM",
            plot_df=plot_df,
            corr_annotation=False,
            mid_point_norm=False,
            mid_point=None,
            cmap="viridis",
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        sns.despine(ax=ax, left=False, bottom=False, right=False, top=False)

        PhenPred.save_figure(f"{plot_folder}/latent/{self.timestamp}_umap_joint_{t}")

    def plot_latent_spaces(
        self,
        view_names,
        umap_neighbors=25,
        umap_min_dist=0.25,
        umap_metric="euclidean",
        umap_n_components=2,
        markers=None,
    ):
        # Get Tissue Types
        samplesheet = self.data.samplesheet.copy()
        samplesheet = samplesheet["tissue"].fillna("Other tissue")

        # Read latent spaces
        latent_spaces = {
            n: pd.read_csv(
                f"{plot_folder}/files/{self.timestamp}_latent_{n}.csv.gz", index_col=0
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

            PhenPred.save_figure(f"{plot_folder}/latent/{self.timestamp}_umap_{l_name}")

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

                    PhenPred.save_figure(
                        f"{plot_folder}/latent/{self.timestamp}_umap_by_marker_{m}_{l_name}"
                    )

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

                    PhenPred.save_figure(
                        f"{plot_folder}/latent/{self.timestamp}_umap_by_marker_{m}_{l_name}"
                    )
