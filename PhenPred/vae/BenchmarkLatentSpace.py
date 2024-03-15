import os
import umap
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from PhenPred import PALETTE_TTYPE
from PhenPred.vae.PlotUtils import GIPlot
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


class LatentSpaceBenchmark:
    def __init__(self, timestamp, data, latent_space, mofa_latent, move_diabetes_latent):
        self.data = data
        self.timestamp = timestamp

        self.latent_space = latent_space

        self.mofa_latent = mofa_latent
        self.move_diabetes_latent = move_diabetes_latent

        self.ss = data.samplesheet.copy()

        self.ss_ccell = pd.read_csv(
            f"{data_folder}/samplesheet_cancercell.csv", index_col=0
        )

        # Import novel drug response data
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

        # Define covariates
        if "proteomics" in self.data.dfs:
            covariates_prot = self.data.dfs["proteomics"][["CDH1", "VIM"]].add_suffix(
                "_prot"
            )

        if "transcriptomics" in self.data.dfs:
            covariates_trans = self.data.dfs["transcriptomics"][
                ["CDH1", "VIM"]
            ].add_suffix("_gexp")

        self.covariates = pd.concat(
            [
                self.ss_ccell["CopyNumberAttenuation"],
                self.ss_ccell["GeneExpressionCorrelation"],
                self.ss_ccell["CopyNumberInstability"],
                self.ss_ccell[["ploidy", "mutational_burden", "growth", "size"]],
                self.ss_ccell["replicates_correlation"].rename("RepsCorrelation"),
                covariates_prot if "proteomics" in self.data.dfs else None,
                covariates_trans if "transcriptomics" in self.data.dfs else None,
                pd.get_dummies(self.ss_ccell["media"]),
                pd.get_dummies(self.ss["growth_properties_sanger"]).add_prefix(
                    "sanger_"
                ),
                pd.get_dummies(self.ss["growth_properties_broad"]).add_prefix("broad_"),
                self.data.dfs["proteomics"].mean(1).rename("MeanProteomics")
                if "proteomics" in self.data.dfs
                else None,
                self.data.dfs["methylation"].mean(1).rename("MeanMethylation")
                if "methylation" in self.data.dfs
                else None,
                zscore(self.data.dfs["drugresponse"], nan_policy="omit")
                .mean(1)
                .rename("MeanDrugResponse")
                if "drugresponse" in self.data.dfs
                else None,
                self.df_drug_novel_bin.sum(1).rename("drug_responses").apply(np.log2),
            ],
            axis=1,
        )

        if not os.path.exists(f"{plot_folder}/latent"):
            os.makedirs(f"{plot_folder}/latent")

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
        factors = self.mofa_latent["factors"]
        variance = self.mofa_latent["rsquare"]

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
        plot_df = corr.pivot(index="factor", columns="latent", values="corr")
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
                random_state=42,
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
        markers=None,
    ):
        self.plot_latent_spaces_aux(method="UMAP", markers=markers)

    def plot_latent_spaces_aux(
        self,
        method="UMAP",
        umap_neighbors=25,
        umap_min_dist=0.25,
        umap_metric="euclidean",
        umap_n_components=2,
        markers=None,
    ):
        samplesheet = self.data.samplesheet["tissue"].fillna("Other tissue")
        centroid_distance_df = []
        clustering_score_df = {"model": [], "metric": [], "score": []}
        for n, z_joint in [
            ("vae", self.latent_space),
            ("mofa", self.mofa_latent["factors"]),
            ("move_diabetes", self.move_diabetes_latent["factors"]),
        ]:
            if method == "UMAP":
                # Get UMAP projections
                z_joint_dr = pd.DataFrame(
                    umap.UMAP(
                        n_neighbors=umap_neighbors,
                        min_dist=umap_min_dist,
                        metric=umap_metric,
                        n_components=umap_n_components,
                        random_state=42,
                    ).fit_transform(z_joint),
                    columns=[f"UMAP_{i+1}" for i in range(umap_n_components)],
                    index=z_joint.index,
                )
            elif method == "PCA":
                z_joint_dr = pd.DataFrame(
                    PCA(n_components=2).fit_transform(z_joint),
                    columns=[f"PCA_{i+1}" for i in range(2)],
                    index=z_joint.index,
                )
            else:
                raise Exception("Invalid DR method")

            cluster_labels = samplesheet[z_joint.index]
            clustering_score_df["model"].append(n)
            clustering_score_df["metric"].append("calinski_harabasz")
            clustering_score_df["score"].append(
                calinski_harabasz_score(
                    StandardScaler().fit_transform(z_joint), cluster_labels
                )
            )
            clustering_score_df["model"].append(n)
            clustering_score_df["metric"].append("davies_bouldin")
            clustering_score_df["score"].append(
                davies_bouldin_score(
                    StandardScaler().fit_transform(z_joint), cluster_labels
                )
            )

            # Plot projections by tissue type
            plot_df = pd.concat([z_joint_dr, samplesheet], axis=1)

            _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)
            sns.scatterplot(
                data=plot_df,
                x=f"{method}_1",
                y=f"{method}_2",
                hue="tissue",
                palette=PALETTE_TTYPE,
                alpha=0.75,
                ax=ax,
            )
            ax.set(
                xlabel=f"{method}_1",
                ylabel=f"{method}_2",
                xticklabels=[],
                yticklabels=[],
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            ax.get_legend().get_title().set_fontsize("6")

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)

            PhenPred.save_figure(
                f"{plot_folder}/latent/{self.timestamp}_{method.lower()}_joint{'' if n == 'vae' else f'_{n}'}"
            )

            # Plot projections by marker
            if markers is not None and n == "vae":
                for m in markers:
                    self.plot_latent_continuous(
                        pd.concat([z_joint_dr, markers[m]], axis=1).dropna(),
                        "joint",
                        m,
                        method=method,
                    )

        clustering_score_df = pd.DataFrame(clustering_score_df)
        _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)
        sns.barplot(data=clustering_score_df, x="metric", y="score", ax=ax, hue="model")
        PhenPred.save_figure(
            f"{plot_folder}/latent/{self.timestamp}_clustering_score_barplot"
        )

    def plot_latent_continuous(self, plot_df, name, m, method="UMAP"):
        ax = GIPlot.gi_continuous_plot(
            x=f"{method}_1",
            y=f"{method}_2",
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
            f"{plot_folder}/latent/{self.timestamp}_{method.lower()}_by_marker_{name}_{m}"
        )
