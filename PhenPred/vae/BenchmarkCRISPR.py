import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from math import sqrt
from scipy.stats import skew
from datetime import datetime
from scipy.special import stdtr
from PhenPred.vae.PlotUtils import GIPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from statsmodels.stats.multitest import multipletests
from PhenPred.vae.Utils import two_vars_correlation, LModel


class CRISPRBenchmark:
    def __init__(
        self, timestamp, data, vae_imputed, mofa_imputed, min_obs=15, skew_threshold=-2
    ):
        self.timestamp = timestamp

        self.min_obs = min_obs
        self.skew_threshold = skew_threshold

        self.data = data
        self.vae_imputed = vae_imputed
        self.mofa_imputed = mofa_imputed

        # CRISPR-Cas9 datasets
        self.df_original = data.dfs["crisprcas9"].dropna(how="all").dropna(axis=1)
        self.df_vae = self.vae_imputed["crisprcas9"]

        # Genomics
        self.mutations = self.data.mutations.add_suffix("_mut")
        self.deletions = (self.data.cnv == "Deletion").astype(int).add_suffix("_del")
        self.amplitifications = (
            (self.data.cnv == "Amplification").astype(int).add_suffix("_amp")
        )
        self.fusions = self.data.fusions.add_suffix("_fusion")
        self.msi = self.data.labels["MSI"]

        self.genomics = pd.concat(
            [
                self.mutations,
                self.deletions,
                self.amplitifications,
                self.fusions,
                self.msi,
            ],
            axis=1,
        )
        self.genomics = self.genomics.loc[:, self.genomics.sum() >= self.min_obs]

        # Sample sheet
        self.ss = data.samplesheet.copy()

    def run(self, run_associations=True):
        if run_associations:
            self.lm_genomics = self.genomic_associations()
            self.lm_genomics.to_csv(
                f"{plot_folder}/crispr/{self.timestamp}_genomics_crisprcas9.csv.gz",
                compression="gzip",
                index=False,
            )
        else:
            self.lm_genomics = pd.read_csv(
                f"{plot_folder}/crispr/{self.timestamp}_genomics_crisprcas9.csv.gz"
            )

        self.associations_scatter_pvals(self.lm_genomics)
        self.gene_skew_correlation()

    def gene_skew_correlation(self):
        plot_df = pd.concat(
            [
                self.df_original.apply(skew).astype(float).rename("orig"),
                self.df_vae.apply(skew).astype(float).rename("vae"),
            ],
            axis=1,
        ).dropna()

        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        sns.scatterplot(
            data=plot_df,
            x="orig",
            y="vae",
            alpha=0.75,
            color="#656565",
            ax=ax,
        )
        sns.regplot(
            data=plot_df,
            x="orig",
            y="vae",
            scatter=False,
            color="#fc8d62",
            truncate=True,
            line_kws={"lw": 1},
            ax=ax,
        )
        ax.set(
            title=f"CRISPR-Cas9 skew (N={plot_df.shape[0]:,})",
            xlabel="Skew original",
            ylabel=f"Skew VAE",
        )

        # same axes limits and step sizes
        ax_min, ax_max = (
            min(plot_df["orig"].min(), plot_df["vae"].min()),
            max(plot_df["orig"].max(), plot_df["vae"].max()),
        )

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())

        rmse = sqrt(mean_squared_error(plot_df["orig"], plot_df["vae"]))
        s, _ = stats.spearmanr(
            plot_df["orig"],
            plot_df["vae"],
        )
        r, _ = stats.pearsonr(
            plot_df["orig"],
            plot_df["vae"],
        )
        annot_text = f"R={r:.2g}; Rho={s:.2g}; RMSE={rmse:.2f}"
        ax.text(0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right")

        ax.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        PhenPred.save_figure(
            f"{plot_folder}/crispr/{self.timestamp}_gene_skew_corrplot"
        )

    def genomic_associations(self):
        # Covariates
        covs = pd.concat(
            [
                self.ss["growth_properties_sanger"]
                .str.get_dummies()
                .add_prefix("sanger_"),
                self.ss["growth_properties_broad"]
                .str.get_dummies()
                .add_prefix("broad_"),
                self.ss["tissue"].str.get_dummies()[["Haematopoietic and Lymphoid"]],
            ],
            axis=1,
        )

        y_features = pd.concat(
            [
                self.df_vae.apply(skew).astype(float).rename("vae"),
                self.df_original.apply(skew).astype(float).rename("orig"),
            ],
            axis=1,
        )
        y_features = list(
            y_features.loc[(y_features < self.skew_threshold).any(axis=1)].index
        )

        # Genomics ~ CRISPR VAE
        samples = list(
            set(self.df_vae.dropna().index)
            .intersection(self.data.mutations.index)
            .intersection(self.data.cnv.index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.genomics.loc[samples].replace(np.nan, 0).astype(int)

        lm_genomics_vae = LModel(
            Y=self.df_vae.loc[samples, y_features],
            X=x.loc[samples],
            M=covs.loc[samples],
        ).fit_matrix()

        lm_genomics_vae = LModel.multipletests(
            lm_genomics_vae, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_genomics_vae = lm_genomics_vae.set_index(["y_id", "x_id"])

        # Genomics ~ CRISPR original
        samples = list(
            set(self.df_original.dropna().index)
            .intersection(self.data.mutations.index)
            .intersection(self.data.cnv.index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.genomics.loc[samples].replace(np.nan, 0).astype(int)

        lm_genomics_orig = LModel(
            Y=self.df_original.loc[samples],
            X=x.loc[samples],
            M=covs.loc[samples],
        ).fit_matrix()

        lm_genomics_orig = LModel.multipletests(
            lm_genomics_orig, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_genomics_orig = lm_genomics_orig.set_index(["y_id", "x_id"])

        # Concatenate
        lm_genomics = (
            pd.concat(
                [
                    lm_genomics_orig.add_suffix("_orig"),
                    lm_genomics_vae.add_suffix("_vae"),
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )

        return lm_genomics

    def associations_scatter_pvals(self, lm_genomics):
        plot_df = lm_genomics.query("fdr_orig < 0.05 | fdr_vae < 0.05").copy()

        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        sns.scatterplot(
            x=-np.log10(plot_df["pval_orig"]),
            y=-np.log10(plot_df["pval_vae"]),
            color="black",
            alpha=0.5,
            linewidth=0,
            s=5,
            zorder=1,
            ax=ax,
        )
        # same axes limits and step sizes
        ax_min, ax_max = (
            min(
                (-np.log10(plot_df["pval_orig"])).min(),
                (-np.log10(plot_df["pval_vae"])).min(),
            ),
            max(
                (-np.log10(plot_df["pval_orig"])).max(),
                (-np.log10(plot_df["pval_vae"])).max(),
            ),
        )

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())

        ax.axline((0, 0), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        y_fdr = -np.log10(plot_df.query("fdr_vae < 0.01")["pval_vae"].max())
        ax.axhline(y_fdr, lw=0.5, ls="--", color="black", zorder=-1)
        ax.text(
            0.99,
            0.05,
            "FDR 1%",
            fontsize=4,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        x_fdr = -np.log10(plot_df.query("fdr_orig < 0.01")["pval_orig"].max())
        ax.axvline(x_fdr, lw=0.5, ls="--", color="black", zorder=-1)
        ax.text(
            0.01,
            0.99,
            "FDR 1%",
            fontsize=4,
            transform=ax.transAxes,
            ha="left",
            va="top",
            rotation=90,
        )

        ax.set(
            title=f"CRISPR-Cas9 ~ Genomics associations\n(N={plot_df.shape[0]:,})",
            xlabel="Original log-ratio p-value (-log10)",
            ylabel="VAE log-ratio p-value (-log10)",
        )

        PhenPred.save_figure(
            f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_pval_scatter",
        )

    def plot_associations(self, associations=None):
        if associations is None:
            associations = [
                ("BRAF", "MAPK1", "BRAF_mut"),
            ]

        for y_id, x_id, z_id in associations:
            # y_id, x_id, z_id = ("BRAF", "MAPK1", "BRAF_mut")

            plot_df = pd.concat(
                [
                    self.df_original[[y_id, x_id]].add_suffix(f"_orig"),
                    self.df_vae[[y_id, x_id]].add_suffix(f"_vae"),
                    self.genomics[z_id],
                    self.ss["tissue"],
                ],
                axis=1,
            ).dropna(subset=[f"{x_id}_vae", f"{y_id}_vae", z_id])

            plot_df[z_id].replace({0: "WT", 1: z_id}, inplace=True)
            plot_df["predicted"] = (
                plot_df[[f"{y_id}_orig", f"{x_id}_orig"]].isnull().any(axis=1)
            )
            plot_df["predicted"].replace(
                {
                    True: f"Predicted (N={plot_df['predicted'].sum()})",
                    False: f"Observed (N={(~plot_df['predicted']).sum()})",
                },
                inplace=True,
            )

            pal, pal_order = {
                z_id: "#fc8d62",
                "WT": "#e1e1e1",
                0: "#e1e1e1",
            }, ["WT", z_id]

            # Predicted
            g = GIPlot.gi_regression_marginal(
                x=f"{x_id}_vae",
                y=f"{y_id}_vae",
                z=z_id,
                style="predicted",
                plot_df=plot_df,
                discrete_pal=pal,
                hue_order=pal_order,
                legend_title=f"{z_id}",
                scatter_kws=dict(edgecolor="w", lw=0.1, s=10, alpha=0.75),
            )

            g.ax_joint.set_xlabel(f"{x_id} CRISPR-Cas9 (VAE)")
            g.ax_joint.set_ylabel(f"{y_id} CRISPR-Cas9 (VAE)")

            plt.gcf().set_size_inches(2, 2)

            PhenPred.save_figure(
                f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}",
            )
