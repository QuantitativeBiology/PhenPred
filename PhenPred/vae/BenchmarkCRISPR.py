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
from scipy.stats import pearsonr, spearmanr
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Utils import two_vars_correlation, LModel


class CRISPRBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Original dataset
        self.df_original = pd.read_csv(
            f"{data_folder}/crisprcas9_22Q2.csv", index_col=0
        ).T.dropna()

        # VAE imputed dataset
        self.df_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_crisprcas9.csv.gz", index_col=0
        )

        # Genomics
        self.genomics = pd.read_csv(f"{data_folder}/genomics.csv", index_col=0).T

        # Sample sheet
        self.ss = pd.read_csv(f"{data_folder}/cmp_model_list_20230307.csv", index_col=0)

    def run(self):
        self.sample_correlation()

        lm_genomics = self.genomic_associations()

        self.plot_associations(lm_genomics)
        self.gene_skew_correlation()

    def sample_correlation(self):
        samples = list(set(self.df_original.index).intersection(self.df_vae.index))
        genes = list(set(self.df_original.columns).intersection(self.df_vae.columns))

        samples_corr = pd.DataFrame(
            {
                s: two_vars_correlation(
                    self.df_original.loc[s, genes], self.df_vae.loc[s, genes]
                )
                for s in samples
            }
        ).T

        _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

        sns.histplot(samples_corr["corr"], bins=20, ax=ax)

        ax.set(
            xlabel="Pearson correlation",
            ylabel="Number of samples",
            title=f"Sample correlation (N={len(samples):,})",
        )

        plt.savefig(
            f"{plot_folder}/crispr/{self.timestamp}_samples_corr_histogram.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def gene_skew_correlation(self):
        plot_df = pd.concat(
            [
                self.df_original.apply(skew).astype(float).rename("orig"),
                self.df_vae.apply(skew).astype(float).rename("vae"),
            ],
            axis=1,
        )

        _, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=600)

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
            color="#F2C500",
            truncate=True,
            ax=ax,
        )
        ax.set(
            title=f"CRISPR-Cas9 skew (N={plot_df.shape[0]:,})",
            xlabel="Skew original",
            ylabel=f"Skew VAE",
        )

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

        plt.savefig(
            f"{plot_folder}/crispr/{self.timestamp}_gene_skew_corrplot.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def genomic_associations(self):
        # Covariates
        covs = pd.concat(
            [
                self.ss["growth_properties"].str.get_dummies(),
                self.ss["tissue"].str.get_dummies()[
                    ["Haematopoietic and Lymphoid", "Lung"]
                ],
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
        y_features = list(y_features.loc[(y_features < -2).any(axis=1)].index)

        # Genomics ~ CRISPR VAE
        samples = list(
            set(self.df_vae.dropna().index)
            .intersection(self.genomics.dropna().index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x_features = list(self.genomics.columns[self.genomics.loc[samples].sum() >= 10])

        lm_genomics_vae = LModel(
            Y=self.df_vae.loc[samples, y_features],
            X=self.genomics.loc[samples, x_features],
            M=covs.loc[samples],
        ).fit_matrix()

        lm_genomics_vae = LModel.multipletests(
            lm_genomics_vae, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_genomics_vae = lm_genomics_vae.set_index(["y_id", "x_id"])

        # Genomics ~ CRISPR original
        samples = list(
            set(self.df_original.dropna().index)
            .intersection(self.genomics.dropna().index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x_features = list(self.genomics.columns[self.genomics.loc[samples].sum() >= 10])

        lm_genomics_orig = LModel(
            Y=self.df_original.loc[samples],
            X=self.genomics.loc[samples, x_features],
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

    def plot_associations(self, lm_genomics):
        plot_df = lm_genomics.query("fdr_orig < 0.05 | fdr_vae < 0.05").copy()

        _, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=600)

        sns.scatterplot(
            x=-np.log10(plot_df["pval_orig"]),
            y=-np.log10(plot_df["pval_vae"]),
            color="#656565",
            lw=0,
            s=3,
            zorder=1,
            rasterized=True,
            ax=ax,
        )

        loc = plticker.MultipleLocator(base=25)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # set same axis range for both axes
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)

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
            title=f"CRISPR-Cas9 ~ Genomics associations (N={plot_df.shape[0]:,})",
            xlabel="Original log-ratio p-value (-log10)",
            ylabel="VAE log-ratio p-value (-log10)",
        )

        plt.savefig(
            f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_pval_scatter.pdf",
            bbox_inches="tight",
        )
        plt.close()

        for y_id, x_id, z_id in [
            ("BRAF", "MAPK1", "BRAF_mut"),
            ("FLI1", "TRIM8", "EWSR1.FLI1_mut"),
            ("CTNNB1", "TCF7L2", "APC_mut"),
            ("PAX8", "PARD3", "loss.cnaPANCAN399"),
            ("WRN", "RPL22L1", "msi_status"),
        ]:
            # y_id, x_id, z_id = ("WRN", "RPL22L1", "msi_status")

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

            tissues = [
                "Endometrium",
                "Ovary",
                "Large Intestine",
                "Stomach",
                "Haematopoietic and Lymphoid",
            ]
            plot_df["tissue_parsed"] = plot_df["tissue"].apply(
                lambda x: x if x in tissues else "Other"
            )

            pal, pal_order = {
                z_id: "#fc8d62",
                "WT": "#e1e1e1",
                0: "#E1E1E1",
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
                scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
            )

            g.ax_joint.set_xlabel(f"{x_id} CRISPR-Cas9 (VAE)")
            g.ax_joint.set_ylabel(f"{y_id} CRISPR-Cas9 (VAE)")

            plt.gcf().set_size_inches(2, 2)

            plt.savefig(
                f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}.pdf",
                bbox_inches="tight",
            )
            plt.close("all")

            # Original
            df = plot_df[plot_df["predicted"].apply(lambda x: x.startswith("Observed"))]
            g = GIPlot.gi_regression_marginal(
                x=f"{x_id}_orig",
                y=f"{y_id}_orig",
                z=z_id,
                style=None,
                plot_df=df,
                discrete_pal=pal,
                hue_order=pal_order,
                legend_title=f"{z_id}",
                scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
            )

            g.ax_joint.set_xlabel(f"{x_id} CRISPR-Cas9 (Measured)")
            g.ax_joint.set_ylabel(f"{y_id} CRISPR-Cas9 (Measured)")

            plt.gcf().set_size_inches(2, 2)

            plt.savefig(
                f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}_original.pdf",
                bbox_inches="tight",
            )
            plt.close("all")

            # Tissue
            pal = GIPlot.PAL_TISSUE_2
            pal["Other"] = "#e1e1e1"
            pal[0] = "#e1e1e1"
            order = ["Other"] + tissues

            if y_id in ["WRN"]:
                g = GIPlot.gi_regression_marginal(
                    x=f"{x_id}_vae",
                    y=f"{y_id}_vae",
                    z="tissue_parsed",
                    style="predicted",
                    plot_df=plot_df,
                    discrete_pal=pal,
                    legend_title="Tissue",
                    hue_order=order,
                    plot_annot=False,
                    scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
                )

                for i, c in enumerate(tissues[::-1]):
                    df = plot_df.query(f"tissue_parsed == '{c}'")
                    r, p = pearsonr(df[f"{x_id}_vae"], df[f"{y_id}_vae"])

                    sns.regplot(
                        x=f"{x_id}_vae",
                        y=f"{y_id}_vae",
                        data=plot_df.query(f"tissue_parsed == '{c}'"),
                        color=pal[c],
                        truncate=True,
                        fit_reg=True,
                        scatter=False,
                        label=f"{c} (r={r:.2f}, p={p:.2e})",
                        ci=None,
                        line_kws=dict(lw=1.0, color=pal[c]),
                        ax=g.ax_joint,
                    )

                    g.ax_joint.text(
                        0.95,
                        0.05 + 0.05 * i,
                        f"{c} (r={r:.2f}, p={p:.2e})",
                        fontsize=4,
                        transform=g.ax_joint.transAxes,
                        ha="right",
                    )

                g.ax_joint.set_xlabel(f"{x_id} CRISPR-Cas9 (VAE)")
                g.ax_joint.set_ylabel(f"{y_id} CRISPR-Cas9 (VAE)")

                plt.gcf().set_size_inches(2, 2)

                plt.savefig(
                    f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}_tissue.pdf",
                    bbox_inches="tight",
                )
                plt.close("all")

    def gls(self):
        """
        GLS regression of CRISPR-Cas9 screens.
        Code adapted from https://github.com/kundajelab/coessentiality/blob/master/gene_pairs.py
        """

        screens = self.df_vae.T

        cholsigmainv = np.linalg.cholesky(np.linalg.inv(np.cov(screens.T)))
        warped_screens = screens.values @ cholsigmainv
        warped_intercept = cholsigmainv.sum(axis=0)

        def linear_regression(warped_screens, warped_intercept):
            GLS_coef = np.empty((len(warped_screens), len(warped_screens)))
            GLS_se = np.empty((len(warped_screens), len(warped_screens)))
            ys = warped_screens.T

            for gene_index in range(len(warped_screens)):
                X = np.stack((warped_intercept, warped_screens[gene_index]), axis=1)
                coef, residues = np.linalg.lstsq(X, ys, rcond=None)[:2]
                df = warped_screens.shape[1] - 2
                GLS_coef[gene_index] = coef[1]
                GLS_se[gene_index] = np.sqrt(
                    np.linalg.pinv(X.T @ X)[1, 1] * residues / df
                )

            return GLS_coef, GLS_se

        GLS_coef, GLS_se = linear_regression(warped_screens, warped_intercept)
        df = warped_screens.shape[1] - 2
        GLS_p = 2 * stdtr(df, -np.abs(GLS_coef / GLS_se))
        np.fill_diagonal(GLS_p, 1)
