import sys

sys.path.extend(["/home/egoncalves/PhenPred"])

import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder
from statsmodels.stats.multitest import multipletests
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA


class DrugResponseBenchmark:
    def __init__(self, timestamp, data, vae_imputed, mofa_imputed):
        self.timestamp = timestamp

        self.data = data
        self.vae_imputed = vae_imputed
        self.mofa_imputed = mofa_imputed

        # New drug response values
        self.drugresponse_new = pd.read_csv(
            f"{data_folder}/drugresponse_24Jul22.csv", index_col=0
        ).T

        # Fully imputed autoencoder dataset
        self.drugresponse_vae = self.vae_imputed["drugresponse"]

        # Mean imputed dataset
        self.drugresponse_mean = self.data.dfs["drugresponse"].fillna(
            self.data.dfs["drugresponse"].mean()
        )

        # CTD2
        self.drugresponse_ctd2 = pd.read_csv(
            f"{data_folder}/CTRPv2.0_AUC_parsed.csv.gz", index_col=0
        ).T

        # Intersection samples
        self.samples = (
            set(self.data.dfs["drugresponse"].index)
            .intersection(set(self.drugresponse_new.index))
            .intersection(set(self.drugresponse_vae.index))
            .intersection(set(self.mofa_imputed["drugresponse"].index))
            .intersection(set(self.drugresponse_mean.index))
        )
        self.samples = list(self.samples)

        # Intersection features
        self.features = (
            set(self.data.dfs["drugresponse"].columns)
            .intersection(set(self.drugresponse_new.columns))
            .intersection(set(self.drugresponse_vae.columns))
            .intersection(set(self.mofa_imputed["drugresponse"].columns))
            .intersection(set(self.drugresponse_mean.columns))
        )
        self.features = list(self.features)

    def run(self):
        self.correlation_new_values()
        self.correlation_ctd2()
        self.compare_drug_predictions()

    def correlation_new_values(self):
        # Define subset of newly measured drug - cell line pairs
        df_new_values = (
            self.drugresponse_new.loc[self.samples, self.features].unstack().dropna()
        )
        df_new_values = df_new_values.loc[
            self.data.dfs["drugresponse"]
            .loc[self.samples, self.features]
            .unstack()
            .loc[df_new_values.index]
            .isna()
        ]
        df_new_values = pd.concat(
            [
                df_new_values.rename("original"),
                self.mofa_imputed["drugresponse"]
                .unstack()
                .loc[df_new_values.index]
                .rename("MOFA"),
                self.drugresponse_mean.unstack()
                .loc[df_new_values.index]
                .rename("mean"),
                self.drugresponse_vae.unstack().loc[df_new_values.index].rename("VAE"),
            ],
            axis=1,
        )

        # Scatter plots
        corr_dict = []
        for y_var in ["MOFA", "mean", "VAE"]:
            plot_df = df_new_values[["original", y_var]].dropna()

            _, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=600)

            sns.scatterplot(
                data=plot_df,
                x="original",
                y=y_var,
                s=1.5,
                alpha=0.50,
                color="#656565",
                linewidth=0.1,
                ax=ax,
            )

            sns.kdeplot(
                data=plot_df,
                x="original",
                y=y_var,
                levels=5,
                color="#fc8d62",
                linewidths=0.5,
                ax=ax,
            )

            # sns.regplot(
            #     data=plot_df,
            #     x="original",
            #     y=y_var,
            #     line_kws={"color": "#fc8d62", "linewidth": 0.5},
            #     scatter=False,
            #     truncate=True,
            #     ax=ax,
            # )

            # same axes limits and step sizes
            ax_min, ax_max = (
                min(plot_df["original"].min(), plot_df[y_var].min()),
                max(plot_df["original"].max(), plot_df[y_var].max()),
            )
            ax.set_xlim([ax_min, ax_max])
            ax.set_ylim([ax_min, ax_max])
            ax.set_xticks(ax.get_yticks())
            ax.set_yticks(ax.get_xticks())

            ax.set(
                title=f"Drug response prediction (N={plot_df.shape[0]:,})",
                xlabel="Measured (novel dataset)",
                ylabel=f"Predicted ({y_var})",
            )

            mse = mean_squared_error(plot_df["original"], plot_df[y_var])
            s, _ = stats.spearmanr(
                plot_df["original"],
                plot_df[y_var],
            )
            r, _ = stats.pearsonr(
                plot_df["original"],
                plot_df[y_var],
            )
            corr_dict.append(
                {
                    "MSE": mse,
                    "Spearman's rho": s,
                    "Pearson's r": r,
                    "method": y_var,
                }
            )
            annot_text = f"r={r:.2g}; rho={s:.2g}; MSE={mse:.2f}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right"
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_scatter_{y_var}"
            )

        # Bar plot
        plot_df = pd.DataFrame(corr_dict)

        for y_var in ["MSE", "Spearman's rho", "Pearson's r"]:
            _, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=600)

            sns.barplot(
                data=plot_df,
                x=y_var,
                y="method",
                order=["VAE", "MOFA", "mean"],
                orient="h",
                color="black",
                saturation=0.8,
                ax=ax,
            )

            ax.set(
                title=f"Drug response prediction (N={df_new_values.shape[0]:,})",
                xlabel=y_var,
                ylabel="Imputation method",
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_{y_var}_barplot"
            )

    def ctd2_parse_drugresponse_dfs(self, drop_duplicates=True):
        # Original GDSC
        drespo_gdsc = self.data.dfs["drugresponse"].copy()
        drespo_gdsc.columns = [c.split(";")[1].upper() for c in drespo_gdsc]
        if drop_duplicates:
            drespo_gdsc = drespo_gdsc.loc[
                :, ~drespo_gdsc.columns.duplicated(keep="last")
            ]

        # CTD2
        drespo_ctd2 = self.drugresponse_ctd2.copy()

        # VAE
        drespo_vae = self.drugresponse_vae.copy()
        drespo_vae.columns = [c.split(";")[1].upper() for c in drespo_vae]
        if drop_duplicates:
            drespo_vae = drespo_vae.loc[:, ~drespo_vae.columns.duplicated(keep="last")]

        # MOFA
        drespo_mofa = self.mofa_imputed["drugresponse"].copy()
        drespo_mofa.columns = [c.split(";")[1].upper() for c in drespo_mofa]
        if drop_duplicates:
            drespo_mofa = drespo_mofa.loc[
                :, ~drespo_mofa.columns.duplicated(keep="last")
            ]

        # Mean
        drespo_mean = drespo_gdsc.fillna(drespo_gdsc.mean(axis=0))

        return drespo_gdsc, drespo_ctd2, drespo_vae, drespo_mofa, drespo_mean

    def correlation_ctd2(self):
        (
            drespo_gdsc,
            drespo_ctd2,
            drespo_vae,
            drespo_mofa,
            drespo_mean,
        ) = self.ctd2_parse_drugresponse_dfs()

        # Overlap of drugs and samples
        drugs = drespo_vae.columns.intersection(drespo_ctd2.columns).tolist()
        samples = drespo_vae.index.intersection(drespo_ctd2.index).tolist()

        # Samples without drug response
        samples_without_drug = set(drespo_gdsc.index[drespo_gdsc.isna().all(axis=1)])

        # Correlation dfs
        df_corrs_methods = {
            n: pd.DataFrame(
                [
                    two_vars_correlation(
                        df.loc[s, drugs],
                        drespo_ctd2.loc[s, drugs],
                        method="pearson",
                        extra_fields=dict(
                            sample=s,
                            outofsample="Out-of-sample"
                            if s in samples_without_drug
                            else "In-sample",
                        ),
                    )
                    for s in samples
                ]
            )
            for n, df in [
                ("VAE", drespo_vae),
                ("mofa", drespo_mofa),
                ("mean", drespo_mean),
            ]
        }

        # Correlation dataframe
        for name in ["VAE"]:
            df_corrs = df_corrs_methods[name]

            ttest_stat = stats.ttest_ind(
                df_corrs.query("outofsample == 'In-sample'")["corr"],
                df_corrs.query("outofsample == 'Out-of-sample'")["corr"],
                equal_var=False,
            )

            # histogram coloured by out-of-sample
            _, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

            g = sns.histplot(
                data=df_corrs,
                x="corr",
                hue="outofsample",
                hue_order=["Out-of-sample", "In-sample"],
                palette=["#80b1d3", "#fc8d62"],
                alpha=0.8,
                ax=ax,
            )

            g.set(
                title=f"Comparison {name} with CTD2\n(T-test p={ttest_stat[1]:.2e})",
                xlabel=f"Cell line correlation (Pearson's r) across {len(drugs)} drugs\nReconstructed (IC50s) vs CTD2 (AUCs)",
                ylabel=f"Number of cell lines",
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr_with_vae_hist_{name}",
            )

        # Comparison of correlations with methods
        plot_df = pd.concat(
            [
                df_corrs_methods[n].set_index("sample").add_prefix(f"{n}_")
                for n in df_corrs_methods
            ],
            axis=1,
        ).dropna()

        # Regression
        pal = dict(zip(["In-sample", "Out-of-sample"], ["#80b1d3", "#fc8d62"]))

        g = GIPlot.gi_regression_marginal(
            x=f"mofa_corr",
            y=f"VAE_corr",
            z="VAE_outofsample",
            plot_reg=False,
            plot_df=plot_df,
            discrete_pal=pal,
            scatter_kws=dict(edgecolor="w", lw=0.1, s=10, alpha=0.75),
        )

        g.ax_joint.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        g.ax_joint.legend_.remove()

        plt.gcf().set_size_inches(2, 2)

        PhenPred.save_figure(
            f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr_methods_regression",
        )

        # Boxplot
        plot_df = plot_df.melt(
            value_vars=[c for c in plot_df if c.endswith("_corr")],
            id_vars=["VAE_outofsample"],
            var_name="method",
            value_name="corr",
        )

        _, ax = plt.subplots(1, 1, figsize=(1, 2.5), dpi=600)

        sns.boxplot(
            data=plot_df,
            x="VAE_outofsample",
            y="corr",
            hue="method",
            orient="v",
            palette="tab20c",
            linewidth=0.3,
            fliersize=1,
            notch=True,
            saturation=1.0,
            showcaps=False,
            boxprops=dict(linewidth=0.5, edgecolor="black"),
            whiskerprops=dict(linewidth=0.5, color="black"),
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markersize=1.0,
                linestyle="none",
                markeredgecolor="none",
                alpha=0.6,
            ),
            medianprops=dict(linestyle="-", linewidth=0.5),
            ax=ax,
        )

        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

        ax.set(
            xlabel="",
            ylabel="Cell line correlation (Pearson's r)\nReconstructed (IC50s) vs CTD2 (AUCs)",
        )

        PhenPred.save_figure(
            f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr_methods_boxplot",
        )

        ttest_stat = stats.ttest_ind(
            plot_df.query("(VAE_outofsample == 'In-sample') & (method == 'VAE_corr')")[
                "corr"
            ],
            plot_df.query("(VAE_outofsample == 'In-sample') & (method == 'mofa_corr')")[
                "corr"
            ],
            equal_var=False,
        )
        print(f"VAE vs mofa (in-sample): {ttest_stat}")

        ttest_stat = stats.ttest_ind(
            plot_df.query(
                "(VAE_outofsample == 'Out-of-sample') & (method == 'VAE_corr')"
            )["corr"],
            plot_df.query(
                "(VAE_outofsample == 'Out-of-sample') & (method == 'mofa_corr')"
            )["corr"],
            equal_var=False,
        )
        print(f"VAE vs mofa (Out-of-sample): {ttest_stat}")

    def compare_drug_predictions(self):
        (
            drespo_gdsc,
            drespo_ctd2,
            drespo_vae,
            drespo_mofa,
            drespo_mean,
        ) = self.ctd2_parse_drugresponse_dfs()

        # Overlap of drugs and samples
        drugs = drespo_ctd2.columns.intersection(drespo_vae.columns).tolist()

        # Union of samples
        samples = (
            drespo_gdsc.index.union(drespo_ctd2.index)
            .union(drespo_vae.index)
            .union(drespo_mofa.index)
            .union(drespo_mean.index)
            .tolist()
        )

        # Correlation dataframe
        df_corrs = pd.DataFrame(
            [
                two_vars_correlation(
                    df.reindex(index=samples)[d],
                    drespo_ctd2.reindex(index=samples)[d],
                    method="spearman",
                    extra_fields=dict(drug=d, df=df_name, samples="Samples all"),
                )
                for d in drugs
                for (df_name, df) in [
                    ("GDSC", drespo_gdsc),
                    ("VAE", drespo_vae),
                    ("mofa", drespo_mofa),
                    ("mean", drespo_mean),
                ]
            ]
        )

        # Plot
        _, ax = plt.subplots(1, 1, figsize=(2.5, 1), dpi=600)

        sns.boxplot(
            data=df_corrs,
            x="corr",
            y="df",
            orient="h",
            linewidth=0.3,
            fliersize=1,
            notch=True,
            saturation=1.0,
            showcaps=False,
            boxprops=dict(linewidth=0.5, facecolor="white", edgecolor="black"),
            whiskerprops=dict(linewidth=0.5, color="black"),
            flierprops=dict(
                marker="o",
                markerfacecolor="black",
                markersize=1.0,
                linestyle="none",
                markeredgecolor="none",
                alpha=0.6,
            ),
            medianprops=dict(linestyle="-", linewidth=0.5, color="#fc8d62"),
            ax=ax,
        )

        ax.set(
            title=f"Drug correlation with CTD2",
            xlabel="Drug correlation (spearman's rho)",
            ylabel=f"",
        )

        PhenPred.save_figure(
            f"{plot_folder}/drugresponse/{self.timestamp}_drug_correlation_boxplot",
        )
