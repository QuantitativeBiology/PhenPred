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
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder
from statsmodels.stats.multitest import multipletests
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA


class DrugResponseBenchmark:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp

        self.data = data

        # Import MOFA dataset
        self.mofa_db = CLinesDatasetMOFA()

        # New drug response values
        self.drugresponse_new = pd.read_csv(
            f"{data_folder}/drugresponse_24Jul22.csv", index_col=0
        ).T

        # Fully imputed autoencoder dataset
        self.drugresponse_vae = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_imputed_drugresponse.csv.gz",
            index_col=0,
        )

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
            .intersection(set(self.mofa_db.imputed["drugresponse"].index))
            .intersection(set(self.drugresponse_mean.index))
        )
        self.samples = list(self.samples)

        # Intersection features
        self.features = (
            set(self.data.dfs["drugresponse"].columns)
            .intersection(set(self.drugresponse_new.columns))
            .intersection(set(self.drugresponse_vae.columns))
            .intersection(set(self.mofa_db.imputed["drugresponse"].columns))
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
                self.mofa_db.imputed["drugresponse"]
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

            _, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)

            sns.scatterplot(
                data=plot_df,
                x="original",
                y=y_var,
                alpha=0.75,
                color="#656565",
                ax=ax,
            )
            sns.regplot(
                data=plot_df,
                x=plot_df["original"],
                y=plot_df[y_var],
                scatter=False,
                color="#F2C500",
                truncate=True,
                ax=ax,
            )
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
            annot_text = f"R={r:.2g}; Rho={s:.2g}; MSE={mse:.2f}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right"
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_scatter_{y_var}"
            )

        # Bar plot
        plot_df = pd.DataFrame(corr_dict)

        for y_var in ["MSE", "Spearman's rho", "Pearson's r"]:
            _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

            sns.barplot(
                data=plot_df,
                x=y_var,
                y="method",
                order=["VAE", "MOFA", "mean"],
                orient="h",
                color="#656565",
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
        drespo_mofa = self.mofa_db.predicted["drugresponse"].copy()
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

        # Correlation dataframe
        for name, df in [
            ("VAE", drespo_vae),
            ("mofa", drespo_mofa),
            ("mean", drespo_mean),
        ]:
            df_corrs = pd.DataFrame(
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
                title=f"Comparison {name} with CTD2 (T-test p={ttest_stat[1]:.2e})",
                xlabel="Sample correlation (Pearson's r)",
                ylabel=f"Number of cell lines",
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr_with_vae_hist_{name}",
            )

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
            color="#ababab",
            orient="h",
            linewidth=0.3,
            fliersize=1,
            notch=True,
            saturation=1.0,
            showcaps=False,
            boxprops=dict(linewidth=0.5),
            whiskerprops=dict(linewidth=0.5),
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

        ax.set(
            title=f"Drug correlation with CTD2",
            xlabel="Drug correlation (spearman's rho)",
            ylabel=f"",
        )

        PhenPred.save_figure(
            f"{plot_folder}/drugresponse/{self.timestamp}_drug_correlation_boxplot",
        )
