import os
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder


class DrugResponseBenchmark:
    def __init__(
        self,
        timestamp,
        data,
        vae_imputed,
        mofa_imputed,
        move_diabetes_imputed=None,
        jamie_imputed=None,
        scvaeit_imputed=None,
        vae_imputed_7omics=None,
    ):
        self.timestamp = timestamp

        self.data = data
        self.vae_imputed = vae_imputed
        self.vae_imputed_7omics = vae_imputed_7omics
        self.mofa_imputed = mofa_imputed

        if move_diabetes_imputed is not None:
            self.move_diabetes_imputed = move_diabetes_imputed

        if jamie_imputed is not None:
            self.jamie_imputed = jamie_imputed
        if scvaeit_imputed is not None:
            self.scvaeit_imputed = scvaeit_imputed

        # New drug response values
        self.drugresponse_new = pd.read_csv(
            f"{data_folder}/drugresponse_24Jul22.csv", index_col=0
        ).T

        # Fully imputed autoencoder dataset
        self.drugresponse_vae = self.vae_imputed["drugresponse"]
        self.drugresponse_vae_7omics = self.vae_imputed_7omics["drugresponse"]
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

        if move_diabetes_imputed is not None:
            self.samples = self.samples.intersection(
                set(self.move_diabetes_imputed["drugresponse"].index)
            )
        if jamie_imputed is not None:
            self.samples = self.samples.intersection(
                set(self.jamie_imputed["drugresponse"].index)
            )
        if scvaeit_imputed is not None:
            self.samples = self.samples.intersection(
                set(self.scvaeit_imputed["drugresponse"].index)
            )
        if vae_imputed_7omics is not None:
            self.samples = self.samples.intersection(
                set(self.vae_imputed_7omics["drugresponse"].index)
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

        if move_diabetes_imputed is not None:
            self.features = self.features.intersection(
                set(self.move_diabetes_imputed["drugresponse"].columns)
            )
        if jamie_imputed is not None:
            self.features = self.features.intersection(
                set(self.jamie_imputed["drugresponse"].columns)
            )
        if scvaeit_imputed is not None:
            self.features = self.features.intersection(
                set(self.scvaeit_imputed["drugresponse"].columns)
            )
        if vae_imputed_7omics is not None:
            self.features = self.features.intersection(
                set(self.vae_imputed_7omics["drugresponse"].columns)
            )

        self.features = list(self.features)

        if not os.path.exists(f"{plot_folder}/drugresponse"):
            os.makedirs(f"{plot_folder}/drugresponse")

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
                self.move_diabetes_imputed["drugresponse"]
                .unstack()
                .loc[df_new_values.index]
                .rename("MOVE"),
                self.jamie_imputed["drugresponse"]
                .unstack()
                .loc[df_new_values.index]
                .rename("JAMIE"),
                self.scvaeit_imputed["drugresponse"]
                .unstack()
                .loc[df_new_values.index]
                .rename("scVAEIT"),
                self.drugresponse_mean.unstack()
                .loc[df_new_values.index]
                .rename("mean"),
                self.drugresponse_vae.unstack().loc[df_new_values.index].rename("MOSA"),
            ],
            axis=1,
        )

        # Scatter plots
        corr_dict = []
        for y_var in ["MOFA", "MOVE", "JAMIE", "scVAEIT", "mean", "MOSA"]:
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
                # title=f"Drug response prediction (N={plot_df.shape[0]:,})",
                xlabel="Measured (novel dataset)",
                ylabel=f"Reconstructed ({y_var})",
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
            annot_text = f"Pearson's r={r:.2g}\nSpearman's rho={s:.2g}\nMSE={mse:.2f}"
            ax.text(
                0.95, 0.1, annot_text, fontsize=6, transform=ax.transAxes, ha="right"
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
                order=["MOSA", "MOFA", "MOVE", "JAMIE", "scVAEIT", "mean"],
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

        # Box plot per drug
        m_res, s_res, r_res = [], [], []
        tmp_df = df_new_values.reset_index()
        tmp_df.columns = [
            "drug_id",
            "model_id",
            "original",
            "MOFA",
            "MOVE",
            "JAMIE",
            "scVAEIT",
            "mean",
            "MOSA",
        ]
        for drug in tmp_df["drug_id"].unique():
            sub_df = tmp_df[tmp_df["drug_id"] == drug]
            if sub_df.shape[0] < 5:
                continue

            mse_dict = {
                "drug_id": drug,
                "MOFA": mean_squared_error(sub_df["original"], sub_df["MOFA"]),
                "MOVE": mean_squared_error(sub_df["original"], sub_df["MOVE"]),
                "JAMIE": mean_squared_error(sub_df["original"], sub_df["JAMIE"]),
                "scVAEIT": mean_squared_error(sub_df["original"], sub_df["scVAEIT"]),
                "mean": mean_squared_error(sub_df["original"], sub_df["mean"]),
                "MOSA": mean_squared_error(sub_df["original"], sub_df["MOSA"]),
            }

            s_dict = {
                "drug_id": drug,
                "MOFA": stats.spearmanr(sub_df["original"], sub_df["MOFA"])[0],
                "MOVE": stats.spearmanr(sub_df["original"], sub_df["MOVE"])[0],
                "JAMIE": stats.spearmanr(sub_df["original"], sub_df["JAMIE"])[0],
                "scVAEIT": stats.spearmanr(sub_df["original"], sub_df["scVAEIT"])[0],
                "MOSA": stats.spearmanr(sub_df["original"], sub_df["MOSA"])[0],
            }
            r_dict = {
                "drug_id": drug,
                "MOFA": stats.pearsonr(sub_df["original"], sub_df["MOFA"])[0],
                "MOVE": stats.pearsonr(sub_df["original"], sub_df["MOVE"])[0],
                "JAMIE": stats.pearsonr(sub_df["original"], sub_df["JAMIE"])[0],
                "scVAEIT": stats.pearsonr(sub_df["original"], sub_df["scVAEIT"])[0],
                "MOSA": stats.pearsonr(sub_df["original"], sub_df["MOSA"])[0],
            }

            m_res.append(mse_dict)
            s_res.append(s_dict)
            r_res.append(r_dict)

        mse_res_df = pd.DataFrame(m_res).set_index("drug_id")
        spearman_res_df = pd.DataFrame(s_res).set_index("drug_id")
        pearson_res_df = pd.DataFrame(r_res).set_index("drug_id")
        mse_res_df["Metric"] = "MSE"
        spearman_res_df["Metric"] = "Spearman"
        pearson_res_df["Metric"] = "Pearson"
        plot_df = pd.concat([mse_res_df, spearman_res_df, pearson_res_df])
        plot_df = (
            pd.melt(
                plot_df,
                id_vars=["Metric"],
                value_vars=["MOFA", "MOVE", "JAMIE", "scVAEIT", "MOSA", "mean"],
            )
            .rename(columns={"variable": "Method", "value": "Value"})
            .dropna()
        )
        for metric in ["MSE", "Spearman", "Pearson"]:
            plot_df_metric = plot_df[plot_df["Metric"] == metric]
            _, ax = plt.subplots(1, 1, figsize=(2, 1), dpi=600)

            sns.boxplot(
                data=plot_df_metric,
                x="Value",
                y="Method",
                orient="h",
                saturation=0.8,
                linewidth=0.25,
                showfliers=True,
                fliersize=0.25,
                ax=ax,
            )
            ax.set(
                title=f"Drug response prediction per Drug",
                xlabel=metric,
                ylabel="Method",
            )

            PhenPred.save_figure(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_per_drug_boxplot_{metric}"
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

        # MOSA
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

        # MOVE
        drespo_move_diabetes = self.move_diabetes_imputed["drugresponse"].copy()
        drespo_move_diabetes.columns = [
            c.split(";")[1].upper() for c in drespo_move_diabetes
        ]
        if drop_duplicates:
            drespo_move_diabetes = drespo_move_diabetes.loc[
                :, ~drespo_move_diabetes.columns.duplicated(keep="last")
            ]

        # JAMIE
        drespo_jamie = self.jamie_imputed["drugresponse"].copy()
        drespo_jamie.columns = [c.split(";")[1].upper() for c in drespo_jamie]
        if drop_duplicates:
            drespo_jamie = drespo_jamie.loc[
                :, ~drespo_jamie.columns.duplicated(keep="last")
            ]

        # SCVAEIT
        drespo_scvaeit = self.scvaeit_imputed["drugresponse"].copy()
        drespo_scvaeit.columns = [c.split(";")[1].upper() for c in drespo_scvaeit]
        if drop_duplicates:
            drespo_scvaeit = drespo_scvaeit.loc[
                :, ~drespo_scvaeit.columns.duplicated(keep="last")
            ]

        # seven omics
        drespo_vae_7omics = self.vae_imputed_7omics["drugresponse"].copy()
        drespo_vae_7omics.columns = [c.split(";")[1].upper() for c in drespo_vae_7omics]
        if drop_duplicates:
            drespo_vae_7omics = drespo_vae_7omics.loc[
                :, ~drespo_vae_7omics.columns.duplicated(keep="last")
            ]

        # Mean
        drespo_mean = drespo_gdsc.fillna(drespo_gdsc.mean(axis=0))

        return (
            drespo_gdsc,
            drespo_ctd2,
            drespo_vae,
            drespo_mofa,
            drespo_move_diabetes,
            drespo_jamie,
            drespo_scvaeit,
            drespo_vae_7omics,
            drespo_mean,
        )

    def correlation_ctd2(self):
        (
            drespo_gdsc,
            drespo_ctd2,
            drespo_vae,
            drespo_mofa,
            drespo_move_diabetes,
            drespo_jamie,
            drespo_scvaeit,
            drespo_vae_7omics,
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
                            outofsample=(
                                "Out-of-sample"
                                if s in samples_without_drug
                                else "In-sample"
                            ),
                        ),
                    )
                    for s in samples
                ]
            ).drop_duplicates(subset=["sample"])
            for n, df in [
                ("MOSA", drespo_vae),
                ("MOFA", drespo_mofa),
                ("MOVE", drespo_move_diabetes),
                ("JAMIE", drespo_jamie),
                ("scVAEIT", drespo_scvaeit),
                ("MOSA_7omics", drespo_vae_7omics),
                ("mean", drespo_mean),
            ]
        }

        # Correlation dataframe
        for name in ["MOSA_7omics", "MOSA", "MOFA", "MOVE", "JAMIE", "scVAEIT", "mean"]:
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
                title=f"Comparison {name} with CTD2\n(Welch's t-test p={ttest_stat[1]:.2e})",
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
            x=f"MOFA_corr",
            y=f"MOSA_corr",
            z="MOSA_outofsample",
            plot_reg=False,
            plot_df=plot_df,
            discrete_pal=pal,
            scatter_kws=dict(edgecolor="w", linewidths=0.1, s=10, alpha=0.75),
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
            id_vars=["MOSA_outofsample"],
            var_name="method",
            value_name="corr",
        )

        _, ax = plt.subplots(1, 1, figsize=(1, 2.5), dpi=600)

        sns.boxplot(
            data=plot_df,
            x="MOSA_outofsample",
            y="corr",
            hue="method",
            hue_order=[
                "MOSA_7omics_corr",
                "MOSA_corr",
                "MOFA_corr",
                "MOVE_corr",
                "JAMIE_corr",
                "scVAEIT_corr",
                "mean_corr",
            ],
            orient="v",
            palette="tab10",
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
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        PhenPred.save_figure(
            f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr_methods_boxplot",
        )

        plot_df.to_csv(
            f"{plot_folder}/drugresponse/{self.timestamp}_predicted_ctd2_corr.csv"
        )

        ttest_stat = stats.ttest_ind(
            plot_df.query(
                "(MOSA_outofsample == 'In-sample') & (method == 'MOSA_corr')"
            )["corr"],
            plot_df.query(
                "(MOSA_outofsample == 'In-sample') & (method == 'mofa_corr')"
            )["corr"],
            equal_var=False,
        )
        print(f"MOSA vs MOFA (in-sample): {ttest_stat}")

        ttest_stat = stats.ttest_ind(
            plot_df.query(
                "(MOSA_outofsample == 'Out-of-sample') & (method == 'MOSA_corr')"
            )["corr"],
            plot_df.query(
                "(MOSA_outofsample == 'Out-of-sample') & (method == 'mofa_corr')"
            )["corr"],
            equal_var=False,
        )
        print(f"MOSA vs MOFA (Out-of-sample): {ttest_stat}")

    def compare_drug_predictions(self):
        (
            drespo_gdsc,
            drespo_ctd2,
            drespo_vae,
            drespo_mofa,
            drespo_move_diabetes,
            drespo_jamie,
            drespo_scvaeit,
            drespo_vae_7omics,
            drespo_mean,
        ) = self.ctd2_parse_drugresponse_dfs()

        # Overlap of drugs and samples
        drugs = drespo_ctd2.columns.intersection(drespo_vae.columns).tolist()

        # Union of samples
        samples = (
            drespo_gdsc.index.union(drespo_ctd2.index)
            .union(drespo_vae.index)
            .union(drespo_mofa.index)
            .union(drespo_move_diabetes.index)
            .union(drespo_jamie.index)
            .union(drespo_scvaeit.index)
            .union(drespo_vae_7omics.index)
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
                    ("MOSA", drespo_vae),
                    ("MOFA", drespo_mofa),
                    ("MOVE", drespo_move_diabetes),
                    ("JAMIE", drespo_jamie),
                    ("scVAEIT", drespo_scvaeit),
                    ("MOSA_7omics", drespo_vae_7omics),
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
