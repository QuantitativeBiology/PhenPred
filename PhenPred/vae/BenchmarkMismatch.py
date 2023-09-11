import os
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from PhenPred.vae.PlotUtils import GIPlot
from PhenPred.vae import plot_folder, data_folder


class MismatchBenchmark:
    def __init__(self, timestamp, data, vae_predicted, cvtest_datasets):
        self.timestamp = timestamp

        self.data = data
        self.vae_predicted = vae_predicted
        self.cvtest_datasets = cvtest_datasets

        if not os.path.exists(f"{plot_folder}/mismatch"):
            os.makedirs(f"{plot_folder}/mismatch")

        # DepMap predictability
        self.chronos_pred = pd.read_csv(
            f"{data_folder}/Chronos_Combined_predictability_results.csv"
        )
        self.chronos_pred["gene"] = self.chronos_pred["gene"].apply(
            lambda x: x.split(" ")[0]
        )
        self.chronos_best_pred = self.chronos_pred.loc[
            self.chronos_pred["best"]
        ].set_index("gene")

    def run(self):
        self.drug_response()
        self.top_reconstructed_genes()
        self.top_reconstructed_drug()

    def top_reconstructed_drug(self):
        # plot_df
        m_true = self.data.dfs["drugresponse"]
        m_pred = self.cvtest_datasets["drugresponse"].reindex(
            index=m_true.index,
            columns=m_true.columns,
        )

        m_mse = (m_true - m_pred) ** 2
        m_pearson = m_true.corrwith(m_pred, axis=0)
        m_skew = m_true.skew()

        df = pd.concat(
            [
                m_mse.mean().rename("mse"),
                m_true.mean().rename("mean"),
                m_skew.rename("skew"),
                m_pearson.rename("pearson"),
            ],
            axis=1,
        )

        # Top reconstructed genes
        plot_df = df.sort_values("pearson", ascending=False)
        plot_df["index"] = plot_df.reset_index().index
        plot_df = plot_df.sort_values("skew", ascending=False)

        _, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

        sns.scatterplot(
            x=plot_df["index"],
            y=plot_df["pearson"],
            hue=plot_df["skew"],
            alpha=0.5,
            palette="viridis",
            linewidth=0,
            edgecolor=None,
            s=3,
        )

        ax.set_xlabel("Drugs")
        ax.set_ylabel("MOVE reconstruction (pearson's r)")

        labels = [
            ax.text(
                x=plot_df.loc[g, "index"],
                y=plot_df.loc[g, "pearson"],
                s=g.split(";")[1],
                fontsize=5,
                ha="center",
                va="center",
            )
            for g in plot_df.query("skew < -1.5").index
        ]
        adjust_text(
            labels,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            ax=ax,
        )

        PhenPred.save_figure(
            f"{plot_folder}/mismatch/{self.timestamp}_top_reconstructed_scatterplot_drespo",
        )

    def top_reconstructed_genes(self):
        # plot_df
        m_true = self.data.dfs["crisprcas9"]
        m_pred = self.cvtest_datasets["crisprcas9"].reindex(
            index=m_true.index,
            columns=m_true.columns,
        )

        m_mse = (m_true - m_pred) ** 2
        m_pearson = m_true.corrwith(m_pred, axis=0)
        m_skew = m_true.skew()

        df = pd.concat(
            [
                m_mse.mean().rename("mse"),
                m_true.mean().rename("mean"),
                m_skew.rename("skew"),
                m_pearson.rename("pearson"),
                m_true.count().rename("count"),
                (m_true < -0.5).sum().rename("ess"),
                self.chronos_best_pred["pearson"].rename("chronos"),
            ],
            axis=1,
        )

        df["density"] = GIPlot.density_interpolate(
            df["chronos"].values,
            df["pearson"].values,
        )

        # MOVE vs Chronos pearson's r
        _, ax = plt.subplots(1, 1, figsize=(3, 2.5))

        GIPlot.gi_continuous_plot(
            plot_df=df,
            x="chronos",
            y="pearson",
            z="density",
            cmap="viridis",
            mid_point=df["density"].mean(),
            ax=ax,
        )

        ax.axline((0.5, 0.5), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        ax.set_ylabel("MOVE reconstruction (pearson's r)")
        ax.set_xlabel("Chronos prediction (best pearson's r)")
        ax.set_title("CRISPR-Cas9 predictability")

        PhenPred.save_figure(
            f"{plot_folder}/mismatch/{self.timestamp}_predictability_scatterplot",
        )

        # Top reconstructed genes
        plot_df = df.sort_values("pearson", ascending=False).query("ess >= 5")
        plot_df["index"] = plot_df.reset_index().index
        plot_df = plot_df.sort_values("skew", ascending=False)

        _, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))

        sns.scatterplot(
            x=plot_df["index"],
            y=plot_df["pearson"],
            hue=plot_df["skew"],
            alpha=0.5,
            palette="viridis",
            linewidth=0,
            edgecolor=None,
            s=3,
        )

        ax.set_xlabel("CRISPR-Cas9 genes")
        ax.set_ylabel("MOVE reconstruction (pearson's r)")

        genes = set(plot_df.query("skew < -3 & ess >= 5").index)
        genes_label = [
            ax.text(
                x=plot_df.loc[g, "index"],
                y=plot_df.loc[g, "pearson"],
                s=g,
                fontsize=5,
                ha="center",
                va="center",
            )
            for g in genes
        ]
        adjust_text(
            genes_label,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
            ax=ax,
        )

        PhenPred.save_figure(
            f"{plot_folder}/mismatch/{self.timestamp}_top_reconstructed_scatterplot",
        )

    def drug_response(self):
        # Drug response
        m_true = self.data.dfs["drugresponse"]
        m_pred = self.vae_predicted["drugresponse"].reindex(
            index=m_true.index,
            columns=m_true.columns,
        )

        # MSE
        m_mse = (m_true - m_pred) ** 2
        print(m_mse.unstack().dropna().sort_values(ascending=False).head(30))

        # Study case
        for drug, sample, df in [
            ("2508;Trametinib;GDSC2", "SIDM01134", "drugresponse"),
            ("1909;Venetoclax;GDSC2", "SIDM00391", "drugresponse"),
            ("2125;Mcl1_7350;GDSC2", "SIDM00461", "drugresponse"),
        ]:
            # drug, sample, df = "2125;Mcl1_7350;GDSC2", "SIDM00461", "drugresponse"
            drug_targets = self.data.drug_targets[drug].split(";")

            m_true = self.data.dfs[df]
            m_pred = self.vae_predicted[df].reindex(
                index=m_true.index,
                columns=m_true.columns,
            )

            # Drug regression
            plot_df = pd.concat(
                [
                    m_true[drug].rename("true"),
                    m_pred[drug].rename("pred"),
                ],
                axis=1,
            )

            g = GIPlot.gi_regression(
                plot_df=plot_df,
                x_gene="true",
                y_gene="pred",
            )

            g.ax_joint.scatter(
                plot_df.loc[sample, "true"],
                plot_df.loc[sample, "pred"],
                color="red",
                marker="x",
                s=6,
            )

            g.ax_joint.text(
                plot_df.loc[sample, "true"],
                plot_df.loc[sample, "pred"],
                sample,
                color="red",
                fontsize=6,
            )

            g.ax_marg_x.set_title(drug)

            g.ax_joint.set_xlabel("Measured (IC50)")
            g.ax_joint.set_ylabel("Predicted (IC50)")

            plt.gcf().set_size_inches(2, 2)

            PhenPred.save_figure(
                f"{plot_folder}/mismatch/{self.timestamp}_{drug}_regression",
            )

            # Top correlated drugs
            corr_thres = 0.7

            drug_corr = m_true.corrwith(m_true[drug])
            drug_corr = drug_corr[drug_corr > corr_thres]

            plot_df = m_true.loc[sample, drug_corr.index].sort_values().dropna()

            fig, ax = plt.subplots(1, 1, figsize=(2, 1))

            sns.barplot(
                x=plot_df.values,
                y=plot_df.index,
                color="grey",
                orient="h",
                ax=ax,
            )

            ax.set_xlabel(f"{sample}\nmeasured (IC50)")
            ax.set_ylabel("")
            ax.set_title(f"{drug}\ntop correlated drugs (r > {corr_thres})")

            PhenPred.save_figure(
                f"{plot_folder}/mismatch/{self.timestamp}_{drug}_top_correlated_barplot",
            )

            # Correlation with CRISPR
            plot_df = pd.concat(
                [
                    m_true[drug].rename("drug"),
                    self.data.dfs["crisprcas9"].reindex(columns=drug_targets),
                ],
                axis=1,
            )
            plot_df = plot_df.assign(sample=[int(i == sample) for i in plot_df.index])
            plot_df = pd.melt(
                plot_df, id_vars=["drug", "sample"], value_vars=drug_targets
            ).sort_values("sample")

            g = sns.FacetGrid(
                plot_df,
                col="variable",
                aspect=1,
                height=2.0,
            )

            g.map_dataframe(
                sns.scatterplot,
                x="drug",
                y="value",
                hue="sample",
                style="sample",
                palette={0: "grey", 1: "red"},
                s=6,
            )

            g.set_xlabels(f"{drug}\nmeasured (IC50)")
            g.set_ylabels("Measured\n(CRISPR-Cas9 scaled log2 fold change)")

            PhenPred.save_figure(
                f"{plot_folder}/mismatch/{self.timestamp}_{drug}_crispr_correlation",
            )
