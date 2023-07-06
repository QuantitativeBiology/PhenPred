import PhenPred
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PhenPred.vae import plot_folder
from PhenPred.vae.PlotUtils import GIPlot


class MismatchBenchmark:
    def __init__(self, timestamp, data, vae_predicted):
        self.timestamp = timestamp

        self.data = data
        self.vae_predicted = vae_predicted

    def run(self):
        self.drug_response()

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
