import sys

from sympy import plot

sys.path.extend(["/home/egoncalves/PhenPred"])

import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as plticker
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Utils import LModel
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA


_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class ProteomicsBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Import MOFA dataset
        self.mofa_db = CLinesDatasetMOFA()

        # Sample sheet
        self.ss = pd.read_csv(f"{data_folder}/cmp_model_list_20230307.csv", index_col=0)

        self.ss_ccell = pd.read_csv(
            f"{data_folder}/samplesheet_cancercell.csv", index_col=0
        )

        # Original dataset
        self.df_original = pd.read_csv(f"{data_folder}/proteomics.csv", index_col=0).T

        # Fully imputed autoencoder dataset
        self.df_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_proteomics.csv.gz", index_col=0
        )

        # MOFA imputed dataset
        self.df_mofa_imputed = self.mofa_db.imputed["proteomics"]
        self.df_mofa_predicted = self.mofa_db.predicted["proteomics"]

        # Independent proteomics dataset - CCLE
        self.df_ccle = pd.read_csv(f"{data_folder}/proteomics_ccle.csv", index_col=0).T
        self.df_ccle.index = (
            pd.Series(self.df_ccle.index)
            .replace(
                self.ss["CCLE_ID"]
                .reset_index()
                .dropna()
                .set_index("CCLE_ID")["model_id"]
                .to_dict()
            )
            .values
        )

        # Samples and features intersection
        self.samples = (
            set(self.df_original.index)
            .intersection(set(self.df_vae.index))
            .intersection(set(self.df_mofa_imputed.index))
            .intersection(set(self.df_mofa_predicted.index))
            .intersection(set(self.df_ccle.index))
        )

        self.features = (
            set(self.df_original.columns)
            .intersection(set(self.df_vae.columns))
            .intersection(set(self.df_mofa_imputed.columns))
            .intersection(set(self.df_mofa_predicted.columns))
            .intersection(set(self.df_ccle.columns))
        )

        # CRISPR original dataset
        self.df_crispr = pd.read_csv(f"{data_folder}/crisprcas9.csv", index_col=0).T

        # Drug response original dataset
        self.df_drug = pd.read_csv(f"{data_folder}/drugresponse.csv", index_col=0).T

        # Genomics
        self.df_genomics = pd.read_csv(f"{data_folder}/genomics.csv", index_col=0).T

        print(
            f"[{_timestamp}] Samples = {len(self.samples)}, Features = {len(self.features)}"
        )

    def run(self):
        self.compare_imputed_ccle()
        self.ccle_compare_by_genes()
        self.associations()

    def place_imputed_values_in_nans(self):
        df_original = self.df_original.copy().reindex(
            index=self.samples, columns=self.features
        )

        df_original_mofa_imputed = self.df_mofa_imputed.reindex(
            index=self.samples, columns=self.features
        )

        df_original_mofa_predicted = self.df_mofa_predicted.reindex(
            index=self.samples, columns=self.features
        )

        df_original_vae_imputed = df_original.copy().fillna(
            self.df_vae.reindex(index=self.samples, columns=self.features)
        )

        df_original_vae_predicted = self.df_vae.reindex(
            index=self.samples, columns=self.features
        )

        df_original_mean_imputed = df_original.copy().fillna(df_original.mean())

        return dict(
            original=df_original,
            mofa_imputed=df_original_mofa_imputed,
            mofa_predicted=df_original_mofa_predicted,
            vae_imputed=df_original_vae_imputed,
            vae_predicted=df_original_vae_predicted,
            mean=df_original_mean_imputed,
        )

    def compare_imputed_ccle(self):
        # Proteomic datasets
        df_imputed = self.place_imputed_values_in_nans()
        df_ccle = self.df_ccle.reindex(index=self.samples, columns=self.features)

        # Standardize
        df_imputed = {
            k: stats.zscore(df, nan_policy="omit") for k, df in df_imputed.items()
        }
        df_ccle = stats.zscore(df_ccle, nan_policy="omit")

        # Correlation dataframe
        df_corrs = pd.DataFrame(
            [
                two_vars_correlation(
                    df_ccle.loc[s], df.loc[s], extra_fields=dict(sample=s, impute=k)
                )
                for s in self.samples
                for k, df in df_imputed.items()
            ]
        )

        # Boxplot
        _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

        sns.boxplot(
            data=df_corrs,
            x="corr",
            y="impute",
            palette="tab10",
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
            title=f"Sample correlation between imputed and CCLE (N={df_ccle.shape[0]:,})",
            xlabel="Pearson's r",
            ylabel=f"",
        )

        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
        ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=-1)

        plt.savefig(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_boxplot.pdf",
            bbox_inches="tight",
        )
        plt.close()

        # Grid plot
        plot_df = pd.pivot_table(
            df_corrs, index="sample", columns="impute", values="corr"
        )
        plot_df = pd.melt(
            plot_df.reset_index(),
            value_vars=[
                "mean",
                "mofa_imputed",
                "mofa_predicted",
                "vae_imputed",
                "vae_predicted",
            ],
            var_name="impute",
            value_name="corr",
            id_vars=["sample", "original"],
        )

        def annotate(data, **kws):
            rmse = sqrt(mean_squared_error(data["original"], data["corr"]))
            s, _ = stats.spearmanr(
                data["original"],
                data["corr"],
            )
            r, _ = stats.pearsonr(
                data["original"],
                data["corr"],
            )
            ax = plt.gca()
            ax.text(
                0.95,
                0.05,
                f"R={r:.2g}; Rho={s:.2g}; RMSE={rmse:.2f}",
                fontsize=6,
                transform=ax.transAxes,
                ha="right",
            )
            ax.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        g = sns.FacetGrid(plot_df, col="impute")

        g.map_dataframe(sns.scatterplot, x="original", y="corr")
        g.map_dataframe(annotate)

        plt.savefig(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_facetgrid.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def proteomics_genomics(self):
        covs = pd.concat(
            [
                self.ss["growth_properties"].str.get_dummies(),
                self.ss["tissue"].str.get_dummies()[
                    ["Haematopoietic and Lymphoid", "Lung"]
                ],
            ],
            axis=1,
        ).dropna()

        samples = list(
            set(self.df_vae.index)
            .intersection(self.df_genomics.index)
            .intersection(covs.index)
        )

        X = self.df_genomics.loc[samples]
        X = X.loc[:, X.sum() > 10]

        Y = self.df_vae.loc[samples]

        lm_genomics_vae = LModel(
            Y=Y,
            X=X,
            M=covs.loc[samples],
        ).fit_matrix()

        return lm_genomics_vae

    def associations(self):
        z_vars = pd.concat(
            [self.df_genomics, pd.get_dummies(self.ss["tissue"]).astype(int)], axis=1
        )

        for x_id, y_id, z_id, z_type in [
            (
                "MET",
                "1403;AZD6094;GDSC1",
                "gain.cnaPANCAN129..MET.",
                "drug",
            ),
            (
                "ERBB2",
                "ERBB2",
                "gain.cnaPANCAN301..CDK12.ERBB2.MED24.",
                "crispr",
            ),
            (
                "ERBB2",
                "1558;Lapatinib;GDSC2",
                "gain.cnaPANCAN301..CDK12.ERBB2.MED24.",
                "drug",
            ),
            (
                "EGFR",
                "EGFR",
                "EGFR_mut",
                "crispr",
            ),
            (
                "EGFR",
                "EGFR",
                "gain.cnaPANCAN124..EGFR.",
                "crispr",
            ),
            (
                "EGFR",
                "1032;Afatinib;GDSC2",
                "gain.cnaPANCAN124..EGFR.",
                "drug",
            ),
            (
                "RPL22L1",
                "WRN",
                "msi_status",
                "crispr",
            ),
            (
                "TP53",
                "1047;Nutlin-3a (-);GDSC2",
                "TP53_mut",
                "drug",
            ),
        ]:
            # x_id, y_id, z_id, z_type = (
            #     "TP53",
            #     "1047;Nutlin-3a (-);GDSC2",
            #     "TP53_mut",
            #     "drug",
            # )

            # Build plot dataframe
            plot_df = pd.concat(
                [
                    self.df_vae[x_id].rename(f"{x_id}_vae"),
                    self.df_original[x_id].rename(f"{x_id}_orig"),
                    z_vars[z_id].replace({0: "WT", 1: z_id}),
                ],
                axis=1,
            )
            plot_df[y_id] = (
                self.df_crispr[y_id] if z_type == "crispr" else self.df_drug[y_id]
            )

            plot_df.dropna(subset=[f"{x_id}_vae", y_id, z_id], inplace=True)

            plot_df["predicted"] = plot_df[f"{x_id}_orig"].isnull()
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
                0: "#E1E1E1",
            }, ["WT", z_id]

            # Plot
            g = GIPlot.gi_regression_marginal(
                x=f"{x_id}_vae",
                y=f"{y_id}",
                z=z_id,
                style="predicted",
                plot_df=plot_df,
                discrete_pal=pal,
                hue_order=pal_order,
                legend_title=f"{z_id}",
                scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
            )

            g.ax_joint.set_xlabel(f"{x_id} Proteomics (VAE)")
            g.ax_joint.set_ylabel(
                f"{y_id}\n{'CRISPR-Cas9' if z_type == 'crispr' else 'Drug'} (Observed)"
            )

            plt.gcf().set_size_inches(2, 2)

            plt.savefig(
                f"{plot_folder}/proteomics/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}.pdf",
                bbox_inches="tight",
            )
            plt.close("all")

    def ccle_compare_by_genes(self):
        features = list(set(self.df_vae).intersection(self.df_ccle))
        samples = list(set(self.df_vae.index).intersection(self.df_ccle.index))

        # Correlation dataframe
        df_corrs = pd.DataFrame(
            [
                two_vars_correlation(
                    self.df_ccle.loc[samples, f],
                    df.reindex(index=samples)[f],
                    method="spearman",
                    extra_fields=dict(protein=f, impute=n),
                )
                for f in features
                for n, df in [("vae", self.df_vae), ("original", self.df_original)]
            ]
        )

        _, ax = plt.subplots(1, 1, figsize=(2.5, 1), dpi=600)

        sns.boxplot(
            data=df_corrs,
            x="corr",
            y="impute",
            palette="tab10",
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
            title=f"Gene correlation with CCLE",
            xlabel="Pearson's r",
            ylabel=f"",
        )

        plt.savefig(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_by_gene_boxplot.pdf",
            bbox_inches="tight",
        )
        plt.close()
