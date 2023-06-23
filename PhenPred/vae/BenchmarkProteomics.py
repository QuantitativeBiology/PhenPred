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


class ProteomicsBenchmark:
    def __init__(self, timestamp, data):
        self.timestamp = timestamp

        # Data
        self.data = data

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
        self.cnv = pd.read_csv(f"{data_folder}/cnv_summary_20230303.csv")
        self.cnv["cn_category"] = pd.Categorical(
            self.cnv["cn_category"],
            categories=["Neutral", "Deletion", "Loss", "Gain", "Amplification"],
            ordered=True,
        )
        self.cnv = self.cnv.sort_values("cn_category")

        self.df_cnv = pd.pivot_table(
            self.cnv,
            index="model_id",
            columns="symbol",
            values="cn_category",
            aggfunc="first",
        )

        # Transcriptomics
        self.df_gexp = pd.read_csv(f"{data_folder}/transcriptomics.csv", index_col=0).T

    def run(self):
        self.compare_imputed_ccle()
        self.ccle_compare_by_genes()
        self.ccle_compare_with_vae()
        self.copy_number()

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

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_boxplot",
        )

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
                f"R={r:.2g}; Rho={s:.2g}; RMSE={rmse:.2f}, N={len(data):,}",
                fontsize=6,
                transform=ax.transAxes,
                ha="right",
            )
            ax.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        g = sns.FacetGrid(plot_df, col="impute")

        g.map_dataframe(sns.scatterplot, x="original", y="corr")
        g.map_dataframe(annotate)

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_facetgrid",
        )

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

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_by_gene_boxplot",
        )

    def ccle_compare_with_vae(self):
        features = list(set(self.df_vae).intersection(self.df_ccle))
        samples = list(set(self.df_vae.index).intersection(self.df_ccle.index))

        # scale
        df_vae_zscore = stats.zscore(self.df_vae, nan_policy="omit")
        df_ccle_zscore = stats.zscore(self.df_ccle, nan_policy="omit")

        # Correlation dataframe
        df_corrs = pd.DataFrame(
            [
                two_vars_correlation(
                    df_vae_zscore.loc[s, features],
                    df_ccle_zscore.loc[s, features],
                    method="pearson",
                    extra_fields=dict(
                        sample=s,
                        outofsample="Out-of-sample"
                        if s not in self.df_original.index
                        else "In-sample",
                    ),
                )
                for s in samples
            ]
        )

        ttest_stat = (
            stats.ttest_ind(
                df_corrs.query("outofsample == 'In-sample'")["corr"],
                df_corrs.query("outofsample == 'Out-of-sample'")["corr"],
                equal_var=False,
            ),
        )

        # t-test
        print(
            "T-test for correlation between in-sample and out-of-sample correlations:",
            ttest_stat,
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

        # change legend title
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=[
                f"{l} (N={len(df_corrs.query('outofsample == @l'))})" for l in labels
            ],
            title="",
            fontsize=6,
            loc="upper left",
            frameon=False,
        )

        g.set(
            title=f"Comparison VAE with CCLE (T-test p={ttest_stat[1]:.2e})",
            xlabel="Sample correlation (Pearson's r)",
            ylabel=f"Number of cell lines",
        )

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_with_vae_hist",
        )

    def copy_number(self):
        # Color palette
        palette = dict(
            zip(
                ["Deletion", "Loss", "Neutral", "Gain", "Amplification"],
                sns.color_palette("RdYlGn", as_cmap=False, n_colors=5).as_hex(),
            )
        )
        palette["Neutral"] = "#d9d9d9"

        # Copy number loss events
        loss_events = (self.df_cnv == "Deletion").sum().sort_values(ascending=False)
        loss_events = loss_events[loss_events.index.isin(self.df_vae.columns)]

        # Assemble dataframe
        loss_events_list = [
            ("CDKN2A", "CDKN2A"),
            ("SMAD4", "SMAD4"),
            ("CTNNB1", "CTNNB1"),
        ]

        for protein, cnv in loss_events_list:
            # protein, cnv = ("TOP2A", "TOP2A")
            df = (
                pd.concat(
                    [
                        self.df_original[[protein]].add_suffix("_orig"),
                        self.df_vae[[protein]].add_suffix("_vae"),
                        self.df_gexp[[protein]].add_suffix("_trans"),
                        self.df_cnv[[cnv]].add_suffix("_cnv"),
                    ],
                    axis=1,
                )
                .reindex(index=self.df_vae.index)
                .dropna(subset=[f"{cnv}_cnv"])
            )

            df["predicted"] = df[f"{protein}_orig"].isnull()
            df["predicted"].replace(
                {
                    True: f"Predicted (N={df['predicted'].sum()})",
                    False: f"Observed (N={(~df['predicted']).sum()})",
                },
                inplace=True,
            )

            # Plot
            g = GIPlot.gi_regression_marginal(
                x=f"{protein}_vae",
                y=f"{protein}_trans",
                z=f"{protein}_cnv",
                style="predicted",
                plot_df=df,
                discrete_pal=palette,
                hue_order=palette.keys(),
                legend_title=f"{protein}",
                scatter_kws=dict(edgecolor="w", lw=0.1, s=16),
            )

            g.ax_joint.set_xlabel(f"{protein} Proteomics (VAE)")
            g.ax_joint.set_ylabel(f"{protein} Transcriptomics (measured)")

            plt.gcf().set_size_inches(2, 2)

            PhenPred.save_figure(
                f"{plot_folder}/proteomics/{self.timestamp}_loss_event_{protein}_{cnv}_scatter",
            )
