import os
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from math import sqrt
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder


class ProteomicsBenchmark:
    def __init__(self, timestamp, data, vae_imputed, mofa_imputed):
        self.timestamp = timestamp

        self.data = data
        self.vae_imputed = vae_imputed
        self.mofa_imputed = mofa_imputed

        # Samplesheet
        self.ss = pd.read_csv(f"{data_folder}/cmp_model_list_20230307.csv", index_col=0)

        # Proteomics datasets
        self.df_original = self.data.dfs["proteomics"]
        self.df_vae = self.vae_imputed["proteomics"]
        self.df_mofa = self.mofa_imputed["proteomics"]
        self.df_mean = self.df_original.fillna(self.df_original.mean())

        # Other relevant datasets
        if "copynumber" in self.data.dfs:
            self.df_cnv = self.data.dfs["copynumber"]
        else:
            self.df_cnv = self.data.cnv

        self.df_gexp = self.data.dfs["transcriptomics"]

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
            .intersection(set(self.df_mofa.index))
            .intersection(set(self.df_ccle.index))
        )

        self.features = (
            set(self.df_original.columns)
            .intersection(set(self.df_vae.columns))
            .intersection(set(self.df_mofa.columns))
            .intersection(set(self.df_ccle.columns))
        )

        self.samples_without_prot = set(
            self.df_original.index[self.df_original.isnull().all(1)]
        )

        if not os.path.exists(f"{plot_folder}/proteomics"):
            os.makedirs(f"{plot_folder}/proteomics")

    def run(self):
        self.compare_imputed_ccle()
        self.ccle_compare_by_genes()
        self.ccle_compare_with_vae()

    def proteomics_datasets_dict(self, zscore=True, reindex=None):
        dfs = dict(
            original=self.df_original,
            vae_imputed=self.df_vae,
            mofa_imputed=self.df_mofa,
            mean=self.df_mean,
        )

        if reindex is not None:
            dfs = {
                n: df.reindex(index=reindex[0], columns=reindex[1])
                for n, df in dfs.items()
            }

        if zscore:
            dfs = {k: stats.zscore(df, nan_policy="omit") for k, df in dfs.items()}

        return dfs

    def compare_imputed_ccle(self):
        samples = self.samples.difference(self.samples_without_prot)

        df_imputed = self.proteomics_datasets_dict(reindex=(samples, self.features))

        df_ccle = self.df_ccle.reindex(index=samples, columns=self.features)
        df_ccle = stats.zscore(df_ccle, nan_policy="omit")

        # Correlation dataframe
        df_corrs = (
            pd.DataFrame(
                [
                    two_vars_correlation(
                        df_ccle.loc[s], df.loc[s], extra_fields=dict(sample=s, impute=k)
                    )
                    for s in samples
                    for k, df in df_imputed.items()
                ]
            )
            .dropna()
            .sort_values("pval")
        )
        df_corrs.to_csv(
            f"{plot_folder}/proteomics/{self.timestamp}_ccle_correlations.csv"
        )

        # Boxplot
        _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

        sns.boxplot(
            data=df_corrs,
            x="corr",
            y="impute",
            order=["original", "vae_imputed", "mofa_imputed", "mean"],
            color="#ababab",
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
            title=f"Sample correlation between imputed and CCLE (N={df_ccle.shape[0]:,})",
            xlabel="Pearson's r",
            ylabel=f"",
        )

        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.15))
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
                "vae_imputed",
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
                f"r={r:.2g}; rho={s:.2g}; RMSE={rmse:.2f}",
                fontsize=6,
                transform=ax.transAxes,
                ha="right",
            )
            ax.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        g = sns.FacetGrid(plot_df, col="impute", height=2, aspect=1, sharey=True)

        g.map_dataframe(
            sns.scatterplot,
            x="original",
            y="corr",
            color="#656565",
            s=5,
            alpha=0.8,
            linewidth=0.1,
        )
        g.map_dataframe(annotate)

        g.set_axis_labels("Original correlation (r)", "Imputed correlation (r)")

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_facetgrid",
        )

    def ccle_compare_by_genes(self):
        features = list(set(self.df_vae).intersection(self.df_ccle))

        samples = set(self.df_vae.index).intersection(self.df_ccle.index)
        samples = list(samples.difference(self.samples_without_prot))

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
            color="#ababab",
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
                        if s in self.samples_without_prot
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
            nan_policy="omit",
        )

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

        g.set(
            title=f"Comparison VAE with CCLE\n(Welch's t-test p={ttest_stat[1]:.2e})",
            xlabel="Sample correlation (Pearson's r)",
            ylabel=f"Number of cell lines",
        )

        PhenPred.save_figure(
            f"{plot_folder}/proteomics/{self.timestamp}_imputed_corr_with_vae_hist",
        )

    def copy_number(self, loss_events_list=None, proteomics_only=False):
        # Color palette
        palette = dict(
            zip(
                ["Deletion", "Loss", "Neutral", "Gain", "Amplification"],
                sns.color_palette("RdYlGn", as_cmap=False, n_colors=5).as_hex(),
            )
        )
        palette["Neutral"] = "#d9d9d9"

        # Copy number loss events
        loss_events = (self.df_cnv == -2).sum().sort_values(ascending=False)
        loss_events = loss_events[loss_events.index.isin(self.df_vae.columns)]

        # Copy number map
        cnv_map = {
            -2: "Deletion",
            -1: "Loss",
            0: "Neutral",
            1: "Gain",
            2: "Amplification",
        }

        # Assemble dataframe
        if loss_events_list is None:
            loss_events_list = [
                ("SMAD4", "SMAD4"),
                ("TP53", "TP53"),
                ("CDKN2A", "CDKN2A"),
                ("ZMYM2", "ZMYM2"),
                ("PCM1", "PCM1"),
                ("NDRG1", "NDRG1"),
                ("NPEPPS", "NPEPPS"),
                ("TOP2A", "TOP2A"),
                ("RAC1", "RAC1"),
            ]

        for protein, cnv in loss_events_list:
            # protein, cnv = ("ZMYM2", "ZMYM2")
            df = (
                pd.concat(
                    [
                        self.df_original[[protein]].add_suffix("_orig"),
                        self.df_vae[[protein]].add_suffix("_vae"),
                        self.df_gexp[[protein]].add_suffix("_trans"),
                        self.df_cnv[[cnv]].add_suffix("_cnv").replace(cnv_map),
                    ],
                    axis=1,
                )
                .reindex(index=self.df_vae.index)
                .dropna(subset=[f"{cnv}_cnv"])
            )

            df["predicted"] = df[f"{protein}_orig"].isnull()
            df["predicted"].replace(
                {
                    True: f"Reconstructed (N={df['predicted'].sum()})",
                    False: f"Observed (N={(~df['predicted']).sum()})",
                },
                inplace=True,
            )

            if proteomics_only:
                samples_with_proteomics = (
                    self.data.dfs["proteomics"].dropna(how="all").index.tolist()
                )
                df = df.loc[df.index.isin(samples_with_proteomics)]

            # Plot
            g = GIPlot.gi_regression_marginal(
                x=f"{protein}_vae",
                y=f"{protein}_trans",
                z=f"{protein}_cnv",
                style="predicted",
                plot_df=df.sort_values("predicted"),
                discrete_pal=palette,
                hue_order=list(palette.keys())[::-1],
                legend_title=f"{protein}",
                scatter_kws=dict(edgecolor="w", lw=0.1, s=8, alpha=0.7),
            )

            g.ax_joint.set_xlabel(f"{protein} Proteomics (MOVE)")
            g.ax_joint.set_ylabel(f"{protein} Transcriptomics (measured)")

            plt.gcf().set_size_inches(2, 2)

            PhenPred.save_figure(
                f"{plot_folder}/proteomics/{self.timestamp}_loss_event_{protein}_{cnv}_scatter",
            )
