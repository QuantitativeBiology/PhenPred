import sys

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
from sklearn.metrics import mean_squared_error
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import data_folder, plot_folder


_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class ProteomicsBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Original dataset
        self.df_original = pd.read_csv(f"{data_folder}/proteomics.csv", index_col=0).T

        # Fully imputed autoencoder dataset
        self.df_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_proteomics.csv.gz", index_col=0
        )

        # MOFA imputed dataset
        self.df_mofa = pd.read_csv(f"{data_folder}/proteomicsMOFA.csv", index_col=0).T

        # Independent proteomics dataset - CCLE
        self.df_ccle = pd.read_csv(f"{data_folder}/proteomics_ccle.csv", index_col=0).T

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

        print(
            f"[{_timestamp}] Samples = {len(self.samples)}, Features = {len(self.features)}"
        )

    def run(self):
        self.compare_imputed_ccle()

    def place_imputed_values_in_nans(self):
        df_original = self.df_original.copy().reindex(
            index=self.samples, columns=self.features
        )

        df_original_imp_mofa = df_original.copy().fillna(
            self.df_mofa.reindex(index=self.samples, columns=self.features)
        )

        df_original_imp_vae = df_original.copy().fillna(
            self.df_vae.reindex(index=self.samples, columns=self.features)
        )

        df_original_imp_mean = df_original.copy().fillna(df_original.mean())

        return dict(
            original=df_original,
            mofa=df_original_imp_mofa,
            mean=df_original_imp_mean,
            vae=df_original_imp_vae,
            vae_full=self.df_vae.reindex(index=self.samples, columns=self.features),
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
            value_vars=["mean", "mofa", "vae", "vae_full"],
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
