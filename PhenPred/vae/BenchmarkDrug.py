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
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder

_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class DrugResponseBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Original dataset
        self.df_original = pd.read_csv(
            f"{data_folder}/drugresponse.csv", index_col=0
        ).T

        # New drug response values
        self.df_original_new = pd.read_csv(
            f"{data_folder}/drugresponse_24Jul22.csv", index_col=0
        ).T

        # Fully imputed autoencoder dataset
        self.df_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_drugresponse.csv.gz", index_col=0
        )

        # MOFA imputed dataset
        self.df_mofa = pd.read_csv(
            f"{data_folder}/drugresponseMOFA.csv", index_col=0
        ).T

        # Mean imputed dataset
        self.df_mean = self.df_original.fillna(self.df_original.mean())

        # Intersection samples
        self.samples = (
            set(self.df_original.index)
            .intersection(set(self.df_original_new.index))
            .intersection(set(self.df_vae.index))
            .intersection(set(self.df_mofa.index))
            .intersection(set(self.df_mean.index))
        )
        self.samples = list(self.samples)

        # Intersection features
        self.features = (
            set(self.df_original.columns)
            .intersection(set(self.df_original_new.columns))
            .intersection(set(self.df_vae.columns))
            .intersection(set(self.df_mofa.columns))
            .intersection(set(self.df_mean.columns))
        )
        self.features = list(self.features)

        print(
            f"[{_timestamp}] Samples = {len(self.samples)}, Features = {len(self.features)}"
        )

    def run(self):
        self.correlation_new_values()

    def correlation_new_values(self):
        # Define subset of newly measured drug - cell line pairs
        df_new_values = (
            self.df_original_new.loc[self.samples, self.features].unstack().dropna()
        )
        df_new_values = df_new_values.loc[
            self.df_original.loc[self.samples, self.features]
            .unstack()
            .loc[df_new_values.index]
            .isna()
        ]
        df_new_values = pd.concat(
            [
                df_new_values.rename("original"),
                self.df_mofa.unstack().loc[df_new_values.index].rename("MOFA"),
                self.df_mean.unstack().loc[df_new_values.index].rename("mean"),
                self.df_vae.unstack().loc[df_new_values.index].rename("VAE"),
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

            rmse = sqrt(mean_squared_error(plot_df["original"], plot_df[y_var]))
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
                    "RMSE": rmse,
                    "Spearman's rho": s,
                    "Pearson's r": r,
                    "method": y_var,
                }
            )
            annot_text = f"R={r:.2g}; Rho={s:.2g}; RMSE={rmse:.2f}"
            ax.text(
                0.95, 0.05, annot_text, fontsize=6, transform=ax.transAxes, ha="right"
            )

            plt.savefig(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_scatter_{y_var}.png",
                bbox_inches="tight",
            )
            plt.close()

        # Bar plot
        plot_df = pd.DataFrame(corr_dict)

        for y_var in ["RMSE", "Spearman's rho", "Pearson's r"]:
            _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

            sns.barplot(
                data=plot_df,
                x=y_var,
                y="method",
                color="#656565",
                ax=ax,
            )

            ax.set(
                title=f"Drug response prediction (N={df_new_values.shape[0]:,})",
                xlabel=y_var,
                ylabel="Imputation method",
            )

            plt.savefig(
                f"{plot_folder}/drugresponse/{self.timestamp}_imputed_{y_var}_barplot.pdf",
                bbox_inches="tight",
            )
            plt.close()
