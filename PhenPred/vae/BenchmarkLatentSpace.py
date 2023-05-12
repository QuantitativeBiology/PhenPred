import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from math import sqrt
from scipy.stats import skew
from datetime import datetime
from scipy.special import stdtr
from scipy.stats import pearsonr, spearmanr
from PhenPred.vae.PlotUtils import GIPlot
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.Utils import two_vars_correlation, LModel


class LatentSpaceBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        self.latent_space = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz", index_col=0
        )

        self.ss_cmp = pd.read_csv(
            f"{data_folder}/cmp_model_list_20230307.csv", index_col=0
        )

        self.ss_ccell = pd.read_csv(
            f"{data_folder}/samplesheet_cancercell.csv", index_col=0
        )

        self.prot = pd.read_csv(f"{data_folder}/proteomics.csv", index_col=0)
        self.gexp = pd.read_csv(f"{data_folder}/transcriptomics.csv", index_col=0)
        self.methy = pd.read_csv(f"{data_folder}/methylation.csv", index_col=0)
        self.drespo = pd.read_csv(f"{data_folder}/drugresponse.csv", index_col=0)

        self.covariates = pd.concat(
            [
                self.ss_ccell["CopyNumberAttenuation"],
                self.ss_ccell["GeneExpressionCorrelation"],
                self.ss_ccell["CopyNumberInstability"],
                self.prot.loc[["CDH1", "VIM"]].T.add_suffix("_prot"),
                self.gexp.loc[["CDH1", "VIM"]].T.add_suffix("_gexp"),
                pd.get_dummies(self.ss_ccell["media"]),
                pd.get_dummies(self.ss_ccell["growth_properties"]),
                pd.get_dummies(self.ss_ccell["growth_properties"]),
                self.ss_ccell[["ploidy", "mutational_burden", "growth", "size"]],
                self.ss_ccell["replicates_correlation"].rename("RepsCorrelation"),
                self.prot.mean().rename("MeanProteomics"),
                self.methy.mean().rename("MeanMethylation"),
                self.drespo.mean().rename("MeanDrugResponse"),
            ],
            axis=1,
        )

    def correlate_latents_with_covariates(self):
        latents_corr = {}

        for l in self.latent_space:
            latents_corr[l] = {}

            for c in self.covariates:
                fc_samples = list(
                    self.covariates.reindex(self.latent_space[l].index)[c]
                    .dropna()
                    .index
                )
                latents_corr[l][c] = two_vars_correlation(
                    self.latent_space[l][fc_samples], self.covariates[c][fc_samples]
                )["corr"]

        latents_corr = pd.DataFrame(latents_corr)

        return latents_corr

    def run(self):
        self.correlation_latents()

        latents_corr = self.correlate_latents_with_covariates()
        latents_corr.to_csv(
            f"{plot_folder}/latent/{self.timestamp}_latents_covariates_corr.csv",
        )

        self.covariates_latents(latents_corr)

    def correlation_latents(self):
        plot_df = self.latent_space.corr()

        g = sns.clustermap(
            plot_df,
            cmap="RdYlGn",
            center=0,
            xticklabels=False,
            yticklabels=False,
            linewidths=0.0,
            cbar_kws={"shrink": 0.5},
            figsize=(4, 4),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        plt.savefig(
            f"{plot_folder}/latent/{self.timestamp}_latents_corr_clustermap.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def covariates_latents(self, latents_corr):
        g = sns.clustermap(
            latents_corr,
            cmap="RdYlGn",
            center=0,
            linewidths=0.0,
            col_cluster=False,
            cbar_kws={"shrink": 0.5},
            figsize=(8, 3.5),
        )

        g.ax_cbar.set_ylabel("Pearson correlation")

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")

        plt.savefig(
            f"{plot_folder}/latent/{self.timestamp}_latents_covariates_clustermap.pdf",
            bbox_inches="tight",
        )
        plt.close()
