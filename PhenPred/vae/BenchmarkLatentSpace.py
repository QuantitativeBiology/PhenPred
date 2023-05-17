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
    def __init__(self, timestamp, data):
        self.data = data
        self.timestamp = timestamp

        self.latent_space = pd.read_csv(
            f"{plot_folder}/files/{self.timestamp}_latent_joint.csv.gz", index_col=0
        )

        self.ss = data.samplesheet.copy()

        self.ss_ccell = pd.read_csv(
            f"{data_folder}/samplesheet_cancercell.csv", index_col=0
        )

        covariates_prot = self.data.dfs["proteomics"][["CDH1", "VIM"]].add_suffix(
            "_prot"
        )
        covariates_trans = self.data.dfs["transcriptomics"][["CDH1", "VIM"]].add_suffix(
            "_gexp"
        )

        self.covariates = pd.concat(
            [
                self.ss_ccell["CopyNumberAttenuation"],
                self.ss_ccell["GeneExpressionCorrelation"],
                self.ss_ccell["CopyNumberInstability"],
                self.ss_ccell[["ploidy", "mutational_burden", "growth", "size"]],
                self.ss_ccell["replicates_correlation"].rename("RepsCorrelation"),
                covariates_prot,
                covariates_trans,
                pd.get_dummies(self.ss_ccell["media"]),
                pd.get_dummies(self.ss["growth_properties_sanger"]).add_prefix(
                    "sanger_"
                ),
                pd.get_dummies(self.ss["growth_properties_broad"]).add_prefix("broad_"),
                self.data.dfs["proteomics"].mean(1).rename("MeanProteomics"),
                self.data.dfs["methylation"].mean(1).rename("MeanMethylation"),
                self.data.dfs["drugresponse"].mean(1).rename("MeanDrugResponse"),
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

        latents_corr = pd.DataFrame(latents_corr).dropna()

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
            xticklabels=True,
            yticklabels=True,
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
