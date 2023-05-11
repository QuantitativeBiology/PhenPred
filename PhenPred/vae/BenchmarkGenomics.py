from turtle import color
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime
from scipy.special import stdtr
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler


class GenomicsBenchmark:
    def __init__(self, timestamp):
        self.timestamp = timestamp

        # Original dataset
        self.genomics = pd.read_csv(f"{data_folder}/genomics.csv", index_col=0).T

        # Original drug response values
        self.drespo = pd.read_csv(f"{data_folder}/drugresponse.csv", index_col=0).T

        # Fully generated drug response autoencoder dataset
        self.drespo_vae = pd.read_csv(
            f"{plot_folder}/files/{timestamp}_imputed_drugresponse.csv.gz", index_col=0
        )

        # Sample sheet
        self.ss = pd.read_csv(f"{data_folder}/cmp_model_list_20230307.csv", index_col=0)

        # Samples overlap
        self.samples = list(
            set(self.genomics.index)
            .intersection(set(self.drespo.index))
            .intersection(set(self.drespo_vae.index))
            .intersection(set(self.ss.index))
        )

        # Covariates
        self.covs = pd.concat(
            [
                self.ss["growth_properties"].str.get_dummies(),
                self.ss["tissue"].str.get_dummies()["Haematopoietic and Lymphoid"],
            ],
            axis=1,
        ).reindex(index=self.samples)

        # Impute missing values
        self.drespo_imputed = self.drespo.reindex(
            index=self.samples, columns=self.drespo_vae.columns, copy=True
        )
        self.drespo_imputed = self.drespo_imputed.fillna(
            self.drespo_vae.reindex(
                index=self.samples, columns=self.drespo_imputed.columns
            )
        )

        # Subset original
        self.drespo = self.drespo.reindex(
            index=self.samples, columns=self.drespo_vae.columns
        )

        # Filter low occurance genomic features
        self.genomics = self.genomics.loc[:, self.genomics.count() > 3]

    def run(self):
        lm_res = self.associations()
        lm_res.to_csv(f"{plot_folder}/genomics/{self.timestamp}_lm_res.csv")

    def associations(self):
        lm_drug_orig = pd.concat(
            [
                LModel(
                    Y=self.drespo.loc[self.samples, [d]].dropna(),
                    X=self.genomics.loc[self.samples],
                    M=self.covs.loc[self.samples],
                ).fit_matrix()
                for d in self.drespo_imputed.columns
            ]
        )
        lm_drug_orig = LModel.multipletests(lm_drug_orig).sort_values("fdr")
        lm_drug_orig = lm_drug_orig.set_index(["y_id", "x_id"])

        lm_drug_vae = LModel(
            Y=self.drespo_imputed.loc[self.samples],
            X=self.genomics.loc[self.samples],
            M=self.covs.loc[self.samples],
        ).fit_matrix()
        lm_drug_vae = LModel.multipletests(lm_drug_vae).sort_values("fdr")
        lm_drug_vae = lm_drug_vae.set_index(["y_id", "x_id"])

        lm_drug = pd.concat(
            [
                lm_drug_orig.add_suffix("_orig"),
                lm_drug_vae.add_suffix("_vae"),
            ],
            axis=1,
        ).dropna()

        return lm_drug

    def plot_associations(self, lm_res):
        plot_df = lm_res.query("fdr_orig < 0.05 | fdr_var < 0.05").copy()
        plot_df = plot_df.sort_values("pval_orig").reset_index()

        _, ax = plt.subplots(1, 1, figsize=(3, 1.5), dpi=600)

        sns.scatterplot(
            x=plot_df.index,
            y=-np.log10(plot_df["pval_orig"]),
            color="#E1E1E1",
            lw=0,
            s=1,
            zorder=1,
            rasterized=True,
            ax=ax,
        )

        ax.set(
            title=f"Pharmacogenomics associations (N={plot_df.shape[0]:,})",
            xlabel="Ranked drug ~ genomic feature associations",
            ylabel="Log-ratio p-value (log10)",
        )

        plt.savefig(
            f"{plot_folder}/genomics/{self.timestamp}_lm_assoc_pval_scatter.pdf",
            bbox_inches="tight",
        )
        plt.close()

        d_drespo_original = self.drespo[["1373;Dabrafenib;GDSC2"]]
        d_drespo_original_scl = pd.DataFrame(
            StandardScaler().fit_transform(d_drespo_original),
            index=d_drespo_original.index,
        )

        d_drespo_vae = self.drespo_vae[["1373;Dabrafenib;GDSC2"]]
        d_drespo_vae_scl = pd.DataFrame(
            StandardScaler().fit_transform(d_drespo_vae), index=d_drespo_vae.index
        )

        plot_df = pd.concat(
            [
                d_drespo_original_scl[0].rename("original"),
                d_drespo_vae_scl[0].rename("vae"),
                self.genomics["BRAF_mut"],
            ],
            axis=1,
        ).dropna()

        sns.scatterplot(
            x=plot_df["original"], y=plot_df["vae"], hue=plot_df["BRAF_mut"]
        )
        plt.show()
