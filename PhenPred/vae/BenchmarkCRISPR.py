import os
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
from PhenPred.vae.PlotUtils import GIPlot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from PhenPred.vae import data_folder, plot_folder
from statsmodels.stats.multitest import multipletests
from PhenPred.vae.Utils import two_vars_correlation, LModel
from sklearn.decomposition import PCA


class CRISPRBenchmark:
    def __init__(
        self, timestamp, data, vae_imputed, mofa_imputed, min_obs=15, skew_threshold=-2
    ):
        self.timestamp = timestamp

        self.min_obs = min_obs
        self.skew_threshold = skew_threshold

        self.data = data
        self.vae_imputed = vae_imputed
        self.mofa_imputed = mofa_imputed

        # CRISPR-Cas9 datasets
        self.df_original = data.dfs["crisprcas9"].dropna(how="all").dropna(axis=1)
        self.df_vae = self.vae_imputed["crisprcas9"]

        self.skew_df = pd.concat(
            [
                self.df_original.apply(skew).astype(float).rename("skew_orig"),
                self.df_vae.apply(skew).astype(float).rename("skew_mosa"),
            ],
            axis=1,
        )

        # Genomics
        self.mutations = self.data.mutations.add_suffix("_mut")
        self.deletions = (
            (self.data.dfs["copynumber"] == -2).astype(int).add_suffix("_del")
        )
        self.amplitifications = (
            (self.data.dfs["copynumber"] == 2).astype(int).add_suffix("_amp")
        )
        self.fusions = self.data.fusions.add_suffix("_fusion")
        self.msi = (
            self.data.ss_cmp["msi_status"]
            .replace({"MSS": "0", "MSI": "1"})
            .astype(float)
            .rename("MSI")
        )

        self.genomics = pd.concat(
            [
                self.mutations,
                self.deletions,
                self.amplitifications,
                self.fusions,
                self.msi,
            ],
            axis=1,
        )
        self.genomics = self.genomics.loc[:, self.genomics.sum() >= self.min_obs]

        # Transcriptomics
        self.transcriptomics = self.data.dfs["transcriptomics"].dropna(how="all")
        # self.transcriptomics = self.vae_imputed["transcriptomics"]

        # Copy number
        self.copynumber = self.data.dfs["copynumber"].dropna(how="all")

        # Sample sheet
        self.ss = data.samplesheet.copy()

        if not os.path.exists(f"{plot_folder}/crispr"):
            os.makedirs(f"{plot_folder}/crispr")

    def run(self, run_associations=True):
        if run_associations:
            if not os.path.exists(
                f"{plot_folder}/crispr/{self.timestamp}_genomics_crisprcas9.csv.gz"
            ):
                self.lm_genomics = self.genomic_associations()
                self.lm_genomics.to_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_genomics_crisprcas9.csv.gz",
                    compression="gzip",
                    index=False,
                )
            else:
                self.lm_genomics = pd.read_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_genomics_crisprcas9.csv.gz"
                )

            if not os.path.exists(
                f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9.csv.gz"
            ):
                self.lm_transcriptomics = self.transcriptomics_associations()
                self.lm_transcriptomics.to_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9.csv.gz",
                    compression="gzip",
                    index=False,
                )
            else:
                self.lm_transcriptomics = pd.read_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9.csv.gz"
                )

            if not os.path.exists(
                f"{plot_folder}/crispr/{self.timestamp}_copynumber_crisprcas9.csv.gz"
            ):
                self.lm_copynumber = self.copynumber_associations()
                self.lm_copynumber.to_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_copynumber_crisprcas9.csv.gz",
                    compression="gzip",
                    index=False,
                )
            else:
                self.lm_copynumber = pd.read_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_copynumber_crisprcas9.csv.gz"
                )

            if not os.path.exists(
                f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9_tissue.csv.gz"
            ):
                self.lm_transcriptomics_tissue = (
                    self.transcriptomics_associations_tissue_level()
                )
                self.lm_transcriptomics_tissue.to_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9_tissue.csv.gz",
                    compression="gzip",
                    index=False,
                )
            else:
                self.lm_transcriptomics_tissue = pd.read_csv(
                    f"{plot_folder}/crispr/{self.timestamp}_transcriptomics_crisprcas9_tissue.csv.gz"
                )

        self.associations_scatter_pvals(self.lm_genomics)
        self.gene_skew_correlation()

    def gene_skew_correlation(self):
        original_skew = self.df_original.apply(skew).astype(float).rename("orig")
        original_ess = (self.df_original < -0.5).sum().rename("orig_ess")

        # index not in self.df_original
        vae_only_skew = (
            self.df_vae.loc[self.df_vae.index.difference(self.df_original.index)]
            .apply(skew)
            .astype(float)
            .rename("vae")
        )

        plot_df = (
            pd.concat([original_skew, vae_only_skew, original_ess], axis=1)
            .dropna()
            .sort_values("orig_ess")
        )

        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        sns.scatterplot(
            data=plot_df,
            x="orig",
            y="vae",
            lw=0,
            hue="orig_ess",
            size="orig_ess",
            sizes=(3, 15),
            alpha=0.75,
            ax=ax,
        )
        sns.regplot(
            data=plot_df,
            x="orig",
            y="vae",
            scatter=False,
            color="#fc8d62",
            truncate=True,
            line_kws={"lw": 1},
            ax=ax,
        )
        ax.set(
            title=f"CRISPR-Cas9 skew (N={plot_df.shape[0]:,})",
            xlabel="Skew original",
            ylabel=f"Skew MOSA",
        )

        # same axes limits and step sizes
        ax_min, ax_max = (
            min(plot_df["orig"].min(), plot_df["vae"].min()),
            max(plot_df["orig"].max(), plot_df["vae"].max()),
        )

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())

        rmse = sqrt(mean_squared_error(plot_df["orig"], plot_df["vae"]))
        s, _ = stats.spearmanr(
            plot_df["orig"],
            plot_df["vae"],
        )
        r, _ = stats.pearsonr(
            plot_df["orig"],
            plot_df["vae"],
        )
        annot_text = f"R={r:.2g}; RMSE={rmse:.2f}"
        ax.text(0.05, 0.95, annot_text, fontsize=6, transform=ax.transAxes, ha="left")

        ax.axline((1, 1), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        PhenPred.save_figure(
            f"{plot_folder}/crispr/{self.timestamp}_gene_skew_corrplot"
        )

    def genomic_associations(self):
        # Covariates
        covs = pd.concat(
            [
                self.ss["growth_properties_sanger"]
                .str.get_dummies()
                .add_prefix("sanger_"),
                self.ss["growth_properties_broad"]
                .str.get_dummies()
                .add_prefix("broad_"),
                self.ss["tissue"].str.get_dummies()[["Haematopoietic and Lymphoid"]],
            ],
            axis=1,
        )

        y_features = pd.concat(
            [
                self.df_vae.apply(skew).astype(float).rename("vae"),
                self.df_original.apply(skew).astype(float).rename("orig"),
            ],
            axis=1,
        )
        y_features = list(
            y_features.loc[(y_features < self.skew_threshold).any(axis=1)].index
        )

        # Genomics ~ CRISPR MOSA
        samples = list(
            set(self.df_vae.dropna().index)
            .intersection(self.data.mutations.index)
            .intersection(self.data.dfs["copynumber"].index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.genomics.loc[samples].replace(np.nan, 0).astype(int)

        lm_genomics_vae = LModel(
            Y=self.df_vae.loc[samples, y_features],
            X=x.loc[samples],
            M=covs.loc[samples],
        ).fit_matrix()

        lm_genomics_vae = LModel.multipletests(
            lm_genomics_vae, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_genomics_vae = lm_genomics_vae.set_index(["y_id", "x_id"])

        # Genomics ~ CRISPR original
        samples = list(
            set(self.df_original.dropna().index)
            .intersection(self.data.mutations.index)
            .intersection(self.data.dfs["copynumber"].index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.genomics.loc[samples].replace(np.nan, 0).astype(int)

        lm_genomics_orig = LModel(
            Y=self.df_original.loc[samples],
            X=x.loc[samples],
            M=covs.loc[samples],
        ).fit_matrix()

        lm_genomics_orig = LModel.multipletests(
            lm_genomics_orig, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_genomics_orig = lm_genomics_orig.set_index(["y_id", "x_id"])

        # Concatenate
        lm_genomics = (
            pd.concat(
                [
                    lm_genomics_orig.add_suffix("_orig"),
                    lm_genomics_vae.add_suffix("_vae"),
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )

        return lm_genomics

    def transcriptomics_associations(self):
        # Covariates
        covs = pd.concat(
            [
                self.ss["growth_properties_sanger"]
                .str.get_dummies()
                .add_prefix("sanger_"),
                self.ss["growth_properties_broad"]
                .str.get_dummies()
                .add_prefix("broad_"),
                pd.get_dummies(
                    self.ss["tissue"].apply(
                        lambda x: (
                            x
                            if x in ["Haematopoietic and Lymphoid", "Lung"]
                            else "Other"
                        )
                    )
                ),
            ],
            axis=1,
        )
        cas9_measured_samples = self.df_original.index
        gexp_measured_samples = self.transcriptomics.index
        covs["cas9"] = covs.index.isin(
            list(set(cas9_measured_samples) - set(gexp_measured_samples))
        ).astype(int)
        covs["gexp"] = covs.index.isin(
            list(set(gexp_measured_samples) - set(cas9_measured_samples))
        ).astype(int)
        covs["cas9_gexp"] = covs.index.isin(
            list(set(cas9_measured_samples).intersection(gexp_measured_samples))
        ).astype(int)

        covs = covs.loc[:, covs.std() > 0]

        y_features = pd.concat(
            [
                self.df_vae.apply(skew).astype(float).rename("vae"),
                self.df_original.apply(skew).astype(float).rename("orig"),
            ],
            axis=1,
        )
        y_features = list(
            y_features.loc[(y_features < self.skew_threshold).any(axis=1)].index
        )

        # Transcriptomics ~ CRISPR MOSA
        samples = list(
            set(self.df_vae.dropna().index)
            .intersection(self.vae_imputed["transcriptomics"].index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.vae_imputed["transcriptomics"].loc[samples]
        x_features = x.var().sort_values(ascending=False).index[:5000]

        # select top 5000 variable genes from x
        x = x.loc[:, x_features]

        cov_vae = covs.loc[samples].copy()
        # add first 5 principal components from gexp data as covariates
        pca = PCA(n_components=5).fit(x)
        cov_vae = pd.concat(
            [
                cov_vae,
                pd.DataFrame(
                    pca.transform(x),
                    index=samples,
                    columns=[f"PC{i}" for i in range(1, 6)],
                ),
            ],
            axis=1,
        )

        lm_transcriptomics_vae = LModel(
            Y=self.df_vae.loc[samples, y_features],
            X=x.loc[samples],
            M=cov_vae.loc[samples],
        ).fit_matrix()

        lm_transcriptomics_vae = LModel.multipletests(
            lm_transcriptomics_vae, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_transcriptomics_vae = lm_transcriptomics_vae.set_index(["y_id", "x_id"])

        # Transcriptomics ~ CRISPR original
        samples = list(
            set(self.df_original.dropna().index)
            .intersection(self.transcriptomics.index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.transcriptomics.loc[samples]
        x = x.loc[:, x_features]

        cov_orig = covs.loc[samples].copy()
        # add first 5 principal components from gexp data as covariates
        pca = PCA(n_components=5).fit(x)
        cov_orig = pd.concat(
            [
                cov_orig,
                pd.DataFrame(
                    pca.transform(x),
                    index=samples,
                    columns=[f"PC{i}" for i in range(1, 6)],
                ),
            ],
            axis=1,
        )

        lm_transcriptomics_orig = LModel(
            Y=self.df_original.loc[samples],
            X=x.loc[samples],
            M=cov_orig.loc[samples],
        ).fit_matrix()

        lm_transcriptomics_orig = LModel.multipletests(
            lm_transcriptomics_orig, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_transcriptomics_orig = lm_transcriptomics_orig.set_index(["y_id", "x_id"])

        # Concatenate
        lm_transcriptomics = (
            pd.concat(
                [
                    lm_transcriptomics_orig.add_suffix("_orig"),
                    lm_transcriptomics_vae.add_suffix("_vae"),
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )

        return lm_transcriptomics

    def transcriptomics_associations_tissue_level(self):
        """Run tissue-specific transcriptomic associations with CRISPR-Cas9 data"""

        # Base covariates excluding tissue
        covs = pd.concat(
            [
                self.ss["growth_properties_sanger"]
                .str.get_dummies()
                .add_prefix("sanger_"),
                self.ss["growth_properties_broad"]
                .str.get_dummies()
                .add_prefix("broad_"),
            ],
            axis=1,
        )

        # Add CRISPR and expression measurement indicators
        cas9_measured_samples = self.df_original.index
        gexp_measured_samples = self.transcriptomics.index

        covs["cas9"] = covs.index.isin(
            list(set(cas9_measured_samples) - set(gexp_measured_samples))
        ).astype(int)
        covs["gexp"] = covs.index.isin(
            list(set(gexp_measured_samples) - set(cas9_measured_samples))
        ).astype(int)
        covs["cas9_gexp"] = covs.index.isin(
            list(set(cas9_measured_samples).intersection(gexp_measured_samples))
        ).astype(int)

        # Filter covariates with no variation
        covs = covs.loc[:, covs.std() > 0]

        # Get genes with significant skew in either original or VAE data
        y_features = pd.concat(
            [
                self.df_vae.apply(skew).astype(float).rename("vae"),
                self.df_original.apply(skew).astype(float).rename("orig"),
            ],
            axis=1,
        )
        y_features = list(
            y_features.loc[(y_features < self.skew_threshold).any(axis=1)].index
        )

        # Get unique tissues with minimum sample size
        min_samples_per_tissue = 30
        tissue_counts = self.ss["tissue"].value_counts()
        valid_tissues = tissue_counts[
            tissue_counts >= min_samples_per_tissue
        ].index.tolist()

        # Initialize results storage
        all_tissue_results = []

        # Run analysis for each tissue type
        for tissue in valid_tissues:
            print(f"\nProcessing tissue: {tissue}")

            # Get tissue-specific samples
            tissue_samples = self.ss[self.ss["tissue"] == tissue].index

            # MOSA analysis
            vae_samples = list(
                set(self.df_vae.dropna().index)
                .intersection(self.vae_imputed["transcriptomics"].index)
                .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
                .intersection(tissue_samples)
            )

            if len(vae_samples) < min_samples_per_tissue:
                print(
                    f"Skipping {tissue} - insufficient VAE samples: {len(vae_samples)}"
                )
                continue

            # Get top 500 variable genes for this tissue
            x_tissue = self.vae_imputed["transcriptomics"].loc[vae_samples]
            x_features = x_tissue.var().sort_values(ascending=False).index[:500]
            x = x_tissue.loc[:, x_features]

            # Add PCA components as covariates
            cov_vae = covs.loc[vae_samples].copy()
            pca = PCA(n_components=5).fit(x)
            cov_vae = pd.concat(
                [
                    cov_vae,
                    pd.DataFrame(
                        pca.transform(x),
                        index=vae_samples,
                        columns=[f"PC{i}" for i in range(1, 6)],
                    ),
                ],
                axis=1,
            )

            # Fit MOSA model
            lm_transcriptomics_vae = LModel(
                Y=self.df_vae.loc[vae_samples, y_features],
                X=x.loc[vae_samples],
                M=cov_vae.loc[vae_samples],
            ).fit_matrix()

            lm_transcriptomics_vae = LModel.multipletests(
                lm_transcriptomics_vae, idx_cols=["x_id"]
            ).sort_values("fdr")
            lm_transcriptomics_vae = lm_transcriptomics_vae.set_index(["y_id", "x_id"])

            # Original data analysis
            orig_samples = list(
                set(self.df_original.dropna().index)
                .intersection(self.transcriptomics.index)
                .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
                .intersection(tissue_samples)
            )

            # Use same gene set as VAE analysis
            x = self.transcriptomics.loc[orig_samples, x_features]

            # Add PCA components as covariates
            cov_orig = covs.loc[orig_samples].copy()
            pca = PCA(n_components=5).fit(x)
            cov_orig = pd.concat(
                [
                    cov_orig,
                    pd.DataFrame(
                        pca.transform(x),
                        index=orig_samples,
                        columns=[f"PC{i}" for i in range(1, 6)],
                    ),
                ],
                axis=1,
            )

            # Fit original data model
            lm_transcriptomics_orig = LModel(
                Y=self.df_original.loc[orig_samples],
                X=x.loc[orig_samples],
                M=cov_orig.loc[orig_samples],
            ).fit_matrix()

            lm_transcriptomics_orig = LModel.multipletests(
                lm_transcriptomics_orig, idx_cols=["x_id"]
            ).sort_values("fdr")
            lm_transcriptomics_orig = lm_transcriptomics_orig.set_index(
                ["y_id", "x_id"]
            )

            # Combine results
            tissue_results = (
                pd.concat(
                    [
                        lm_transcriptomics_orig.add_suffix("_orig"),
                        lm_transcriptomics_vae.add_suffix("_vae"),
                    ],
                    axis=1,
                )
                .dropna()
                .reset_index()
            )

            # Add tissue information
            tissue_results["tissue"] = tissue
            tissue_results["n_samples_orig"] = len(orig_samples)
            tissue_results["n_samples_vae"] = len(vae_samples)

            all_tissue_results.append(tissue_results)

        # Combine all tissue results
        final_results = pd.concat(all_tissue_results, axis=0)

        return final_results

    def copynumber_associations(self):
        # Covariates
        covs = pd.concat(
            [
                self.ss["growth_properties_sanger"]
                .str.get_dummies()
                .add_prefix("sanger_"),
                self.ss["growth_properties_broad"]
                .str.get_dummies()
                .add_prefix("broad_"),
                pd.get_dummies(
                    self.ss["tissue"].apply(
                        lambda x: (
                            x
                            if x in ["Haematopoietic and Lymphoid", "Lung"]
                            else "Other"
                        )
                    )
                ),
            ],
            axis=1,
        )
        cas9_measured_samples = self.df_original.index
        copynumber_measured_samples = self.copynumber.index
        covs["cas9"] = covs.index.isin(
            list(set(cas9_measured_samples) - set(copynumber_measured_samples))
        ).astype(int)
        covs["copynumber"] = covs.index.isin(
            list(set(copynumber_measured_samples) - set(cas9_measured_samples))
        ).astype(int)
        covs["cas9_copynumber"] = covs.index.isin(
            list(set(cas9_measured_samples).intersection(copynumber_measured_samples))
        ).astype(int)

        covs = covs.loc[:, covs.std() > 0]

        y_features = pd.concat(
            [
                self.df_vae.apply(skew).astype(float).rename("vae"),
                self.df_original.apply(skew).astype(float).rename("orig"),
            ],
            axis=1,
        )
        y_features = list(
            y_features.loc[(y_features < self.skew_threshold).any(axis=1)].index
        )

        # Copynumber ~ CRISPR MOSA
        samples = list(
            set(self.df_vae.dropna().index)
            .intersection(self.vae_imputed["copynumber"].index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.vae_imputed["copynumber"].loc[samples]

        cov_vae = covs.loc[samples].copy()
        # add first 5 principal components from gexp data as covariates
        pca = PCA(n_components=5).fit(x)
        cov_vae = pd.concat(
            [
                cov_vae,
                pd.DataFrame(
                    pca.transform(x),
                    index=samples,
                    columns=[f"PC{i}" for i in range(1, 6)],
                ),
            ],
            axis=1,
        )

        lm_copynumber_vae = LModel(
            Y=self.df_vae.loc[samples, y_features],
            X=x.loc[samples],
            M=cov_vae.loc[samples],
        ).fit_matrix()

        lm_copynumber_vae = LModel.multipletests(
            lm_copynumber_vae, idx_cols=["x_id"]
        ).sort_values("fdr")
        lm_copynumber_vae = lm_copynumber_vae.set_index(["y_id", "x_id"])

        # Copynumber ~ CRISPR original
        samples = list(
            set(self.df_original.dropna().index)
            .intersection(self.copynumber.index)
            .intersection(covs.reindex(index=self.df_vae.index).dropna().index)
        )

        x = self.copynumber.loc[samples].fillna(0)

        cov_orig = covs.loc[samples].copy()
        # add first 5 principal components from gexp data as covariates
        pca = PCA(n_components=5).fit(x)
        cov_orig = pd.concat(
            [
                cov_orig,
                pd.DataFrame(
                    pca.transform(x),
                    index=samples,
                    columns=[f"PC{i}" for i in range(1, 6)],
                ),
            ],
            axis=1,
        )

        lm_copynumber_orig = LModel(
            Y=self.df_original.loc[samples],
            X=x.loc[samples],
            M=cov_orig.loc[samples],
        ).fit_matrix()

        lm_copynumber_orig = LModel.multipletests(
            lm_copynumber_orig, idx_cols=["x_id"]
        ).sort_values("fdr")

        lm_copynumber_orig = lm_copynumber_orig.set_index(["y_id", "x_id"])

        # Concatenate
        lm_copynumber = (
            pd.concat(
                [
                    lm_copynumber_orig.add_suffix("_orig"),
                    lm_copynumber_vae.add_suffix("_vae"),
                ],
                axis=1,
            )
            .dropna()
            .reset_index()
        )

        return lm_copynumber

    def associations_scatter_pvals(self, lm_genomics):
        plot_df = lm_genomics.query("fdr_orig < 0.05 | fdr_vae < 0.05").copy()

        _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

        sns.scatterplot(
            x=-np.log10(plot_df["pval_orig"]),
            y=-np.log10(plot_df["pval_vae"]),
            color="black",
            alpha=0.5,
            linewidth=0,
            s=5,
            zorder=1,
            ax=ax,
        )
        # same axes limits and step sizes
        ax_min, ax_max = (
            min(
                (-np.log10(plot_df["pval_orig"])).min(),
                (-np.log10(plot_df["pval_vae"])).min(),
            ),
            max(
                (-np.log10(plot_df["pval_orig"])).max(),
                (-np.log10(plot_df["pval_vae"])).max(),
            ),
        )

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())

        ax.axline((0, 0), slope=1, color="black", lw=0.5, ls="-", zorder=-1)

        y_fdr = -np.log10(plot_df.query("fdr_vae < 0.01")["pval_vae"].max())
        ax.axhline(y_fdr, lw=0.5, ls="--", color="black", zorder=-1)
        ax.text(
            0.99,
            0.05,
            "FDR 1%",
            fontsize=4,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        x_fdr = -np.log10(plot_df.query("fdr_orig < 0.01")["pval_orig"].max())
        ax.axvline(x_fdr, lw=0.5, ls="--", color="black", zorder=-1)
        ax.text(
            0.01,
            0.99,
            "FDR 1%",
            fontsize=4,
            transform=ax.transAxes,
            ha="left",
            va="top",
            rotation=90,
        )

        ax.set(
            title=f"CRISPR-Cas9 ~ Genomics associations\n(N={plot_df.shape[0]:,})",
            xlabel="Original log-ratio p-value (-log10)",
            ylabel="MOSA log-ratio p-value (-log10)",
        )

        PhenPred.save_figure(
            f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_pval_scatter",
        )

    def plot_associations(self, associations=None):
        if associations is None:
            associations = [
                ("BRAF", "MAPK1", "BRAF_mut"),
            ]

        for y_id, x_id, z_id in associations:
            # y_id, x_id, z_id = ("BRAF", "MAPK1", "BRAF_mut")

            plot_df = pd.concat(
                [
                    self.df_original[[y_id, x_id]].add_suffix(f"_orig"),
                    self.df_vae[[y_id, x_id]].add_suffix(f"_vae"),
                    self.genomics[z_id],
                    self.ss["tissue"],
                ],
                axis=1,
            )
            plot_df = plot_df.dropna(subset=[f"{x_id}_vae", f"{y_id}_vae", z_id])

            plot_df.replace({z_id: {0: "WT", 1: z_id}}, inplace=True)
            plot_df["predicted"] = (
                plot_df[[f"{y_id}_orig", f"{x_id}_orig"]].isnull().any(axis=1)
            )

            plot_df.replace(
                {
                    "predicted": {
                        True: f"Reconstructed (N={plot_df['predicted'].sum()})",
                        False: f"Measured (N={(~plot_df['predicted']).sum()})",
                    }
                },
                inplace=True,
            )

            pal, pal_order = {
                z_id: "#fc8d62",
                "WT": "#e1e1e1",
                0: "#e1e1e1",
            }, ["WT", z_id]

            # Predicted
            g = GIPlot.gi_regression_marginal(
                x=f"{x_id}_vae",
                y=f"{y_id}_vae",
                z=z_id,
                style="predicted",
                plot_df=plot_df,
                discrete_pal=pal,
                hue_order=pal_order,
                legend_title=f"{z_id}",
                scatter_kws=dict(edgecolor="w", linewidths=0.1, s=10, alpha=0.75),
            )

            g.ax_joint.set_xlabel(f"{x_id} CRISPR-Cas9 (MOSA)")
            g.ax_joint.set_ylabel(f"{y_id} CRISPR-Cas9 (MOSA)")

            plt.gcf().set_size_inches(2, 2)

            PhenPred.save_figure(
                f"{plot_folder}/crispr/{self.timestamp}_lm_assoc_corrplot_{y_id}_{x_id}_{z_id}",
            )
