import json
import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import data_folder, plot_folder, files_folder

# TODO: Conditinal e covariates are largely overlapping features. Merge.


class CLinesDatasetDepMap23Q2(Dataset):
    def __init__(
        self,
        datasets,
        decimals=4,
        conditional_field="tissue",
        feature_miss_rate_thres=0.9,
        covariates=None,
    ):
        self.datasets = datasets
        self.decimals = decimals
        self.conditional_field = conditional_field
        self.feature_miss_rate_thres = feature_miss_rate_thres

        # Read csv files
        self.dfs = {n: pd.read_csv(f, index_col=0) for n, f in self.datasets.items()}
        self.dfs = {
            n: df if n in ["crisprcas9", "transcriptomics", "copynumber"] else df.T
            for n, df in self.dfs.items()
        }

        for n in ["crisprcas9", "transcriptomics", "copynumber"]:
            if n in self.dfs:
                self.dfs[n].columns = self.dfs[n].columns.str.split(" ").str[0]

        if "copynumber" in self.dfs:
            self.dfs["copynumber"] = self.dfs["copynumber"].dropna(how="all", axis=1)

        # Build samplesheet and parse sample IDs
        self._build_samplesheet()
        self._samples_union()
        self._remove_features_missing_values()
        self._standardize_dfs()
        self._conditional_df(conditional_field)

        if covariates is not None:
            self.covariates = pd.get_dummies(
                self.samplesheet.loc[self.samples, covariates]
            )

        self.view_names = list(self.views.keys())

        print(
            f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}] Samples = {len(self.samples)}"
        )

    def _map_genesymbols(self):
        gene_map = (
            pd.read_csv(f"{data_folder}/gene_symbols_hgnc.csv")
            .groupby("Input")["Approved symbol"]
            .first()
        )

        for n, df in self.dfs.items():
            if n in ["methylation", "transcriptomics", "proteomics", "crisprcas9"]:
                self.dfs[n] = df.rename(index=gene_map)

    def _build_samplesheet(self):
        col_rename = dict(
            ModelID="BROAD_ID",
            SangerModelID="model_id",
            SampleCollectionSite="tissue",
            OncotreeLineage="cancer_type",
        )
        cols = ["model_id", "BROAD_ID", "tissue", "cancer_type"]

        # Import samplesheets
        ss_cmp = pd.read_csv(f"{data_folder}/model_list_20230505.csv")

        ss_depmap = pd.read_csv(f"{data_folder}/depmap23Q2/Model.csv")
        ss_depmap.rename(columns=col_rename, inplace=True)

        # Map sample IDs to Sanger IDs
        self.samplesheet = pd.concat(
            [
                ss_cmp[cols].dropna().assign(source="sanger"),
                ss_depmap[cols].dropna().assign(source="broad"),
            ]
        )

        # Replace datafram columns using dict
        self.dfs = {
            n: df.rename(index=self.samplesheet.groupby("BROAD_ID").first()["model_id"])
            for n, df in self.dfs.items()
        }

        # Build samplesheet
        self.samplesheet = self.samplesheet.groupby("model_id").first()

        # Match tissue names
        self.samplesheet["tissue"].replace(
            dict(
                large_intestine="Large Intestine",
                lung="Lung",
                ovary="Ovary",
                upper_aerodigestive_tract="Other tissue",
                ascites="Other tissue",
                pleural_effusion="Other tissue",
            ),
            inplace=True,
        )

        # Growth properties
        self.samplesheet["growth_properties_sanger"] = (
            ss_cmp.set_index("model_id")
            .reindex(self.samplesheet.index)["growth_properties"]
            .fillna("Unknown")
            .values
        )

        self.samplesheet["growth_properties_broad"] = (
            ss_depmap.set_index("BROAD_ID")
            .reindex(self.samplesheet["BROAD_ID"])["GrowthPattern"]
            .fillna("Unknown")
            .values
        )

    @classmethod
    def transform_crispr(cls, df, essential=None, non_essential=None, metric=np.median):
        if essential is None:
            essential = cls.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = cls.get_non_essential_genes(return_series=False)

        ess_metric = metric(df.reindex(essential).dropna(), axis=0)
        ness_metric = metric(df.reindex(non_essential).dropna(), axis=0)

        df = df.subtract(ness_metric).divide(ness_metric - ess_metric)

        return df

    @classmethod
    def get_essential_genes(cls, dfile="EssentialGenes.csv", return_series=True):
        geneset = set(pd.read_csv(f"{files_folder}/{dfile}", sep="\t")["gene"])

        if return_series:
            geneset = pd.Series(list(geneset)).rename("essential")

        return geneset

    @classmethod
    def get_non_essential_genes(cls, dfile="NonessentialGenes.csv", return_series=True):
        geneset = set(pd.read_csv(f"{files_folder}/{dfile}", sep="\t")["gene"])

        if return_series:
            geneset = pd.Series(list(geneset)).rename("non-essential")

        return geneset

    def _conditional_df(self, field):
        self.conditional = pd.get_dummies(self.samplesheet[field].loc[self.samples])

    def _standardize_dfs(self):
        self.views = dict()
        self.view_scalers = dict()
        self.view_feature_names = dict()
        self.view_nans = dict()

        for n, df in self.dfs.items():
            self.views[n], self.view_scalers[n], self.view_nans[n] = self.process_df(df)
            self.view_feature_names[n] = list(df.columns)

    def _samples_union(self):
        # Union samples
        self.samples = pd.concat(
            [pd.Series(df.index) for df in self.dfs.values()], axis=0
        ).value_counts()

        # Keep only samples that are in at least 2 datasets
        self.samples = self.samples[self.samples > 1]

        self.samples = set(self.samples.index).intersection(set(self.samplesheet.index))
        self.samples -= {"SIDM00189", "SIDM00650"}
        self.samples = list(self.samples)

        self.dfs = {n: df.reindex(index=self.samples) for n, df in self.dfs.items()}

    def _remove_features_missing_values(self):
        # Remove features with more than 50% of missing values
        for n in ["proteomics", "metabolomics", "drugresponse", "crisprcas9"]:
            if n in self.dfs:
                print(f"Remove miss features: {self.feature_miss_rate_thres}")
                print(f"\tBefore: {self.dfs[n].shape}")
                self.dfs[n] = self.dfs[n].loc[
                    :, self.dfs[n].isnull().mean() < self.feature_miss_rate_thres
                ]
                print(f"\tAfter: {self.dfs[n].shape}")

    def process_df(self, df):
        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(df).round(self.decimals)

        x_nan = np.isnan(x)
        x = np.nan_to_num(x, copy=False)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float)

        return x, scaler, x_nan

    def plot_samples_overlap(self):
        plot_df = (
            pd.DataFrame({n: (~df.isnull()).sum(1) != 0 for n, df in self.dfs.items()})
            .astype(int)
            .T
        )
        plot_df = plot_df[plot_df.sum().sort_values(ascending=False).index]
        plot_df = plot_df.loc[:, plot_df.sum() > 0]
        plot_df = plot_df.loc[plot_df.sum(1).sort_values(ascending=False).index]

        plot_df.T.to_csv(f"{plot_folder}/datasets_overlap.csv")

        nsamples = plot_df.sum(1)

        cmap = sns.color_palette("tab20").as_hex()
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap",
            [cmap[0], cmap[1]],
            2,
        )

        _, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

        sns.heatmap(plot_df, xticklabels=False, cmap=cmap, cbar=False, ax=ax)

        for i, c in enumerate(plot_df.index):
            ax.text(
                20, i + 0.5, f"N={nsamples[c]:,}", ha="left", va="center", fontsize=6
            )

        ax.set_title(f"Cancer cell lines multi-omics dataset (N={plot_df.shape[1]:,})")

        plt.savefig(
            f"{plot_folder}/datasets_overlap_DepMap23Q2.pdf", bbox_inches="tight"
        )
        plt.close("all")

    def plot_datasets_missing_values(
        self,
        datasets_names=[
            "proteomics",
            "metabolomics",
            "drugresponse",
            "crisprcas9",
            "copynumber",
        ],
    ):
        for n in datasets_names:
            if n not in self.dfs:
                continue

            plot_df = ~self.dfs[n].isnull()
            plot_df = plot_df.loc[plot_df.sum(1) != 0].astype(int)
            plot_df = plot_df[plot_df.sum().sort_values(ascending=False).index]
            plot_df = plot_df.loc[:, plot_df.sum() > 0]
            plot_df = plot_df.loc[plot_df.sum(1).sort_values(ascending=False).index]

            cmap = sns.color_palette("tab20").as_hex()
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "Custom cmap",
                [cmap[0], cmap[1]],
                2,
            )

            miss_rate = 1 - plot_df.sum().sum() / np.prod(plot_df.shape)

            _, ax = plt.subplots(1, 1, figsize=(2, 1.5), dpi=600)

            sns.heatmap(
                plot_df,
                cmap=cmap,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax,
            )

            ax.set_xlabel(f"Features (N={plot_df.shape[1]:,})")
            ax.set_ylabel(f"Samples (N={plot_df.shape[0]:,})")

            ax.text(
                0.5,
                0.5,
                f"{miss_rate*100:.2f}%\nMissing rate",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
            )

            ax.set_title(f"{n} dataset")

            plt.savefig(
                f"{plot_folder}/datasets_missing_values_DepMap23Q2_{n}.png",
                bbox_inches="tight",
            )
            plt.close("all")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = [df[idx] for df in self.views.values()]
        x_nans = [df[idx] for df in self.view_nans.values()]
        y = self.covariates.iloc[idx].values
        return x, y, x_nans
