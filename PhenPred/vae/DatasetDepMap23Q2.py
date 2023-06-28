import torch
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import zscore
from PhenPred.Utils import scale
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import data_folder, plot_folder


class CLinesDatasetDepMap23Q2(Dataset):
    def __init__(
        self,
        datasets,
        label,
        decimals=4,
        feature_miss_rate_thres=0.9,
    ):
        super().__init__()

        self.label = label
        self.datasets = datasets
        self.decimals = decimals
        self.feature_miss_rate_thres = feature_miss_rate_thres

        # Read csv files
        self.dfs = {n: pd.read_csv(f, index_col=0) for n, f in self.datasets.items()}
        self.dfs = {
            n: df if n in ["crisprcas9", "transcriptomics", "copynumber"] else df.T
            for n, df in self.dfs.items()
        }

        # Dataset specific preprocessing
        for n in ["crisprcas9", "transcriptomics", "copynumber"]:
            if n in self.dfs:
                self.dfs[n].columns = self.dfs[n].columns.str.split(" ").str[0]

        if "copynumber" in self.dfs:
            self.dfs["copynumber"] = self.dfs["copynumber"].dropna(how="all", axis=1)

        if "crisprcas9" in self.dfs:
            self.dfs["crisprcas9"] = scale(self.dfs["crisprcas9"].T).T
            self.dfs["crisprcas9"] = self.dfs["crisprcas9"].loc[
                :, (self.dfs["crisprcas9"] < -0.5).sum() > 0
            ]

        if "transcriptomics" in self.dfs:
            self.dfs["transcriptomics"] = self.dfs["transcriptomics"].loc[
                :, self.dfs["transcriptomics"].std() > 0.6
            ]

        self._build_samplesheet()
        self._samples_union()
        self._remove_features_missing_values()
        self._standardize_dfs()

        self._import_mutations()
        self._import_cnv()
        self._import_fusions()
        self._import_growth()

        self._build_labels()

        print(self)

    def __str__(self) -> str:
        str = f"DepMap23Q2 | Samples = {len(self.samples):,}"
        for n, df in self.dfs.items():
            str += f" | {n} = {df.shape[1]:,}"
        str += f" | Labels = {self.labels_size:,}"
        return str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = [df[idx] for df in self.views.values()]
        x_nans = [df[idx] for df in self.view_nans.values()]
        y = self.labels[idx]
        return x, y, x_nans

    def _build_labels(self, min_obs=15):
        # Tissue and cancer type
        tissue = pd.get_dummies(self.samplesheet["tissue"])
        cancer = pd.get_dummies(self.samplesheet["cancer_type"])

        # Culture conditions
        culture = pd.concat(
            [
                pd.get_dummies(self.samplesheet["growth_properties_broad"]),
                pd.get_dummies(self.samplesheet["growth_properties_sanger"]),
            ],
            axis=1,
        )

        # Growth
        growth = zscore(
            self.growth[["day4_day1_ratio", "doubling_time_hours"]],
            nan_policy="omit",
        )

        # Mutations
        mutations = self.mutations.loc[:, self.mutations.sum() >= min_obs]

        # Copy number variations
        cnv_map = dict(Deletion=-2, Loss=-1, Neutral=0, Gain=1, Amplification=2)
        cnv = self.cnv.replace(cnv_map).fillna(0)
        cnv = cnv.loc[:, ((cnv == -2).sum() + (cnv == 2).sum()) > min_obs]

        # Fusions
        fusions = self.fusions.loc[:, self.fusions.sum() >= 5]

        # MSI
        msi = self.ss_cmp["msi_status"].replace({"MSS": 0, "MSI": 1}).astype(float)

        # Concatenate
        self.labels = pd.concat(
            [tissue, cancer, culture, growth, mutations, cnv, fusions, msi],
            axis=1,
        )
        self.labels = self.labels.reindex(index=self.samples).fillna(0)

        # Props
        self.labels_name = self.labels.columns.tolist()
        self.labels_size = self.labels.shape[1]

        self.labels = torch.tensor(self.labels.values, dtype=torch.float)

    def _import_fusions(self):
        self.fusions = pd.read_csv(f"{data_folder}/Fusions_20221214.txt").assign(
            value=1
        )
        self.fusions["fusions"] = (
            self.fusions["gene_symbol_3prime"]
            + "_"
            + self.fusions["gene_symbol_5prime"]
        )

        self.fusions = pd.pivot_table(
            self.fusions,
            index="model_id",
            columns="fusions",
            values="value",
            fill_value=0,
        )

    def _import_mutations(self):
        self.mutations = (
            pd.read_csv(f"{data_folder}/mutations_summary_20230202.csv", index_col=0)
            .assign(value=1)
            .query("cancer_driver == True")
        )
        self.mutations = pd.pivot_table(
            self.mutations,
            index="model_id",
            columns="gene_symbol",
            values="value",
            aggfunc="first",
            fill_value=0,
        )

    def _import_cnv(self):
        self.cnv_df = pd.read_csv(f"{data_folder}/cnv_summary_20230303.csv")
        self.cnv_df["cn_category"] = pd.Categorical(
            self.cnv_df["cn_category"],
            categories=["Neutral", "Deletion", "Loss", "Gain", "Amplification"],
            ordered=True,
        )
        self.cnv_df = self.cnv_df.sort_values("cn_category")

        self.cnv = pd.pivot_table(
            self.cnv_df,
            index="model_id",
            columns="symbol",
            values="cn_category",
            aggfunc="first",
        )

    def _import_growth(self):
        self.growth = (
            pd.read_csv(f"{data_folder}/growth_rate_20220907.csv")
            .groupby("model_id")
            .mean()
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
        self.ss_cmp = pd.read_csv(f"{data_folder}/model_list_20230505.csv")

        self.ss_depmap = pd.read_csv(f"{data_folder}/depmap23Q2/Model.csv")
        self.ss_depmap.rename(columns=col_rename, inplace=True)

        # Map sample IDs to Sanger IDs
        self.samplesheet = pd.concat(
            [
                self.ss_cmp[cols].dropna().assign(source="sanger"),
                self.ss_depmap[cols].dropna().assign(source="broad"),
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
            self.ss_cmp.set_index("model_id")
            .reindex(self.samplesheet.index)["growth_properties"]
            .fillna("Unknown")
            .values
        )

        self.samplesheet["growth_properties_broad"] = (
            self.ss_depmap.set_index("BROAD_ID")
            .reindex(self.samplesheet["BROAD_ID"])["GrowthPattern"]
            .fillna("Unknown")
            .values
        )

    def _standardize_dfs(self):
        self.views = dict()
        self.view_scalers = dict()
        self.view_feature_names = dict()
        self.view_nans = dict()
        self.view_names = []

        for n, df in self.dfs.items():
            self.views[n], self.view_scalers[n], self.view_nans[n] = self.process_df(df)
            self.view_feature_names[n] = list(df.columns)
            self.view_names.append(n)

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
                self.dfs[n] = self.dfs[n].loc[
                    :, self.dfs[n].isnull().mean() < self.feature_miss_rate_thres
                ]

    def process_df(self, df):
        scaler = StandardScaler()
        x = scaler.fit_transform(df).round(self.decimals)

        x_nan = ~np.isnan(x)
        x = np.nan_to_num(x, copy=False)

        x = torch.tensor(x, dtype=torch.float)

        return x, scaler, x_nan

    def n_samples_views(self):
        counts = (
            pd.DataFrame({n: (~df.isnull()).sum(1) != 0 for n, df in self.dfs.items()})
            .astype(int)
            .T
        )
        counts = counts[counts.sum().sort_values(ascending=False).index]
        counts = counts.loc[:, counts.sum() > 0]
        counts = counts.loc[counts.sum(1).sort_values(ascending=False).index]
        return counts

    def samples_by_tissue(self, tissue):
        return (
            (self.samplesheet["tissue"] == tissue)
            .loc[self.samples]
            .astype(int)
            .rename(tissue)
        )

    def get_features(self, view_features_dict):
        return pd.concat(
            [
                self.dfs[v].reindex(columns=f).add_suffix(f"_{v}")
                for v, f in view_features_dict.items()
            ],
            axis=1,
        )

    def plot_samples_overlap(self):
        plot_df = self.n_samples_views()
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
