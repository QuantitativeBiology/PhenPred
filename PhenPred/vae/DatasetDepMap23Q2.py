import torch
import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from PhenPred.Utils import scale
from torch.utils.data import Dataset
from scipy.stats import zscore, norm
from sklearn.mixture import GaussianMixture
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, PowerTransformer, normalize


class CLinesDatasetDepMap23Q2(Dataset):
    def __init__(
        self,
        datasets,
        labels_names=["tissue"],
        decimals=4,
        feature_miss_rate_thres=0.9,
        standardize=False,
        normalize_features=False,
        normalize_samples=False,
        filter_features=[],
        filtered_encoder_only=False,
    ):
        super().__init__()

        self.labels_names = labels_names
        self.datasets = datasets
        self.decimals = decimals
        self.feature_miss_rate_thres = feature_miss_rate_thres
        self.standardize = standardize
        self.normalize_features = normalize_features
        self.normalize_samples = normalize_samples
        self.filter_features = filter_features
        self.filtered_encoder_only = filtered_encoder_only

        # Read csv files
        self.dfs = {n: pd.read_csv(f, index_col=0) for n, f in self.datasets.items()}
        self.dfs = {
            n: df if n in ["crisprcas9", "copynumber"] else df.T
            for n, df in self.dfs.items()
        }

        # Dataset specific preprocessing
        for n in ["crisprcas9"]:
            if n in self.dfs:
                self.dfs[n].columns = self.dfs[n].columns.str.split(" ").str[0]

        self._remove_features_missing_values()
        self._build_samplesheet()
        self._samples_union()
        self._features_mask()

        if self.normalize_samples:
            self.dfs = {
                n: df if n in ["copynumber"] else self.normalize_dataset(df)
                for n, df in self.dfs.items()
            }

        self._standardize_dfs()
        self._import_mutations()
        self._import_fusions()
        self._import_growth()
        self._import_drug_targets()
        self._build_labels()

        self.x_mask = [
            torch.tensor(self.features_mask[n], dtype=torch.bool) for n in self.views
        ]

        # View names
        self.view_name_map = dict(
            copynumber="Copy number",
            mutations="Mutations",
            fusions="Fusions",
            methylation="Methylation",
            transcriptomics="Transcriptomics",
            proteomics="Proteomics",
            phosphoproteomics="Phosphoproteomics",
            metabolomics="Metabolomics",
            drugresponse="Drug response",
            crisprcas9="CRISPR-Cas9",
            growth="Growth",
        )

        print(self)

    def __str__(self) -> str:
        str = f"DepMap23Q2 | Samples = {len(self.samples):,}"

        for n, df in self.dfs.items():
            f_masked = df.shape[1] - self.features_mask[n].sum()
            str += f" | {self.view_name_map[n]} = {df.shape[1]:,} ({f_masked:,} masked)"

        str += f" | Labels = {self.labels_size:,}"
        return str

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = [df[idx] for df in self.views.values()]
        x_nans = [df[idx] for df in self.view_nans.values()]

        y = self.labels[idx]

        return x, y, x_nans, self.x_mask

    def _features_mask(self):
        self.features_mask = {}

        for n in self.dfs:
            self.features_mask[n] = pd.Series(
                np.ones(self.dfs[n].shape[1], dtype=bool),
                index=self.dfs[n].columns,
            )

            if n in self.filter_features:
                if n in ["crisprcas9"]:
                    self.dfs[n] = scale(self.dfs[n].T).T
                    self.features_mask[n] = (self.dfs[n] < -0.5).sum() > 0

                else:
                    thres = self.gaussian_mixture_std(self.dfs[n], plot_name=None)
                    self.features_mask[n] = self.dfs[n].std() > thres

    def _build_labels(self, min_obs=15):
        self.labels = []

        if "tissue" in self.labels_names:
            self.labels.append(pd.get_dummies(self.samplesheet["tissue"]))

        if "cancer_type" in self.labels_names:
            self.labels.append(pd.get_dummies(self.samplesheet["cancer_type"]))

        if "culture" in self.labels_names:
            self.labels.append(
                pd.concat(
                    [
                        pd.get_dummies(self.samplesheet["growth_properties_broad"]),
                        pd.get_dummies(self.samplesheet["growth_properties_sanger"]),
                    ],
                    axis=1,
                )
            )

        if "growth" in self.labels_names:
            self.labels.append(
                zscore(
                    self.growth[["day4_day1_ratio", "doubling_time_hours"]],
                    nan_policy="omit",
                )
            )

        if "mutations" in self.labels_names:
            self.labels.append(self.mutations.loc[:, self.mutations.sum() >= min_obs])

        if "fusions" in self.labels_names:
            self.labels.append(self.fusions.loc[:, self.fusions.sum() >= 5])

        if "msi" in self.labels_names:
            self.labels.append(
                self.ss_cmp["msi_status"].replace({"MSS": 0, "MSI": 1}).astype(float)
            )

        if "mofa" in self.labels_names:
            self.labels.append(CLinesDatasetMOFA.load_factors())

        # Concatenate
        self.labels = pd.concat(self.labels, axis=1)
        self.labels = self.labels.reindex(index=self.samples).fillna(0)

        # Props
        self.labels_name = self.labels.columns.tolist()
        self.labels_size = self.labels.shape[1]

        self.labels = torch.tensor(self.labels.values.astype(float), dtype=torch.float)

    def _import_drug_targets(self):
        self.drug_targets = pd.read_csv(
            f"{data_folder}/drugresponse_drug_targets.csv", index_col=0
        )["putative_gene_target"]

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

    def _import_growth(self):
        self.growth = (
            pd.read_csv(f"{data_folder}/growth_rate_20220907.csv")
            .drop("model_name", axis=1)
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
            self.views[n], self.view_scalers[n], self.view_nans[n] = self.process_df(
                n, df
            )
            self.view_feature_names[n] = list(df.columns)
            self.view_names.append(n)

    def normalize_dataset(self, df):
        l2_norms = np.sqrt(np.nansum(df**2, axis=1))
        df_norm = df / l2_norms[:, np.newaxis]
        return df_norm

    def process_df(self, df_name, df):
        to_standardize = (
            True if df_name not in ["copynumber"] and self.standardize else False
        )

        if self.normalize_features:
            scaler = PowerTransformer(method="yeo-johnson", standardize=to_standardize)
        else:
            scaler = StandardScaler(with_mean=to_standardize, with_std=to_standardize)

        x = scaler.fit_transform(df).round(self.decimals)

        x_nan = ~np.isnan(x)

        x[~x_nan] = np.nanmean(x)

        x = torch.tensor(x, dtype=torch.float)

        return x, scaler, x_nan

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

    def gaussian_mixture_std(self, df, plot_name=None):
        df_std = df.std(axis=0)

        gm = GaussianMixture(n_components=2).fit(df_std.to_frame())

        gm_means = gm.means_.reshape(-1)
        gm_std = np.sqrt(gm.covariances_.reshape(-1))

        def solve(m1, m2, std1, std2):
            a = 1 / (2 * std1**2) - 1 / (2 * std2**2)
            b = m2 / (std2**2) - m1 / (std1**2)
            c = (
                m1**2 / (2 * std1**2)
                - m2**2 / (2 * std2**2)
                - np.log(std2 / std1)
            )
            return np.roots([a, b, c])

        intersections = solve(gm_means[0], gm_means[1], gm_std[0], gm_std[1])

        if plot_name is not None:
            x = df_std.sort_values().values

            _, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=300)

            ax.hist(x, bins=100, density=True, color="#7f7f7f", alpha=0.5)
            ax.plot(x, norm.pdf(x, gm_means[0], gm_std[0]), lw=1, c="#1f77b4")
            ax.plot(x, norm.pdf(x, gm_means[1], gm_std[1]), lw=1, c="#aec7e8")

            for i in intersections:
                ax.axvline(i, linestyle="--", lw=0.5, color="#000000")
                ax.text(
                    i + 0.01,
                    ax.get_ylim()[1] * 0.9,
                    f"{i:.3f}",
                    rotation=90,
                    ha="left",
                    va="top",
                    fontsize=6,
                    color="#000000",
                )

            ax.set_xlabel(f"{plot_name} standard deviation")
            ax.set_ylabel("Density")

            PhenPred.save_figure(
                f"{plot_folder}/datasets_std_gaussian_mixture_{plot_name}"
            )

        return max(intersections)

    def cnv_convert_to_matrix(self):
        """
        Convert CNV data to matrix. This is done separately because CNV data is
        not in the same format (discrete) as the other data types (continous).

        For cell lines screened both by the Broad and Sanger with divergent annotations,
        we sort the CNV categories in the following order: Neutral, Deletion, Loss, Gain, Amplification
        and the first annotation is kept (i.e. preference is given to Neutral annotations)

        Values are mapped to the following values:
        Deletion: -2
        Loss: -1
        Neutral: 0
        Gain: 1
        Amplification: 2
        """

        cnv_df = pd.read_csv(f"{data_folder}/cnv_summary_20230303.csv")
        cnv_df["cn_category"] = pd.Categorical(
            cnv_df["cn_category"],
            categories=["Neutral", "Deletion", "Loss", "Gain", "Amplification"],
            ordered=True,
        )
        cnv_df = cnv_df.sort_values("cn_category")

        cnv = pd.pivot_table(
            cnv_df,
            index="model_id",
            columns="symbol",
            values="cn_category",
            aggfunc="first",
        )

        cnv_map = dict(Deletion=-2, Loss=-1, Neutral=0, Gain=1, Amplification=2)

        cnv = self.cnv.replace(cnv_map)

        cnv.to_csv(f"{data_folder}/cnv_summary_20230303_matrix.csv")

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
        plot_df.index = [self.view_name_map[i] for i in plot_df.index]
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

        ax.set_title(f"Cancer cell lines multi-omics dataset\n(n={plot_df.shape[1]:,})")

        PhenPred.save_figure(
            f"{plot_folder}/datasets_overlap_DepMap23Q2", extensions=["png"]
        )

    def plot_datasets_missing_values(
        self,
        datasets_names=[
            "copynumber",
            "methylation",
            "transcriptomics",
            "proteomics",
            "metabolomics",
            "drugresponse",
            "crisprcas9",
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

            ax.set_title(f"{self.view_name_map[n]} dataset")

            PhenPred.save_figure(
                f"{plot_folder}/datasets_missing_values_DepMap23Q2_{n}",
                extensions=["png"],
            )
