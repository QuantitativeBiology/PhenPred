import json
import torch
import PhenPred
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.stats as stats
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from PhenPred.vae import data_folder, plot_folder


class CLinesDataset(Dataset):
    def __init__(self, datasets, conditional_field="tissue"):
        # Read csv files
        self.dfs = {n: pd.read_csv(f, index_col=0).T for n, f in datasets.items()}

        self.samplesheet = pd.read_csv(
            f"{data_folder}/samplesheet.csv", index_col=0
        ).dropna(subset=["cancer_type", "tissue"])

        self._samples_union()
        self._remove_features_missing_values()
        self._standardize_dfs()
        self._conditional_df(conditional_field)

        self.view_names = list(self.views.keys())

        print(
            f"[{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}] Samples = {len(self.samples)}"
        )

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
        for n in ["proteomics", "metabolomics", "drugresponse"]:
            if n in self.dfs:
                self.dfs[n] = self.dfs[n].loc[:, self.dfs[n].isnull().mean() < 0.5]

    def process_df(self, df):
        # Normalize the data using StandardScaler
        scaler = StandardScaler()
        x = scaler.fit_transform(df)
        x_nan = np.isnan(x)
        x = np.nan_to_num(x, copy=False)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float)

        return x, scaler, x_nan

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = [df[idx] for df in self.views.values()]
        x_nans = [df[idx] for df in self.view_nans.values()]
        y = self.conditional.iloc[idx].values
        return x, y, x_nans
