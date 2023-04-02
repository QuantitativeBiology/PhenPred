import sys

sys.path.extend(["/home/egoncalves/PhenPred"])

import PhenPred
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler


_data_folder = "/data/benchmarks/clines/"
_dirPlots = "/home/egoncalves/PhenPred/reports/vae/"
_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


class ProteomicsBenchmark:
    def __init__(self, timestamp):
        # Original dataset
        self.df_original = pd.read_csv(f"{_data_folder}/proteomics.csv", index_col=0).T

        # Fully imputed autoencoder dataset
        self.df_vae = pd.read_csv(
            f"{_dirPlots}/files/{timestamp}_proteomics.csv.gz", index_col=0
        )

        # MOFA imputed dataset
        self.df_mofa = pd.read_csv(f"{_data_folder}/proteomicsMOFA.csv", index_col=0).T

        # Independent proteomics dataset - CCLE
        self.df_ccle = pd.read_csv(f"{_data_folder}/proteomicsCCLE.csv", index_col=0).T

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

    def run(self):
        pass

    def place_imputed_values_in_nans(self):
        pass
