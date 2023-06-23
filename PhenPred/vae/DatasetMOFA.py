import pandas as pd
from torch.utils.data import Dataset
from PhenPred.vae import data_folder
from PhenPred.vae.Hypers import Hypers


class CLinesDatasetMOFA(Dataset):
    def __init__(
        self,
        hypers=None,
    ):
        if hypers is None:
            self.hypers = Hypers.read_hyperparameters()
        elif isinstance(hypers, str):
            self.hypers = Hypers.read_hyperparameters(hypers)
        else:
            self.hypers = hypers

        self.dataset_name = f"MOFA_{self.hypers['dataname']}"
        self.dataset_name += f"_Factors50"
        self.ddir = f"{data_folder}/mofa/{self.dataset_name}"
        self.view_names = list(self.hypers["datasets"])

        print(f"Loading MOFA results from {self.dataset_name}...")

        self.factors = pd.read_csv(f"{self.ddir}_factors.csv", index_col=0)

        self.rsquare = pd.read_csv(f"{self.ddir}_rsquare.csv", index_col=0)

        self.weights = {
            n: pd.read_csv(f"{self.ddir}_weights_{n}.csv", index_col=0)
            for n in self.view_names
        }

        self.predicted = {
            n: pd.read_csv(f"{self.ddir}_predicted_{n}.csv", index_col=0).T
            for n in self.view_names
        }

        self.imputed = {
            n: pd.read_csv(f"{self.ddir}_imputed_{n}.csv", index_col=0).T
            for n in self.view_names
        }

        for n in self.view_names:
            self.predicted[n].columns = [
                v.split("_")[0] for v in self.predicted[n].columns
            ]
            self.imputed[n].columns = [v.split("_")[0] for v in self.imputed[n].columns]
