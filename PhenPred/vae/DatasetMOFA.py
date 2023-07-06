import os
import pandas as pd
from PhenPred.vae import data_folder
from PhenPred.vae.Hypers import Hypers


class CLinesDatasetMOFA:
    @staticmethod
    def load_reconstructions(data, mode="nans_only", hypers=None, dfs=None):
        """
        Load imputed data and latent space from files. "nans_only" mode, original
        measurements are mantained and only NaNs are imputed. "all" mode all
        data is imputed.

        Parameters
        ----------
        mode : str, optional
            Loading mode of imputed data, by default "nans_only"

        Returns
        -------
        dict
            Dictionary of imputed dataframes
            pandas.DataFrame
                Latent space

        Raises
        ------
        ValueError
            If mode is not "nans_only" or "all"

        """

        if mode not in ["nans_only", "all"]:
            raise ValueError(f"Invalid mode {mode}")

        if hypers is None:
            hypers = Hypers.read_hyperparameters()
        elif isinstance(hypers, str):
            hypers = Hypers.read_hyperparameters(hypers)

        # Dataset details
        dataset_name = f"MOFA_{hypers['dataname']}"
        dataset_name += f"_Factors50"

        ddir = f"{data_folder}/mofa/{dataset_name}"

        if dfs is None:
            dfs = data.dfs

        dfs_imputed = {}
        for n in dfs:
            df_file = f"{ddir}_predicted_{n}.csv"

            if not os.path.isfile(df_file):
                continue

            df_imputed = pd.read_csv(df_file, index_col=0).T
            df_imputed.columns = [c.split("_")[0] for c in df_imputed]

            if mode == "nans_only":
                df_imputed = data.dfs[n].combine_first(df_imputed)

            dfs_imputed[n] = df_imputed

        # Load latent space
        joint_latent = dict(
            factors=pd.read_csv(f"{ddir}_factors.csv", index_col=0),
            rsquare=pd.read_csv(f"{ddir}_rsquare.csv", index_col=0),
            weights={
                n: pd.read_csv(f"{ddir}_weights_{n}.csv", index_col=0)
                for n in data.dfs
                if n not in ["copynumber"]
            },
        )

        return dfs_imputed, joint_latent
