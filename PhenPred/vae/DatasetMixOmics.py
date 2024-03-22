import os
import pandas as pd
from PhenPred.vae import data_folder
from PhenPred.vae.Hypers import Hypers


class CLinesDatasetMixOmics:
    @staticmethod
    def load_reconstructions(
        data, mode="nans_only", hypers=None, dfs=None, n_factors=210
    ):
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
        ddir = f"{data_folder}/mixOmics/"

        if dfs is None:
            dfs = data.dfs

        dfs_imputed = {}

        # Load latent space
        joint_latent = dict(
            factors=pd.read_csv(
                f"{ddir}/diabolo_latent_{n_factors}dims.csv", index_col=0
            )
        )

        return dfs_imputed, joint_latent
