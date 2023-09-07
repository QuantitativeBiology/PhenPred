import os
import sys

proj_dir = "/home/scai/PhenPred"
if not os.path.exists(proj_dir):
    proj_dir = "/Users/emanuel/Projects/PhenPred"
sys.path.extend([proj_dir])

import h5py
import numpy as np
import pandas as pd
from mofapy2.run.entry_point import entry_point
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae import data_folder, plot_folder
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


class MOFABencharmk:
    def __init__(self, hypers=None, hypers_mofa=None):
        self.hypers = Hypers.read_hyperparameters() if hypers is None else hypers

        if hypers_mofa is None:
            self.hypers_mofa = dict(
                scale_groups=True,
                scale_views=True,
                factors=self.hypers["latent_dim"],
                spikeslab_weights=True,
                ard_factors=True,
                ard_weights=True,
                iter=1000,
                convergence_mode="slow",
                startELBO=1,
                freqELBO=1,
                dropR2=True,
                verbose=False,
            )

        self.clines_db_mofa = CLinesDatasetMOFA()
        self.clines_db_original = CLinesDatasetDepMap23Q2(
            datasets=self.hypers["datasets"],
            labels_names=self.hypers["labels"],
            standardize=self.hypers["standardize"],
            filter_features=self.hypers["filter_features"],
            filtered_encoder_only=self.hypers["filtered_encoder_only"],
            feature_miss_rate_thres=self.hypers["feature_miss_rate_thres"],
        )

    def run(self):
        pass

    def run_mofa(self):
        self.parse_data_for_mofa()
        self.config_mofa()

        self.ent.build()
        self.ent.run()

        self.save_mofa()
        self.load_mofa()
        self.write_mofa()

    def save_mofa(self, outfile=None):
        if outfile is None:
            self.outfile = f"{data_folder}/mofa/MOFA_{self.hypers['dataname']}"
            self.outfile += f"_Factors{self.hypers['latent_dim']}"
        else:
            self.outfile = outfile

        self.ent.save(outfile=self.outfile + ".hdf5")

    def load_mofa(self, outfile=None):
        if outfile is None:
            outfile = self.outfile

        self.mofa_file = h5py.File(outfile + ".hdf5", "r")

        self.factors = self.get_factors(self.mofa_file)
        self.weights = self.get_weights(self.mofa_file)
        self.rsquare = self.get_rsquare(self.mofa_file)

    def write_mofa(self):
        # Save the factors, weights and rsquare
        self.factors.to_csv(self.outfile + "_factors.csv")

        for n in self.weights:
            self.weights[n].to_csv(self.outfile + f"_weights_{n}.csv")

        self.rsquare["groupA"].to_csv(self.outfile + f"_rsquare.csv")

    def config_mofa(self):
        self.ent = entry_point()

        self.ent.set_data_options(
            scale_groups=self.hypers_mofa["scale_groups"],
            scale_views=self.hypers_mofa["scale_views"],
        )

        # MOFA view order
        # 'copynumber', 'crisprcas9', 'drugresponse', 'labels', 'metabolomics', 'methylation', 'proteomics', 'transcriptomics'
        self.ent.set_data_df(
            self.mofa_db,
            likelihoods=[
                "gaussian",
                "gaussian",
                "gaussian",
                "bernoulli",
                "gaussian",
                "gaussian",
                "gaussian",
                "gaussian",
            ],
        )

        self.ent.set_model_options(
            factors=self.hypers_mofa["factors"],
            spikeslab_weights=self.hypers_mofa["spikeslab_weights"],
            ard_factors=self.hypers_mofa["ard_factors"],
            ard_weights=self.hypers_mofa["ard_weights"],
        )

        self.ent.set_train_options(
            iter=self.hypers_mofa["iter"],
            convergence_mode=self.hypers_mofa["convergence_mode"],
            startELBO=self.hypers_mofa["startELBO"],
            freqELBO=self.hypers_mofa["freqELBO"],
            dropR2=self.hypers_mofa["dropR2"],
            verbose=self.hypers_mofa["verbose"],
        )

    def parse_data_for_mofa(self):
        mofa_db, mofa_db_cols = [], ["sample", "group", "feature", "value", "view"]

        # Append dataset
        for n, df in self.clines_db_original.dfs.items():
            df_melt = df.stack().reset_index()
            df_melt.columns = ["sample", "feature", "value"]
            df_melt["group"] = "groupA"
            df_melt["view"] = n
            mofa_db.append(df_melt[mofa_db_cols])

        # Append labels
        df_melt = pd.DataFrame(
            self.clines_db_original.labels,
            index=self.clines_db_original.samples,
            columns=self.clines_db_original.labels_name,
        )
        df_melt = df_melt.drop(
            columns=["day4_day1_ratio", "doubling_time_hours"], errors="ignore"
        )
        df_melt = df_melt.astype(int).stack().reset_index()
        df_melt.columns = ["sample", "feature", "value"]
        df_melt["group"] = "groupA"
        df_melt["view"] = "labels"
        mofa_db.append(df_melt[mofa_db_cols])

        self.mofa_db = pd.concat(mofa_db)

    @staticmethod
    def get_factors(mofa_hdf5):
        z = mofa_hdf5["expectations"]["Z"]
        factors = pd.concat(
            [
                pd.DataFrame(
                    df,
                    columns=[s.decode("utf-8") for s in mofa_hdf5["samples"][k]],
                ).T
                for k, df in z.items()
            ]
        )
        factors.columns = [f"F{i + 1}" for i in range(factors.shape[1])]
        return factors

    @staticmethod
    def get_weights(mofa_hdf5):
        w = mofa_hdf5["expectations"]["W"]
        weights = {
            n: pd.DataFrame(
                df,
                index=[f"F{i + 1}" for i in range(df.shape[0])],
                columns=[f.decode("utf-8") for f in mofa_hdf5["features"][n]],
            ).T
            for n, df in w.items()
        }
        return weights

    @staticmethod
    def get_rsquare(mofa_hdf5):
        r2 = mofa_hdf5["variance_explained"]["r2_per_factor"]
        rsquare = {
            k: pd.DataFrame(
                df,
                index=[s.decode("utf-8") for s in mofa_hdf5["views"]["views"]],
                columns=[f"F{i + 1}" for i in range(df.shape[1])],
            )
            for k, df in r2.items()
        }
        return rsquare


if __name__ == "__main__":
    mofa = MOFABencharmk()
    mofa.run_mofa()
