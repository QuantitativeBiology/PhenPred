# This file does not work with the existing SHAP implementation
# Custom changes are required to accept our input data format

import os
import sys

proj_dir = "/home/scai/PhenPred"
if not os.path.exists(proj_dir):
    proj_dir = "/Users/emanuel/Projects/PhenPred"
sys.path.extend([proj_dir])

import json
import PhenPred
import argparse
import pandas as pd
from PhenPred.vae import plot_folder
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
import shap

TIMESTAMP = "20230814_214435"
if __name__ == "__main__":
    hyperparameters = Hypers.read_hyperparameters()

    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        labels_names=hyperparameters["labels"],
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
    )

    # Train and predictions
    gmvae_args_dict = (
        dict(
            k=100,
            init_temp=1.0,
            decay_temp=1.0,
            hard_gumbel=0,
            min_temp=0.5,
            decay_temp_rate=0.013862944,
        )
        if hyperparameters["model"] == "GMVAE"
        else None
    )

    train = CLinesTrain(
        clines_db,
        hyperparameters,
        stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
        timestamp=TIMESTAMP
    )
    train.load_model()
    train.run_shap()
