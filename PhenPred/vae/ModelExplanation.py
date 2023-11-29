# This file does not work with the existing SHAP implementation
# Custom changes are required to accept our input data format

import os
import sys
import time

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

if __name__ == "__main__":
    hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_092657")

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
    train = CLinesTrain(
        clines_db,
        hyperparameters,
        stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
        timestamp=hyperparameters["load_run"]
    )


    start_time = time.time()

    train.load_model()

    # train.run_shap(explain_target="latent")
    # train.run_shap(explain_target="drugresponse")
    
    # train.run_shap(explain_target="metabolomics")
    # train.run_shap(explain_target="copynumber")
    # train.run_shap(explain_target="proteomics")
    train.run_shap(explain_target="crisprcas9")
    train.run_shap(explain_target="transcriptomics")
    train.run_shap(explain_target="methylation")

    end_time = time.time()
    runtime = end_time - start_time
    print("Runtime: {:02d}:{:02d}:{:02d}".format(int(runtime // 3600), int((runtime % 3600) // 60), int(runtime % 60)))
