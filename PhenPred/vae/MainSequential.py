# %load_ext autoreload
# %autoreload 2

import os
import sys

proj_dir = os.getcwd()
if proj_dir not in sys.path:
    sys.path.append(proj_dir)
    
import json
import torch
import numpy as np
import PhenPred
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from PhenPred.vae import plot_folder
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from sklearn.model_selection import KFold
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkMismatch import MismatchBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark


torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    # Class variables - Hyperparameters

    hyperparameters = Hypers.read_hyperparameters()
    # hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_092657")

    datasets = {
        "proteomics": "data/clines//proteomics.csv",
        "metabolomics": "data/clines//metabolomics.csv",
        "drugresponse": "data/clines//drugresponse.csv",
        "crisprcas9": "data/clines//depmap23Q2/CRISPRGeneEffect.csv",
        "methylation": "data/clines//methylation.csv",
        "transcriptomics": "data/clines//depmap23Q2/OmicsExpressionGenesExpectedCountProfileVoom.csv",
        "copynumber": "data/clines//cnv_summary_20230303_matrix.csv",
    }

    loss_type_map = {
        "proteomics": "mean",
        "metabolomics": "mean",
        "drugresponse": "mean",
        "crisprcas9": "mean",
        "methylation": "mean",
        "transcriptomics": "mean",
        "copynumber": "macro",
    }

    selected_datasets = ["drugresponse"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remaining_datasets = [d for d in datasets if d not in selected_datasets]
    loss_records = []
    while len(remaining_datasets) > 0:
        current_best_combination = ["", 1000]
        for dataset in remaining_datasets:
            torch.cuda.empty_cache()

            tmp_new_combination = selected_datasets + [dataset]
            tmp_new_datasets_dict = {k: datasets[k] for k in tmp_new_combination}
            hyperparameters["datasets"] = tmp_new_datasets_dict
            hyperparameters["view_loss_recon_type"] = [
                loss_type_map[x] for x in tmp_new_combination
            ]
            # Load the first dataset
            clines_db = CLinesDatasetDepMap23Q2(
                datasets=hyperparameters["datasets"],
                labels_names=hyperparameters["labels"],
                standardize=hyperparameters["standardize"],
                filter_features=hyperparameters["filter_features"],
                filtered_encoder_only=hyperparameters["filtered_encoder_only"],
                feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
            )

            # Train and predictions
            train = CLinesTrain(
                clines_db,
                hyperparameters,
                verbose=hyperparameters["verbose"],
                stratify_cv_by=clines_db.samples_by_tissue(
                    "Haematopoietic and Lymphoid"
                ),
            )
            losses_df = train.run(
                run_timestamp=hyperparameters["load_run"], return_val_loss=True
            )
            val_df = losses_df[losses_df["type"] == "val"]

            val_df = val_df.groupby("cv").tail(1)
            val_drug_loss = val_df["mse_drugresponse"].mean()
            # val_crispr_loss = val_df["mse_crisprcas9"].mean()
            loss_records.append(
                {
                    "dataset": tmp_new_combination,
                    "val_drug_loss": val_drug_loss,
                    # "val_crispr_loss": val_crispr_loss,
                    "total_loss": val_drug_loss,
                }
            )
            if val_drug_loss < current_best_combination[1]:
                current_best_combination = [tmp_new_combination, val_drug_loss]
            # if val_drug_loss + val_crispr_loss < current_best_combination[1]:
            #     current_best_combination = [
            #         tmp_new_combination,
            #         val_drug_loss + val_crispr_loss,
            #     ]
            print(current_best_combination)
        # print(loss_records)
        selected_datasets = current_best_combination[0]
        print(selected_datasets)
        remaining_datasets = [d for d in datasets if d not in selected_datasets]

    loss_records_df = pd.DataFrame(loss_records)
    loss_records_df.to_csv(
        f"{plot_folder}/files/{timestamp}_sequential_losses.csv",
        index=False,
    )
