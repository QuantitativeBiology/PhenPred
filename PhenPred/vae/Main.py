# %load_ext autoreload
# %autoreload 2

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
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.Train import CLinesTrain, CLinesTrainGMVAE
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


if __name__ == "__main__":
    # Class variables - Hyperparameters
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
    # clines_db.plot_samples_overlap()
    # clines_db.plot_datasets_missing_values()

    # Train and predictions
    if hyperparameters["model"] == "GMVAE":
        train = CLinesTrainGMVAE(
            clines_db,
            hyperparameters,
            stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
            k=100,
        )
    else:
        train = CLinesTrain(
            clines_db,
            hyperparameters,
            stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
        )

    # Run or load previous run
    if hyperparameters["load_run"] is None or hyperparameters["load_run"] == "":
        train.run()
    else:
        # train.timestamp = "20230706_101116"
        train.timestamp = hyperparameters["load_run"]

    # Load imputed data
    vae_imputed, vae_latent = train.load_vae_reconstructions()
    mofa_imputed, mofa_latent = CLinesDatasetMOFA.load_reconstructions(clines_db)

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(
        train.timestamp, clines_db, vae_latent, mofa_latent
    )
    latent_benchmark.plot_latent_spaces(
        markers=clines_db.get_features(
            dict(metabolomics=["1-methylnicotinamide"], proteomics=["VIM", "CDH1"])
        ),
    )

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed
    )
    dres_benchmark.run()

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed
    )
    proteomics_benchmark.run()
    proteomics_benchmark.copy_number([("SMAD4", "SMAD4")])

    # Run CRISPR benchmark
    crispr_benchmark = CRISPRBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed
    )
    crispr_benchmark.run()
    crispr_benchmark.plot_associations(
        [
            ("BRAF", "MAPK1", "BRAF_mut"),
            ("FLI1", "TRIM8", "FLI1_EWSR1_fusion"),
            ("KRAS", "RAF1", "KRAS_mut"),
            ("NRAS", "SHOC2", "NRAS_mut"),
        ]
    )

    # Write the hyperparameters to json file
    json.dump(
        hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
