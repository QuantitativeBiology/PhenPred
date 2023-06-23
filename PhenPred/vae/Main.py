# %load_ext autoreload
# %autoreload 2

import os
import sys

proj_dir = "/home/egoncalves/PhenPred"
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
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()

    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        label=hyperparameters["label"],
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    # Train and predictions
    # train.timestamp = "20230621_192218"
    train = CLinesTrain(
        clines_db,
        hyperparameters,
        stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
    )
    train.run()

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(train.timestamp, clines_db)
    latent_benchmark.plot_latent_spaces(
        markers=clines_db.get_features(
            dict(metabolomics=["1-methylnicotinamide"], proteomics=["VIM", "CDH1"])
        ),
    )

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(train.timestamp)
    dres_benchmark.run()

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(train.timestamp, clines_db)
    proteomics_benchmark.run()

    # Run CRISPR benchmark
    crispr_benchmark = CRISPRBenchmark(train.timestamp, clines_db)
    crispr_benchmark.run()

    # Write the hyperparameters to json file
    json.dump(
        hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
