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
import pandas as pd
from PhenPred.vae import plot_folder
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()

    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
        covariates=hyperparameters["covariates"],
    )
    clines_db.plot_samples_overlap()
    clines_db.plot_datasets_missing_values()

    # Train and predictions
    # train.timestamp = "2023-05-18_19:49:05"
    train = CLinesTrain(clines_db, hyperparameters)
    train.run()

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(train.timestamp)
    dres_benchmark.run()

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(train.timestamp)
    proteomics_benchmark.run()

    # Run CRISPR benchmark
    crispr_benchmark = CRISPRBenchmark(train.timestamp, clines_db)
    crispr_benchmark.run()

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(train.timestamp, clines_db)
    latent_benchmark.run()
    latent_benchmark.plot_latent_spaces(
        view_names=[],
        markers=pd.concat(
            [
                clines_db.dfs["transcriptomics"][["VIM", "CDH1"]],
                clines_db.dfs["metabolomics"][["1-methylnicotinamide"]],
                latent_benchmark.covariates["drug_responses"],
            ],
            axis=1,
        ),
    )

    # Write the hyperparameters to json file
    json.dump(
        hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
