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
        label=hyperparameters["label"],
        datasets=hyperparameters["datasets"],
        covariates=hyperparameters["covariates"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    # Train and predictions
    # train.timestamp = "2023-06-08_12:56:58"
    heam_lines = "Haematopoietic and Lymphoid"
    heam_lines = (
        (clines_db.samplesheet["tissue"] == heam_lines)
        .loc[clines_db.samples]
        .astype(int)
    )

    train = CLinesTrain(
        clines_db,
        hyperparameters,
        stratify_cv_by=heam_lines,
    )
    train.run()

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(train.timestamp, clines_db)
    latent_benchmark.run()
    latent_benchmark.plot_latent_spaces(
        view_names=list(hyperparameters["datasets"]),
        markers_joint=pd.concat(
            [
                # clines_db.dfs["transcriptomics"][["VIM", "CDH1"]],
                clines_db.dfs["metabolomics"][["1-methylnicotinamide"]],
            ],
            axis=1,
        ),
        markers_views=clines_db.n_samples_views().sum().rename("N_Views").to_frame(),
    )

    # Run drug benchmark
    dres_benchmark = DrugResponseBenchmark(train.timestamp)
    dres_benchmark.run()

    # Run proteomics benchmark
    proteomics_benchmark = ProteomicsBenchmark(train.timestamp)
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
