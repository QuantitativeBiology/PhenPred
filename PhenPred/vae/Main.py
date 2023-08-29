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
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkMismatch import MismatchBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark


if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()
    # hyperparameters = Hypers.read_hyperparameters(timestamp="20230725_114145")

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
        stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
    )

    train.run(run_timestamp=hyperparameters["load_run"])

    # Load imputed data
    vae_imputed, vae_latent = train.load_vae_reconstructions()
    vae_predicted, _ = train.load_vae_reconstructions(mode="all")

    mofa_imputed, mofa_latent = CLinesDatasetMOFA.load_reconstructions(clines_db)

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(
        train.timestamp, clines_db, vae_latent, mofa_latent
    )
    latent_benchmark.plot_latent_spaces(
        markers=clines_db.get_features(
            dict(
                metabolomics=[
                    "1-methylnicotinamide",
                    "uridine",
                    "alanine",
                ],
                crisprcas9=["FAM50A", "ARF4", "MCL1"],
                transcriptomics=["VIM", "CDH1", "FDXR"],
                copynumber=[
                    "PCM1",
                    "MYC",
                ],
                drugresponse=[
                    "1079;Dasatinib;GDSC2",
                ],
                proteomics=["MTDH"],
            )
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
    proteomics_benchmark.copy_number(
        proteomics_only=True,
    )

    # Run CRISPR benchmark
    crispr_benchmark = CRISPRBenchmark(
        train.timestamp, clines_db, vae_imputed, mofa_imputed
    )
    # crispr_benchmark.run()
    crispr_benchmark.gene_skew_correlation()
    crispr_benchmark.plot_associations(
        [
            ("BRAF", "MAPK1", "BRAF_mut"),
            ("FLI1", "TRIM8", "FLI1_EWSR1_fusion"),
            ("KRAS", "RAF1", "KRAS_mut"),
            ("NRAS", "SHOC2", "NRAS_mut"),
        ]
    )

    # Run mismatch benchmark
    mismatch_benchmark = MismatchBenchmark(train.timestamp, clines_db, vae_predicted)
    mismatch_benchmark.run()

    # Write the hyperparameters to json file
    json.dump(
        hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
