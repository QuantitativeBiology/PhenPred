# %load_ext autoreload
# %autoreload 2

import os
import sys

proj_dir = "/home/scai/PhenPred"
if not os.path.exists(proj_dir):
    proj_dir = "/Users/emanuel/Projects/PhenPred"
sys.path.extend([proj_dir])

import json
import torch
import numpy as np
import PhenPred
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
<<<<<<< HEAD
    # hyperparameters = Hypers.read_hyperparameters()
    hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_092657")
=======
    hyperparameters = Hypers.read_hyperparameters()
    # hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_153637")
>>>>>>> 924e377d40844b7ed4eb52da1adc4d57173cc2de

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
                transcriptomics=["VIM", "CDH1", "FDXR", "NNMT"],
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

    # Correlate features
    plot_df = clines_db.get_features(
        dict(
            metabolomics=[
                "1-methylnicotinamide",
            ],
            transcriptomics=["VIM", "CDH1", "NNMT"],
            proteomics=["VIM", "CDH1"],
        )
    )

    g = sns.clustermap(
        plot_df.corr(),
        cmap="RdYlGn",
        center=0,
        xticklabels=False,
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"fontsize": 5},
        fmt=".2f",
        linewidths=0.0,
        cbar_kws={"shrink": 0.5},
        figsize=(3.0, 1.5),
    )

    g.ax_cbar.set_ylabel("Pearson\ncorrelation")

    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    PhenPred.save_figure(
        f"{plot_folder}/selected_features_clustermap",
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
    crispr_benchmark.run()
    crispr_benchmark.gene_skew_correlation()
    crispr_benchmark.plot_associations(
        [
            ("BRAF", "MAPK1", "BRAF_mut"),
            ("FLI1", "TRIM8", "FLI1_EWSR1_fusion"),
            ("KRAS", "RAF1", "KRAS_mut"),
            ("NRAS", "SHOC2", "NRAS_mut"),
        ]
    )

    # Make CV predictions
    hyperparameters["skip_cv"] = False
    if not hyperparameters["skip_cv"]:
        _, cvtest_datasets = train.training(
            cv=KFold(n_splits=10, shuffle=True).split(train.data)
        )
        cvtest_datasets = {
            k: pd.read_csv(
                f"{plot_folder}/files/{train.timestamp}_imputed_{k}_cvtest.csv.gz",
                index_col=0,
            )
            for k in hyperparameters["datasets"]
        }

        # Run mismatch benchmark
        mismatch_benchmark = MismatchBenchmark(
            train.timestamp, clines_db, vae_predicted, cvtest_datasets
        )
        mismatch_benchmark.run()
