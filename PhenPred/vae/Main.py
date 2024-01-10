# %load_ext autoreload
# %autoreload 2

import os
import sys

from sqlalchemy import column

proj_dir = os.getcwd()
if proj_dir not in sys.path:
    sys.path.append(proj_dir)

import json
import torch
import PhenPred
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PhenPred.vae.Hypers import Hypers
from sklearn.model_selection import KFold
from PhenPred.vae.Train import CLinesTrain
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae import plot_folder, data_folder
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.BenchmarkCRISPR import CRISPRBenchmark
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkMismatch import MismatchBenchmark
from PhenPred.vae.BenchmarkProteomics import ProteomicsBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2


torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    # Class variables - Hyperparameters

    hyperparameters = Hypers.read_hyperparameters()
    # hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_092657")

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

    # Transcriptomics benchmark
    samples_mgexp = ~clines_db.dfs["transcriptomics"].isnull().all(axis=1)

    gexp_gdsc = pd.read_csv(f"{data_folder}/transcriptomics.csv", index_col=0).T
    gexp_move = vae_imputed["transcriptomics"]

    samples = set(gexp_gdsc.index).intersection(gexp_move.index)
    genes = list(set(gexp_gdsc.columns).intersection(gexp_move.columns))

    gexp_corr = pd.DataFrame(
        [
            two_vars_correlation(
                gexp_gdsc.loc[s, genes],
                gexp_move.loc[s, genes],
                method="pearson",
                extra_fields=dict(sample=s, with_gexp=samples_mgexp.loc[s]),
            )
            for s in samples
        ]
    )

    _, ax = plt.subplots(1, 1, figsize=(2, 0.75), dpi=600)

    sns.boxplot(
        data=gexp_corr,
        x="corr",
        y="with_gexp",
        orient="h",
        palette="tab20c",
        linewidth=0.3,
        fliersize=1,
        notch=True,
        saturation=1.0,
        showcaps=False,
        boxprops=dict(linewidth=0.5, edgecolor="black"),
        whiskerprops=dict(linewidth=0.5, color="black"),
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            markersize=1.0,
            linestyle="none",
            markeredgecolor="none",
            alpha=0.6,
        ),
        medianprops=dict(linestyle="-", linewidth=0.5),
        ax=ax,
    )

    ax.set(
        title=f"",
        xlabel="Correlation between reconstructed\nand GDSC transcriptomics (Pearson's r)",
        ylabel="Sample\nwith transcriptomics",
    )

    PhenPred.save_figure(
        f"{plot_folder}/{hyperparameters['load_run']}_reconstructed_gexp_correlation_boxplot"
    )

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

    if g.ax_cbar:
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
    # hyperparameters["skip_cv"] = False
    if not hyperparameters["skip_cv"]:
        if hyperparameters["load_run"] is None:
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

    # Write the hyperparameters to json file
    json.dump(
        hyperparameters,
        open(f"{plot_folder}/files/{train.timestamp}_hyperparameters.json", "w"),
        indent=4,
        default=lambda o: "<not serializable>",
    )
