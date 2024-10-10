# %load_ext autoreload
# %autoreload 2

import os
import sys
import time

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
from PhenPred.vae.DatasetMOVE import CLinesDatasetMOVE
from PhenPred.vae.DatasetJAMIE import CLinesDatasetJAMIE
from PhenPred.vae.DatasetSCVAEIT import CLinesDatasetSCVAEIT
from PhenPred.vae.DatasetIClusterPlus import CLinesDatasetIClusterPlus
from PhenPred.vae.DatasetMoCluster import CLinesDatasetMoCluster
from PhenPred.vae.DatasetMixOmics import CLinesDatasetMixOmics
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
    start_time = time.time()

    # hyperparameters = Hypers.read_hyperparameters()
    # hyperparameters = Hypers.read_hyperparameters(timestamp="20231023_092657")
    hyperparameters = Hypers.read_hyperparameters(timestamp="20240830_110319")
    hyperparameters_7omics = Hypers.read_hyperparameters(timestamp="20231023_092657")

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

    if "skip_benchmarks" in hyperparameters and hyperparameters["skip_benchmarks"]:
        sys.exit(0)

    clines_db_7omics = CLinesDatasetDepMap23Q2(
        datasets=hyperparameters_7omics["datasets"],
        labels_names=hyperparameters_7omics["labels"],
        standardize=hyperparameters_7omics["standardize"],
        filter_features=hyperparameters_7omics["filter_features"],
        filtered_encoder_only=hyperparameters_7omics["filtered_encoder_only"],
        feature_miss_rate_thres=hyperparameters_7omics["feature_miss_rate_thres"],
    )

    train_7omics = CLinesTrain(
        clines_db_7omics,
        hyperparameters_7omics,
        verbose=hyperparameters_7omics["verbose"],
        stratify_cv_by=clines_db_7omics.samples_by_tissue(
            "Haematopoietic and Lymphoid"
        ),
    )
    train_7omics.run(run_timestamp=hyperparameters_7omics["load_run"])
    vae_imputed_7omics, vae_latent_7omics = train_7omics.load_vae_reconstructions()
    vae_predicted_7omics, _ = train_7omics.load_vae_reconstructions(mode="all")

    # Load imputed data
    vae_imputed, vae_latent = train.load_vae_reconstructions()
    vae_predicted, _ = train.load_vae_reconstructions(mode="all")

    mofa_imputed, mofa_latent = CLinesDatasetMOFA.load_reconstructions(clines_db)
    move_diabetes_imputed, move_diabetes_latent = (
        CLinesDatasetMOVE.load_reconstructions(clines_db)
    )
    jamie_imputed, jamie_latent = CLinesDatasetJAMIE.load_reconstructions(clines_db)
    scvaeit_imputed, scvaeit_latent = CLinesDatasetSCVAEIT.load_reconstructions(
        clines_db
    )

    _, mixOmics_latent = CLinesDatasetMixOmics.load_reconstructions(clines_db)
    _, iClusterPlus_latent = CLinesDatasetIClusterPlus.load_reconstructions(clines_db)
    _, moCluster_latent = CLinesDatasetMoCluster.load_reconstructions(clines_db)

    if "transcriptomics" in hyperparameters["datasets"]:
        # Transcriptomics benchmark
        samples_mgexp = ~clines_db.dfs["transcriptomics"].isnull().all(axis=1)

        gexp_gdsc = pd.read_csv(f"{data_folder}/transcriptomics.csv", index_col=0).T
        gexp_mosa = vae_imputed["transcriptomics"]
        gexp_move = move_diabetes_imputed["transcriptomics"]
        gexp_jamie = jamie_imputed["transcriptomics"]
        gexp_mofa = mofa_imputed["transcriptomics"]
        gexp_scvaeit = scvaeit_imputed["transcriptomics"]

        gexp_dfs = dict(
            [
                ("MOSA", gexp_mosa),
                ("MOVE", gexp_move),
                ("JAMIE", gexp_jamie),
                ("MOFA", gexp_mofa),
                ("scVAEIT", gexp_scvaeit),
            ]
        )

        samples = set(gexp_gdsc.index).intersection(gexp_mosa.index)
        genes = list(set(gexp_gdsc.columns).intersection(gexp_mosa.columns))

        for name in ["MOSA", "MOVE", "JAMIE", "MOFA", "scVAEIT"]:
            gexp_corr = pd.DataFrame(
                [
                    two_vars_correlation(
                        gexp_gdsc.loc[s, genes],
                        gexp_dfs[name].loc[s, genes],
                        method="pearson",
                        extra_fields=dict(sample=s, with_gexp=samples_mgexp.loc[s]),
                    )
                    for s in samples
                ]
            )

            _, ax = plt.subplots(1, 1, figsize=(0.5, 2), dpi=600)

            sns.boxplot(
                data=gexp_corr,
                x="with_gexp",
                y="corr",
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
                ylabel="Correlation between reconstructed\nand GDSC transcriptomics (Pearson's r)",
                xlabel=f"Sample with transcriptomics\nduring {name} training",
            )

            PhenPred.save_figure(
                f"{plot_folder}/{hyperparameters['load_run']}_reconstructed_gexp_correlation_boxplot_{name}"
            )

    # Run Latent Spaces Benchmark
    latent_benchmark = LatentSpaceBenchmark(
        train.timestamp,
        clines_db,
        vae_latent,
        mofa_latent,
        move_diabetes_latent,
        jamie_latent,
        scvaeit_latent,
        mixOmics_latent,
        iClusterPlus_latent,
        moCluster_latent,
    )

    if (
        "two_omics_benchmark" in hyperparameters
        and hyperparameters["two_omics_benchmark"]
    ):
        latent_markers = None
    else:
        latent_markers = clines_db.get_features(
            dict(
                metabolomics=[
                    "1-methylnicotinamide",
                    "uridine",
                    "alanine",
                ],
                transcriptomics=["VIM"],
            )
        )
    latent_benchmark.plot_latent_spaces(
        markers=latent_markers,
    )

    if (
        "two_omics_benchmark" in hyperparameters
        and not hyperparameters["two_omics_benchmark"]
    ):
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
            lw=0.0,
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

        # Run CRISPR benchmark
        print("Running CRISPR benchmark")
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

    # Run drug benchmark
    print("Running drug benchmark")
    dres_benchmark = DrugResponseBenchmark(
        train.timestamp,
        clines_db,
        vae_imputed,
        mofa_imputed,
        move_diabetes_imputed,
        jamie_imputed,
        scvaeit_imputed,
        vae_imputed_7omics,
    )
    dres_benchmark.run()

    # Run proteomics benchmark
    # print("Running proteomics benchmark")
    # proteomics_benchmark = ProteomicsBenchmark(
    #     train.timestamp,
    #     clines_db,
    #     vae_imputed,
    #     mofa_imputed,
    #     move_diabetes_imputed,
    #     jamie_imputed,
    # )
    # proteomics_benchmark.run()
    # if (
    #     "two_omics_benchmark" in hyperparameters
    #     and not hyperparameters["two_omics_benchmark"]
    # ):
    #     proteomics_benchmark.copy_number(
    #         proteomics_only=True,
    #     )

    # Make CV predictions
    # hyperparameters["skip_cv"] = False
    if not hyperparameters["skip_cv"]:
        print("Running mismatch benchmark with CV")
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

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"Total time: {hours:02d}:{minutes:02d}")
