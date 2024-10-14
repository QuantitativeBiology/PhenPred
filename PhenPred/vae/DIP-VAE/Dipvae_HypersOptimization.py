# %load_ext autoreload
# %autoreload 2

import os
import sys
import time
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
from PhenPred.vae.DatasetMixOmics import CLinesDatasetMixOmics
from PhenPred.vae.BenchmarkDrug import DrugResponseBenchmark
from PhenPred.vae.BenchmarkLatentSpace import LatentSpaceBenchmark
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":

    val_mse_fold = []
    val_mse_mean_and_dipvae_metric = []
    lambda_values = [0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1, 10]

    for lambda_d in lambda_values:
        for lambda_od in lambda_values:

            if (lambda_d == 0 and lambda_od == 0) or (lambda_d != 0 and lambda_od != 0):

                hyperparameters = Hypers.read_hyperparameters(
                    hypers_json=f"{plot_folder}/files/optuna_MOSA_updated_model_weights_hyperparameters.json"
                )
                hyperparameters["skip_cv"] = False
                hyperparameters["num_epochs"] = 100

                if lambda_d != 0 and lambda_od != 0:
                    hyperparameters["lambda_d"] = lambda_d
                    hyperparameters["lambda_od"] = lambda_od
                    hyperparameters["dip_vae_type"] = "ii"

                print(f"\nlambda_d: {lambda_d}, lambda_od: {lambda_od}")

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

                train.run()
                train.save_losses()

                losses_df = pd.DataFrame(train.losses)

                # Reconstruction Performance metric
                val_mse_fold = []

                for cv_idx in losses_df["cv"].unique():

                    if (
                        losses_df.query(f"cv == {cv_idx}")["epoch"].iloc[-1]
                        == hyperparameters["num_epochs"]
                    ):

                        best_epoch = hyperparameters["num_epochs"]

                    else:

                        best_epoch = (
                            losses_df.query(f"cv == {cv_idx}")["epoch"].iloc[-1]
                            - train.early_stop_patience
                        )

                    val_mse = (
                        losses_df.query(f"cv == {cv_idx} & epoch == {best_epoch}")
                        .groupby("type")
                        .mean()
                        .loc["val", "reconstruction"]
                    )

                    val_mse_fold.append(val_mse)

                # Run without cv
                hyperparameters["skip_cv"] = True

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

                train.run()
                train.save_losses()

                # Load imputed data
                vae_imputed, vae_latent = train.load_vae_reconstructions()
                vae_predicted, _ = train.load_vae_reconstructions(mode="all")

                mofa_imputed, mofa_latent = CLinesDatasetMOFA.load_reconstructions(
                    clines_db
                )
                move_diabetes_imputed, move_diabetes_latent = (
                    CLinesDatasetMOVE.load_reconstructions(clines_db)
                )
                _, mixOmics_latent = CLinesDatasetMixOmics.load_reconstructions(
                    clines_db
                )
                
                # Run Latent Spaces Benchmark
                latent_benchmark = LatentSpaceBenchmark(
                    train.timestamp,
                    clines_db,
                    vae_latent,
                    mofa_latent,
                    move_diabetes_latent,
                    mixOmics_latent,
                )

                latent_benchmark.run()

                latent_benchmark.plot_latent_spaces(
                    markers=clines_db.get_features(
                        dict(
                            transcriptomics=["VIM", "CDH1"],
                        )
                    ),
                )

                # Correlate features
                plot_df = clines_db.get_features(
                    dict(
                        transcriptomics=["VIM", "CDH1", "NNMT"]
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

                # Run drug benchmark
                print("Running drug benchmark")
                dres_benchmark = DrugResponseBenchmark(
                    train.timestamp, clines_db, vae_imputed, mofa_imputed, move_diabetes_imputed
                )
                dres_benchmark.run()
                
                # Disentanglement Performance Metric
                correlations_np = np.triu(vae_latent.corr().to_numpy())
                total_abs_sum = np.sum(np.abs(correlations_np))
                diagonal_abs_sum = np.sum((np.diag(correlations_np)))
                off_diagonal_abs_sum = total_abs_sum - diagonal_abs_sum

                num_off_diagonal_terms = (
                    correlations_np.shape[0] * correlations_np.shape[0]
                    - correlations_np.shape[0]
                ) / 2
                mean_off_diagonal_abs_sum = (
                    off_diagonal_abs_sum / num_off_diagonal_terms
                )

                val_mse_mean_and_dipvae_metric.append(
                    [
                        lambda_d,
                        lambda_od,
                        np.mean(val_mse_fold),
                        mean_off_diagonal_abs_sum,
                    ]
                )

                # Write the hyperparameters to json file
                json.dump(
                    hyperparameters,
                    open(
                        f"{plot_folder}/files/{train.timestamp}_hyperparameters.json",
                        "w",
                    ),
                    indent=4,
                    default=lambda o: "<not serializable>",
                )


    rounded_array = np.round(val_mse_mean_and_dipvae_metric, decimals=7)

    np.save(f"{plot_folder}/files/dipvae_metrics.npy", rounded_array)