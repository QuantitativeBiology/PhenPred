# %load_ext autoreload
# %autoreload 2

import os
import sys
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

if __name__ == "__main__":
    # Class variables - Hyperparameters
    hyperparameters = Hypers.read_hyperparameters()

    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        datasets=hyperparameters["datasets"],
        labels_names=hyperparameters["labels"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
    )

    # --- Supplementary Table 1 ---
    ss = clines_db.n_samples_views().T
    ss.columns = [clines_db.view_name_map[i] for i in ss]
    ss = ss.replace({1: "Yes", 0: "No"})

    # Meta data
    ss_metadata = clines_db.ss_cmp.groupby("model_id")[
        [
            "model_name",
            "synonyms",
            "tissue",
            "cancer_type",
            "model_type",
            "growth_properties",
            "BROAD_ID",
            "CCLE_ID",
        ]
    ].first()

    # Growth rate
    ss_growth = clines_db.growth[["day4_day1_ratio", "doubling_time_hours"]]

    # Concat
    samplesheet = (
        pd.concat([ss_metadata, ss_growth, ss], axis=1)
        .dropna(how="all")
        .loc[clines_db.samples]
    )

    # Export samplesheet as excel
    samplesheet.to_excel("reports/vae/SupplementaryTables/SupplementaryTable1.xlsx")

    # --- Supplementary Table 2 ---
    latent_dim = pd.read_csv(
        "reports/vae/files/20231023_153637_latent_joint.csv.gz", index_col=0
    ).loc[clines_db.samples]
    latent_dim.to_csv("reports/vae/SupplementaryTables/SupplementaryTable2.csv")

    # --- Supplementary Table 3 ---
    labels = pd.DataFrame(
        clines_db.labels, index=clines_db.samples, columns=clines_db.labels_name
    ).to_csv("reports/vae/SupplementaryTables/SupplementaryTable3.csv")

    # --- Supplementary Table 4 ---
    with pd.ExcelWriter(
        "reports/vae/SupplementaryTables/SupplementaryTable4.xlsx"
    ) as writer:
        for k, v in clines_db.features_mask.items():
            pd.Series(v).rename("mask").to_excel(writer, sheet_name=k)

    # --- Supplementary Table 5 ---
    cv_test_crispr = pd.read_csv(
        "reports/vae/mismatch/20231023_092657_crisprcas9_cvtest.csv.gz", index_col=0
    ).sort_values("pearson", ascending=False)

    cv_test_drug = pd.read_csv(
        "reports/vae/mismatch/20231023_092657_drugresponse_cvtest.csv.gz", index_col=0
    ).sort_values("pearson", ascending=False)

    with pd.ExcelWriter(
        "reports/vae/SupplementaryTables/SupplementaryTable5.xlsx"
    ) as writer:
        cv_test_crispr.to_excel(writer, sheet_name="CRISPR-Cas9")
        cv_test_drug.to_excel(writer, sheet_name="Drug-response")

    # --- Supplementary Table 6 ---
    proteomics_corr = pd.read_csv(
        "/Users/emanuel/Projects/PhenPred/reports/vae/proteomics/20231023_092657_ccle_correlations.csv",
        index_col=0,
    )
    proteomics_corr.to_excel(
        "reports/vae/SupplementaryTables/SupplementaryTable6.xlsx", index=False
    )

    # --- Supplementary Table 7 ---
    crispr_associations = pd.read_csv(
        "reports/vae/crispr/20231023_092657_genomics_crisprcas9.csv.gz"
    )
    crispr_associations.to_excel(
        "reports/vae/SupplementaryTables/SupplementaryTable7.xlsx", index=False
    )
