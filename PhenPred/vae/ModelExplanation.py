# This file does not work with the existing SHAP implementation
# Custom changes are required to accept our input data format

import time
import torch
import pickle
import PhenPred
import numpy as np
from PhenPred.vae.Hypers import Hypers
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae import shap_folder, plot_folder

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    # 20240805_131847 all datasets, lambda_d = lambda_od = 0.001
    # 20240805_132345 all datasets without disentanglement
    timestamp = "20241210_000556"
    hyperparameters = Hypers.read_hyperparameters(timestamp=timestamp)

    # Load the first dataset
    clines_db = CLinesDatasetDepMap23Q2(
        labels_names=hyperparameters["labels"],
        datasets=hyperparameters["datasets"],
        feature_miss_rate_thres=hyperparameters["feature_miss_rate_thres"],
        standardize=hyperparameters["standardize"],
        filter_features=hyperparameters["filter_features"],
        filtered_encoder_only=hyperparameters["filtered_encoder_only"],
    )

    # Train and predictions
    train = CLinesTrain(
        clines_db,
        hyperparameters,
        stratify_cv_by=clines_db.samples_by_tissue("Haematopoietic and Lymphoid"),
        timestamp=timestamp,
    )

    start_time = time.time()

    train.load_model()

    # explain_target = "drugresponse"
    explain_target = "latent"
    # explain_target="metabolomics"
    # explain_target="copynumber"
    # explain_target="proteomics"
    # explain_target="crisprcas9"
    # explain_target="transcriptomics"
    # explain_target="methylation"

    explanation = train.run_shap(explain_target=explain_target)

    end_time = time.time()
    runtime = end_time - start_time
    print(
        "Runtime: {:02d}:{:02d}:{:02d}".format(
            int(runtime // 3600), int((runtime % 3600) // 60), int(runtime % 60)
        )
    )

    # Save shap values in dataframe format
    train.save_shap_top200_features(
        explanation.values, explain_target
    )  # only top 200 features per omic
    train.save_shap(explanation.values, explain_target)

    # Save Explanation object
    with open(
        f"{shap_folder}/files/{train.timestamp}_explanation_{explain_target}.pkl", "wb"
    ) as f:
        pickle.dump(explanation, f)
