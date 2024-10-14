import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import pytesseract
import cv2
import re
from PhenPred.Utils import two_vars_correlation
from PhenPred.vae.Train import CLinesTrain
from PhenPred.vae.DatasetDepMap23Q2 import CLinesDatasetDepMap23Q2
from PhenPred.vae.DatasetMOFA import CLinesDatasetMOFA
from PhenPred.vae.Hypers import Hypers
import seaborn as sns


def compute_mutual_information(latent_variables):
    """Compute the mutual information between each pair of latent variables."""
    num_latent_dims = latent_variables.shape[1]
    mutual_info_matrix = np.zeros((num_latent_dims, num_latent_dims))

    for i in range(num_latent_dims):
        for j in range(i + 1, num_latent_dims):
            mi = mutual_info_regression(
                latent_variables[:, i].reshape(-1, 1), latent_variables[:, j]
            )[0]
            mutual_info_matrix[i, j] = mi
            mutual_info_matrix[j, i] = mi

    return mutual_info_matrix


def compute_avg_mutual_information(latent_variables):
    """Compute the average mutual information among latent variables."""
    mutual_info_matrix = compute_mutual_information(latent_variables)

    # Compute the mean mutual information excluding the diagonal
    num_latent_dims = latent_variables.shape[1]
    sum_mi = np.sum(np.triu(mutual_info_matrix, k=1))
    num_pairs = num_latent_dims * (num_latent_dims - 1) / 2
    mean_mi = sum_mi / num_pairs

    return mean_mi


def PCA_variance_explained(latent_variables):
    """Compute the variance explained by each latent variable using PCA."""
    pca = PCA(n_components=latent_variables.shape[1])
    pca.fit(latent_variables)
    explained_variance = pca.explained_variance_ratio_

    return explained_variance


def variance_of_PCs_explained_variance(latent_variables):
    explained_variance = PCA_variance_explained(latent_variables)

    variance_gap = np.var(explained_variance)

    return variance_gap, explained_variance


def mse_new_drugresponse_dataset(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Apply noise reduction
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

    # Perform OCR
    text = pytesseract.image_to_string(blurred_image)

    # Regular expression to find the MSE value
    mse_pattern = r"MSE\s*=\s*([\d\.\-]+)"

    # Search for the pattern in the extracted text
    mse_match = re.search(mse_pattern, text)

    return mse_match


def corr_VIM_CDH1 (vae_latent_nodipvae, vae_latent_dipvae, markers):
    samples = list(set(vae_latent_dipvae.index).intersection(markers.index))

    corr_nodipvae = pd.DataFrame(
        [
            two_vars_correlation(
                markers.loc[samples, l],
                vae_latent_nodipvae.loc[samples, f],
                method="pearson",
                extra_fields=dict(variable=f, marker=l),
            )
            for f in vae_latent_nodipvae
            for l in markers
        ]
    )

    corr_dipvae = pd.DataFrame(
        [
            two_vars_correlation(
                markers.loc[samples, l],
                vae_latent_dipvae.loc[samples, f],
                method="pearson",
                extra_fields=dict(variable=f, marker=l),
            )
            for f in vae_latent_dipvae
            for l in markers
        ]
    )

    # Separate data by marker
    df_vim_nodipvae = corr_nodipvae[corr_nodipvae["marker"] == markers.columns[0]]
    df_cdh1_nodipvae = corr_nodipvae[corr_nodipvae["marker"] == markers.columns[1]]

    df_vim_dipvae = corr_dipvae[corr_dipvae["marker"] == markers.columns[0]]
    df_cdh1_dipvae = corr_dipvae[corr_dipvae["marker"] == markers.columns[1]]

    # Sorted dataframe by decreasing absolute correlation value
    df_vim_nodipvae_sorted = df_vim_nodipvae.reindex(df_vim_nodipvae["corr"].abs().sort_values(ascending=False).index)
    df_cdh1_nodipvae_sorted = df_cdh1_nodipvae.reindex(df_cdh1_nodipvae["corr"].abs().sort_values(ascending=False).index)

    df_vim_dipvae_sorted = df_vim_dipvae.reindex(df_vim_dipvae["corr"].abs().sort_values(ascending=False).index)
    df_cdh1_dipvae_sorted = df_cdh1_dipvae.reindex(df_cdh1_dipvae["corr"].abs().sort_values(ascending=False).index)

    # Plot VIM_transcriptomics
    plt.figure(figsize=(6, 4))

    # Create an index for each dataframe with the same length
    index = np.arange(len(df_vim_nodipvae_sorted))

    # Plot the No DipVAE data
    plt.plot(index, df_vim_nodipvae_sorted['corr'].abs(), label= "Without Disentanglement", marker='o', markersize = 3)

    # Plot the DipVAE data using the same index
    plt.plot(index, df_vim_dipvae_sorted['corr'].abs(), label= "With Disentanglement", marker='o', markersize = 3)

    plt.ylabel("|Pearson Correlation|", fontsize=7)
    plt.xticks([], [])  # Removing x-axis labels
    plt.xlabel("Latent Dimension", fontsize=7)      
    plt.legend(fontsize=7)
    plt.show()

    # Plot CDH1_transcriptomics
    plt.figure(figsize=(6, 4))

    # Create an index for each dataframe with the same length
    index = np.arange(len(df_cdh1_nodipvae_sorted))

    # Plot the No DipVAE data
    plt.plot(index, df_cdh1_nodipvae_sorted['corr'].abs(), label= "Without Disentanglement", marker='o', markersize = 3)

    # Plot the DipVAE data using the same index
    plt.plot(index, df_cdh1_dipvae_sorted['corr'].abs(), label= "With Disentanglement", marker='o', markersize = 3)

    plt.ylabel("|Pearson Correlation|", fontsize=7)
    plt.xticks([], [])  # Removing x-axis labels
    plt.xlabel("Latent Dimension", fontsize=7)      
    plt.legend(fontsize=7)
    plt.show()