import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.preprocessing
from scipy.stats import entropy


def meanabs_groupby_feature_target(df):
    melted_df = pd.melt(
        df,
        id_vars=["target_name", "Sample ID"],
        var_name="Features",
        value_name="Shap values",
    )
    melted_df["Shap values"] = melted_df["Shap values"].abs()
    final_df = (
        melted_df.drop(columns=["Sample ID"])
        .groupby(["Features", "target_name"])
        .mean()
        .rename(columns=lambda x: x.replace("Shap values", "Mean Abs Shap Values"))
    )

    return final_df


def top_x_extreme_samples(
    x, explanation, features, explain_target_idx, explain_target, sample_IDs
):
    extreme_samples_and_shap = {}
    extreme_samples = []

    for feature_value in features:
        shap_values = explanation[:, [feature_value], explain_target_idx].values
        shap_and_sampleID = np.hstack((shap_values, sample_IDs.reshape(-1, 1)))
        indices = np.argsort(-np.abs(shap_and_sampleID[:, 0].astype(float)))[:x]
        extreme_samples_and_shap[(feature_value, f"{explain_target}")] = (
            shap_and_sampleID[indices]
        )
        extreme_samples.append(shap_and_sampleID[indices][:, 1])

    return extreme_samples_and_shap, extreme_samples


def top_x_features(shap_values_df, x, from_dataset=None):
    mean_shap_per_feature = shap_values_df.drop(
        columns=["target_name", "Sample ID"]
    ).max()

    if from_dataset is not None:
        top_x_features = mean_shap_per_feature[
            mean_shap_per_feature.index.str.contains(from_dataset)
        ].nlargest(x)

    else:
        top_x_features = mean_shap_per_feature.nlargest(x)

    return top_x_features


def top_x_df_entries(df, x):
    # Create a copy of the dataframe
    df_copy = df.copy()
    df_copy = df_copy.drop(columns=["target_name", "Sample ID"])
    # Convert DataFrame to numpy array
    values = df_copy.values

    top_indices = []

    for _ in range(x):
        # Find the index of the maximum value
        max_idx = np.unravel_index(np.argmax(values), values.shape)
        top_indices.append(max_idx)
        # Set the maximum value to a very small number so it will not be picked again
        values[max_idx] = -np.inf

    # Convert indices to row and column names
    top_entries = [
        (df_copy.index[row], df_copy.columns[col]) for row, col in top_indices
    ]

    return top_entries


def heatmap_features_vs_latents(x, absmean_shap_lat): 
    top_x_entries = absmean_shap_lat.sort_values(by='Mean Abs Shap Values', ascending=False).head(x) 
    features = top_x_entries.reset_index()["Features"].unique()

    # Table with the shap values for the selected features and for all latent dimensions
    table_data = absmean_shap_lat.reset_index()
    table_data = table_data[table_data['Features'].isin(features)]

    heatmap_data = table_data.pivot(index='Features', columns='target_name', values='Mean Abs Shap Values')
    latent_columns = sorted([col for col in heatmap_data.columns if col.startswith('Latent_')], 
                        key=lambda x: int(x.split('_')[1]))
    heatmap_data_ordered = heatmap_data[latent_columns]

    plt.figure(figsize=(5, 4))
    heatmap = sns.heatmap(heatmap_data_ordered, cmap="YlGnBu", annot=False)  # cmap is the color scheme
    plt.xlabel('Latent Dimensions', fontsize =11)
    plt.ylabel('Features', fontsize =11)
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks([])
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Avg |Shap Value|", fontsize=10, rotation=270, labelpad=20)
    plt.show()
