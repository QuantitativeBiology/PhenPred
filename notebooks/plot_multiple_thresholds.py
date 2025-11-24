# Improved code for plotting multiple FDR thresholds in a grid

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_multi_threshold_grid(
    df_results,
    thresholds,
    n_pairs_per_threshold,
    plot_scatter_func,
    measured_groups=None,
    model_list_df=None,
    figsize_per_plot=(6, 5),
):
    """
    Create a comprehensive multi-panel figure for different FDR thresholds.

    Parameters:
    -----------
    df_results : pd.DataFrame
        Dataframe containing the results with 'fdr_vae', 'y_id', 'x_id' columns
    thresholds : list
        List of FDR thresholds to plot (e.g., [1e-5, 1e-4, 1e-3])
    n_pairs_per_threshold : int
        Number of gene pairs to plot per threshold
    plot_scatter_func : callable
        Function to plot scatter (should accept ax parameter)
    measured_groups : dict, optional
        Groups for categorizing measurements
    model_list_df : pd.DataFrame, optional
        Model metadata
    figsize_per_plot : tuple
        Size of each individual subplot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axes : numpy.ndarray
        Array of axes objects
    """
    # Calculate grid dimensions
    n_rows = len(thresholds)
    n_cols = n_pairs_per_threshold

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )

    # Ensure axes is always 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Track statistics for summary
    summary_stats = []

    # Iterate through thresholds and plot pairs
    for row_idx, threshold in enumerate(thresholds):
        # Filter data for current threshold
        filtered_data = df_results[df_results["fdr_vae"] < threshold]

        print(f"Threshold FDR < {threshold:.0e}: {len(filtered_data)} total pairs")

        # Get top pairs (using tail to get the last n pairs)
        top_pairs = filtered_data.tail(n_pairs_per_threshold)

        # Check if we have enough pairs
        if len(top_pairs) < n_pairs_per_threshold:
            print(
                f"  Warning: Only {len(top_pairs)} pairs available for threshold {threshold:.0e}"
            )

        # Plot each pair
        for col_idx, row in enumerate(top_pairs.itertuples()):
            ax = axes[row_idx, col_idx]

            # Plot scatter
            plot_scatter_func(
                row.y_id,
                row.x_id,
                measured_groups=measured_groups,
                model_list_df=model_list_df,
                ax=ax,
            )

            # Modify title to include threshold info
            current_title = ax.get_title()
            ax.set_title(f"{current_title}\nFDR < {threshold:.0e}", fontsize=9, pad=5)

            # Add row label on leftmost plots
            if col_idx == 0:
                # Add a text label outside the plot on the left
                ax.text(
                    -0.20,
                    0.5,
                    f"FDR < {threshold:.0e}",
                    transform=ax.transAxes,
                    rotation=90,
                    verticalalignment="center",
                    fontsize=12,
                    fontweight="bold",
                )

            # Store stats
            summary_stats.append(
                {
                    "threshold": threshold,
                    "pair_rank": col_idx + 1,
                    "y_gene": row.y_id,
                    "x_gene": row.x_id,
                    "fdr_vae": row.fdr_vae if hasattr(row, "fdr_vae") else None,
                    "beta_vae": row.beta_vae if hasattr(row, "beta_vae") else None,
                }
            )

    # Clear any unused axes (if we don't have enough pairs)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if not ax.has_data():
                ax.set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Print summary
    print(f"\n{'='*80}")
    print(
        f"Created figure with {n_rows} rows × {n_cols} columns = {n_rows * n_cols} total plots"
    )
    print(f"{'='*80}\n")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)

    return fig, axes, summary_df


# Example usage code (to be run in notebook):
"""
# Define thresholds to test
thresholds = [1e-5, 1e-4, 1e-3]
n_pairs_per_threshold = 3

# Create the multi-panel figure
fig, axes, summary = plot_multi_threshold_grid(
    df_results=df_res_vae_annot_filtered_cap_top,
    thresholds=thresholds,
    n_pairs_per_threshold=n_pairs_per_threshold,
    plot_scatter_func=plot_scatter,
    measured_groups=measured_groups,
    model_list_df=ss_cmp,
    figsize_per_plot=(5.5, 4.5)
)

# Display summary
print("Summary of plotted pairs:")
print(summary.to_string(index=False))

# Save figure if desired
# fig.savefig('multi_threshold_scatter_plots.png', dpi=300, bbox_inches='tight')

plt.show()
"""
