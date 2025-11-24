# ============================================================================
# Multi-Threshold FDR Scatter Plot Grid
# Copy this entire cell into your Jupyter notebook after the plot_scatter function
# ============================================================================

# Define thresholds to test
thresholds = [1e-5, 1e-4, 1e-3]
n_pairs_per_threshold = 3

# Calculate grid dimensions
n_rows = len(thresholds)
n_cols = n_pairs_per_threshold

# Create figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

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
    filtered_data = df_res_vae_annot_filtered_cap_top[
        df_res_vae_annot_filtered_cap_top["fdr_vae"] < threshold
    ]

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
        plot_scatter(
            row.y_id,
            row.x_id,
            measured_groups=measured_groups,
            model_list_df=ss_cmp,
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

# Adjust layout to prevent overlap
plt.tight_layout()

# Print summary
print(f"\n{'='*80}")
print(
    f"Created figure with {n_rows} rows × {n_cols} columns = {n_rows * n_cols} total plots"
)
print(f"{'='*80}\n")

# Create and display summary DataFrame
summary_df = pd.DataFrame(summary_stats)
print("Summary of plotted pairs:")
print(summary_df.to_string(index=False))

# Optionally save the figure
# fig.savefig('multi_threshold_scatter_plots.png', dpi=300, bbox_inches='tight')

plt.show()
