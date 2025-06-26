import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import os


def calculate_principled_weights(fdr_values, min_weight=1.0, max_weight=5.0, power=2.0):
    """
    Calculate weights from FDR values with clear parameterization.

    Args:
        fdr_values: Array of FDR p-values
        min_weight: Minimum weight for least significant pairs (default=1.0)
        max_weight: Maximum weight for most significant pairs (default=5.0)
        power: Exponent > 1 to favor top pairs more strongly (default=2.0)

    Returns:
        Array of weights where lower FDR gets higher weight
    """
    # Convert to -log10 scale (higher values = more significant)
    log_weights = -np.log10(np.maximum(fdr_values, 1e-100))  # Avoid log(0)

    # Normalize to [0, 1]
    if log_weights.max() != log_weights.min():
        normalized = (log_weights - log_weights.min()) / (
            log_weights.max() - log_weights.min()
        )
    else:
        normalized = np.ones_like(log_weights)

    # Apply power transformation (power > 1 to strongly favor top pairs)
    powered = np.power(normalized, power)

    # Scale to desired range
    weights = min_weight + (max_weight - min_weight) * powered

    return weights


def select_validation_cell_lines(
    df_res_vae_annot_filtered_cap_top,
    gexp_df,
    cas9_df,
    num_cell_lines=6,
    score_threshold_percentile=75,
    n_jobs=20,
    crispr_threshold_percentile=50,
    target_expression_threshold_percentile=50,
    biomarker_expression_threshold_percentile=50,
    create_visualizations=False,
    visualize_top_n=10,
    output_dir="./validation_results",
    tissue_map=None,
):
    """
    Integrated pipeline to select optimal cell lines for synthetic lethality validation.
    Now enforces three percentile-based thresholds: CRISPR dependency, target gene expression, and biomarker gene expression.

    Parameters:
    - df_res_vae_annot_filtered_cap_top: DataFrame containing gene pairs and significance measurements
    - gexp_df: DataFrame containing gene expression data
    - cas9_df: DataFrame containing CRISPR scores
    - num_cell_lines: Number of cell lines to select
    - threshold_percentile: Percentile threshold for considering a cell line adequate
    - n_jobs: Number of parallel jobs for score calculation
    - crispr_threshold_percentile: Percentile threshold for CRISPR dependency (default=50)
    - target_expression_threshold_percentile: Percentile threshold for target gene expression (default=50)
    - biomarker_expression_threshold_percentile: Percentile threshold for biomarker gene expression (default=50)
    - create_visualizations: Whether to create visualizations
    - visualize_top_n: Number of top gene pairs to visualize per cell line
    - output_dir: Directory to save visualizations
    - tissue_map: Dictionary mapping cell line IDs to tissue types (default=None)

    Returns:
    - Dictionary with selected cell lines and evaluation metrics
    """
    print(
        "Starting optimized cell line selection pipeline with 3-criteria percentile-based filtering..."
    )
    overall_start_time = time.time()

    print("Using gene-specific percentile-based thresholds...")
    print(f"Threshold percentiles:")
    print(
        f"  CRISPR threshold: {crispr_threshold_percentile}th percentile per target gene"
    )
    print(
        f"  Target expression threshold: {target_expression_threshold_percentile}th percentile per target gene"
    )
    print(
        f"  Biomarker expression threshold: {biomarker_expression_threshold_percentile}th percentile per biomarker gene"
    )

    # Step 1: Extract gene pairs and weights from the filtered results dataframe
    print("Extracting gene pairs and weights...")
    gene_pairs = list(
        zip(
            df_res_vae_annot_filtered_cap_top["y_id"],
            df_res_vae_annot_filtered_cap_top["x_id"],
        )
    )

    # Create weights based on fdr_vae (lower FDR = higher weight) using principled approach
    fdr_values = df_res_vae_annot_filtered_cap_top["fdr_vae"].values
    weights_array = calculate_principled_weights(
        fdr_values, min_weight=1.0, max_weight=5.0, power=2.0
    )

    weights = pd.Series(
        weights_array,
        index=pd.MultiIndex.from_tuples(
            gene_pairs, names=["target_gene", "biomarker_gene"]
        ),
    )

    # Step 2: Calculate validation scores using parallel processing with gene-specific thresholds
    print(
        f"Calculating validation scores with gene-specific percentile-based criteria..."
    )
    validation_scores = calculate_validation_scores(
        gene_pairs,
        gexp_df,
        cas9_df,
        df_res_vae_annot_filtered_cap_top,
        n_jobs=n_jobs,
        crispr_threshold_percentile=crispr_threshold_percentile,
        target_expression_threshold_percentile=target_expression_threshold_percentile,
        biomarker_expression_threshold_percentile=biomarker_expression_threshold_percentile,
    )

    # Check if any valid scores remain after filtering
    if validation_scores.sum().sum() == 0:
        print(
            f"ERROR: No gene pairs have cell lines that pass gene-specific percentile thresholds"
        )
        print(
            f"CRISPR > {crispr_threshold_percentile}th percentile per target gene AND "
            f"Target Expression > {target_expression_threshold_percentile}th percentile per target gene AND "
            f"Biomarker Expression > {biomarker_expression_threshold_percentile}th percentile per biomarker gene"
        )
        print("Please try lower percentile thresholds.")
        return None

    # Step 3: Create binary coverage matrix
    print("Creating coverage matrix using threshold...")
    coverage_matrix = create_coverage_matrix(
        validation_scores, score_threshold_percentile
    )

    # Step 4: Select optimal cell lines with optimized algorithm
    print("Selecting optimal cell lines using optimized algorithm...")
    selection_results = select_optimal_cell_lines(
        coverage_matrix, weights, num_cell_lines
    )

    # Step 5: Evaluate the selection
    print("Evaluating cell line selection...")
    evaluation_results = evaluate_cell_line_selection(
        coverage_matrix,
        selection_results["selected_cell_lines"],
        weights,
        df_res_vae_annot_filtered_cap_top,
    )

    # Combine results
    results = {
        "selected_cell_lines": selection_results["selected_cell_lines"],
        "cell_line_coverage": selection_results["pairs_covered_per_cell"],
        "validation_scores": validation_scores,
        "coverage_matrix": coverage_matrix,
        "evaluation": evaluation_results,
        "gene_pair_examples": selection_results["gene_pair_examples"],
        "weights": weights,  # Add weights for analysis
        "thresholds": "gene_specific_percentiles",
        "threshold_percentiles": {
            "crispr": crispr_threshold_percentile,
            "target_expression": target_expression_threshold_percentile,
            "biomarker_expression": biomarker_expression_threshold_percentile,
        },
    }

    overall_elapsed_time = time.time() - overall_start_time
    print(f"\nTotal pipeline execution time: {overall_elapsed_time:.2f} seconds")

    # Print a summary
    print("\n----- RESULTS SUMMARY -----")
    print(f"\nRESULTS WITH GENE-SPECIFIC PERCENTILE THRESHOLDS:")
    print(f"CRISPR > {crispr_threshold_percentile}th percentile per target gene AND")
    print(
        f"Target Expression > {target_expression_threshold_percentile}th percentile per target gene AND"
    )
    print(
        f"Biomarker Expression > {biomarker_expression_threshold_percentile}th percentile per biomarker gene"
    )
    print("\nSelected Cell Lines:")
    for i, cell_line in enumerate(results["selected_cell_lines"]):
        num_pairs = len(selection_results["covered_by_cell_line"][i])
        # Get tissue information if available
        tissue_info = ""
        if tissue_map and cell_line in tissue_map:
            tissue_info = f" ({tissue_map[cell_line]})"
        print(
            f"{i+1}. {cell_line}{tissue_info} - Covers {num_pairs} gene pairs "
            + f"({num_pairs/len(gene_pairs)*100:.2f}% of total)"
        )

    print(f"\nOverall Coverage: {evaluation_results['overall_coverage_percent']:.2f}%")
    print(f"Top Pair Coverage: {evaluation_results['top_pair_coverage_percent']:.2f}%")
    print(f"Weighted Coverage: {evaluation_results['weighted_coverage_percent']:.2f}%")

    # Create visualizations if requested
    if create_visualizations:
        visualize_results(
            results,
            df_res_vae_annot_filtered_cap_top,
            gexp_df,
            cas9_df,
            output_dir,
            visualize_top_n,
        )

    return results


def calculate_validation_scores(
    gene_pairs,
    gexp_df,
    cas9_df,
    df_res_vae_annot_filtered_cap_top,
    n_jobs=20,
    crispr_threshold_percentile=50,
    target_expression_threshold_percentile=50,
    biomarker_expression_threshold_percentile=50,
):
    """
    Calculate synthetic lethality validation scores for all gene pairs across all cell lines.
    Apply three gene-specific percentile thresholds: CRISPR dependency, target gene expression, and biomarker gene expression.

    Parameters:
    - gene_pairs: List of tuples where each tuple is (target_gene, biomarker_gene)
    - gexp_df: DataFrame containing gene expression data, with genes as columns
    - cas9_df: DataFrame containing CRISPR scores, with genes as columns
    - df_res_vae_annot_filtered_cap_top: DataFrame containing gene pairs and their diff_log10fdr values
    - n_jobs: Number of parallel jobs to run (default=20)
    - crispr_threshold_percentile: Percentile threshold for CRISPR dependency per target gene (default=50)
    - target_expression_threshold_percentile: Percentile threshold for target gene expression per target gene (default=50)
    - biomarker_expression_threshold_percentile: Percentile threshold for biomarker gene expression per biomarker gene (default=50)

    Returns:
    - validation_score_matrix: DataFrame with gene pairs as rows and cell lines as columns
    """
    # Ensure we're working with cell lines common to both datasets
    common_cell_lines = gexp_df.index.intersection(cas9_df.index)
    print(
        f"Found {len(common_cell_lines)} cell lines common to both expression and CRISPR datasets"
    )

    # Initialize validation score matrix with float dtype to avoid dtype warnings
    validation_scores = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            gene_pairs, names=["target_gene", "biomarker_gene"]
        ),
        columns=common_cell_lines,
        dtype=float,
    )

    # Create a lookup dictionary for fdr_vae values (converted to weights)
    # Use consistent weight calculation approach
    fdr_values_for_pairs = []
    pair_list = []
    for _, row in df_res_vae_annot_filtered_cap_top.iterrows():
        fdr_values_for_pairs.append(row["fdr_vae"])
        pair_list.append((row["y_id"], row["x_id"]))

    # Calculate weights using the same principled approach (slightly smaller range for per-pair weighting)
    pair_weights = calculate_principled_weights(
        np.array(fdr_values_for_pairs), min_weight=1.0, max_weight=3.0, power=2.0
    )

    # Create lookup dictionary
    fdr_weight_dict = dict(zip(pair_list, pair_weights))

    # Function to process a single gene pair
    def process_gene_pair(pair_idx, target_gene, biomarker_gene):
        # Check if genes exist in respective datasets
        if (
            target_gene not in cas9_df.columns
            or target_gene not in gexp_df.columns
            or biomarker_gene not in gexp_df.columns
        ):
            return pair_idx, None

        # Get CRISPR and expression values for all cell lines
        cas9_values = cas9_df[target_gene].reindex(common_cell_lines)
        target_gexp_values = gexp_df[target_gene].reindex(common_cell_lines)
        biomarker_gexp_values = gexp_df[biomarker_gene].reindex(common_cell_lines)

        # Handle missing values
        valid_cells = (
            cas9_values.notna()
            & target_gexp_values.notna()
            & biomarker_gexp_values.notna()
        )
        if valid_cells.sum() == 0:
            return pair_idx, None

        cas9_values = cas9_values[valid_cells]
        target_gexp_values = target_gexp_values[valid_cells]
        biomarker_gexp_values = biomarker_gexp_values[valid_cells]

        # Calculate gene-specific thresholds based on percentiles
        # 1. CRISPR threshold: percentile of all CRISPR values for this target gene
        crispr_threshold = np.percentile(cas9_values, crispr_threshold_percentile)
        # 2. Target gene expression threshold: percentile of all expression values for this target gene
        target_expression_threshold = np.percentile(
            target_gexp_values, target_expression_threshold_percentile
        )
        # 3. Biomarker gene expression threshold: percentile of all expression values for this biomarker gene
        biomarker_expression_threshold = np.percentile(
            biomarker_gexp_values, biomarker_expression_threshold_percentile
        )

        # Apply three gene-specific threshold filters - cell line must pass ALL three criteria
        # 1. CRISPR threshold: CRISPR score > gene-specific percentile (dependency)
        # 2. Target gene expression threshold: Expression value > gene-specific percentile
        # 3. Biomarker gene expression threshold: Expression value > gene-specific percentile
        crispr_dependency_mask = cas9_values > crispr_threshold
        target_high_expression_mask = target_gexp_values > target_expression_threshold
        biomarker_high_expression_mask = (
            biomarker_gexp_values > biomarker_expression_threshold
        )

        # Combined mask: cell line must pass all three filters
        combined_mask = (
            crispr_dependency_mask
            & target_high_expression_mask
            & biomarker_high_expression_mask
        )

        # If no cell lines pass all three thresholds, return None
        if combined_mask.sum() == 0:
            return pair_idx, None

        # Filter values to only include cell lines that pass all three thresholds
        cas9_filtered = cas9_values[combined_mask]
        target_gexp_filtered = target_gexp_values[combined_mask]
        biomarker_gexp_filtered = biomarker_gexp_values[combined_mask]

        # Calculate 3D distance from the "ideal" corner (high CRISPR, high target expression, high biomarker expression)
        # First identify the most extreme values to define our reference point
        max_cas9 = cas9_filtered.max()
        max_target_gexp = target_gexp_filtered.max()
        max_biomarker_gexp = biomarker_gexp_filtered.max()

        # Calculate ranges for normalization
        cas9_range = max_cas9 - cas9_filtered.min()
        target_gexp_range = max_target_gexp - target_gexp_filtered.min()
        biomarker_gexp_range = max_biomarker_gexp - biomarker_gexp_filtered.min()

        if cas9_range == 0 or target_gexp_range == 0 or biomarker_gexp_range == 0:
            return pair_idx, None

        # Calculate normalized 3D distances to the ideal point
        # Smaller distance = closer to ideal = better synthetic lethality
        normalized_distances = np.sqrt(
            ((max_cas9 - cas9_filtered) / cas9_range) ** 2
            + ((max_target_gexp - target_gexp_filtered) / target_gexp_range) ** 2
            + ((max_biomarker_gexp - biomarker_gexp_filtered) / biomarker_gexp_range)
            ** 2
        )

        # Convert distances to scores where higher is better (closer to ideal point)
        # Normalize by sqrt(3) since we have 3 dimensions
        sl_scores = 1 - (normalized_distances / np.sqrt(3))

        # Apply the fdr_vae weight to prioritize gene pairs with lower FDR (higher significance)
        fdr_weight = fdr_weight_dict.get((target_gene, biomarker_gene), 1.0)
        weighted_scores = sl_scores * fdr_weight

        # Create a result Series with all cell lines, setting scores to 0 for those that don't
        # meet all three thresholds
        result_scores = pd.Series(0.0, index=cas9_values.index, dtype=float)
        result_scores.loc[weighted_scores.index] = weighted_scores

        return pair_idx, (result_scores.index, result_scores.values)

    # Use joblib to parallelize the computation with reduced verbosity
    print(f"Processing {len(gene_pairs)} gene pairs with {n_jobs} parallel jobs...")
    print(
        f"Applying gene-specific thresholds: CRISPR > {crispr_threshold_percentile}th percentile per target gene AND Target Expression > {target_expression_threshold_percentile}th percentile per target gene AND Biomarker Expression > {biomarker_expression_threshold_percentile}th percentile per biomarker gene"
    )
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_gene_pair)(i, target_gene, biomarker_gene)
        for i, (target_gene, biomarker_gene) in enumerate(gene_pairs)
    )

    # Process results and populate the validation scores dataframe
    filtered_pairs_count = 0
    for pair_idx, result in results:
        if result is not None:
            indices, values = result
            target_gene, biomarker_gene = gene_pairs[pair_idx]
            validation_scores.loc[(target_gene, biomarker_gene), indices] = values
            if (
                values.sum() > 0
            ):  # If at least one cell line passed all three thresholds
                filtered_pairs_count += 1

    elapsed_time = time.time() - start_time
    print(f"Score calculation completed in {elapsed_time:.2f} seconds")
    print(
        f"{filtered_pairs_count} out of {len(gene_pairs)} gene pairs have at least one cell line"
    )
    print(
        f"that passes all three gene-specific thresholds (CRISPR > {crispr_threshold_percentile}th percentile per target gene AND Target Expression > {target_expression_threshold_percentile}th percentile per target gene AND Biomarker Expression > {biomarker_expression_threshold_percentile}th percentile per biomarker gene)"
    )

    # Fill any missing values with zeros
    return validation_scores.fillna(0)


def create_coverage_matrix(validation_scores, score_threshold_percentile=75):
    """
    Convert validation scores to a binary coverage matrix using a threshold.

    Parameters:
    - validation_scores: DataFrame with gene pairs as rows and cell lines as columns
    - score_threshold_percentile: Percentile value to use as threshold (default: 75th percentile)

    Returns:
    - Binary coverage matrix where 1 indicates the cell line adequately demonstrates
      the synthetic lethality for that gene pair
    """
    # Calculate threshold based on percentile of non-zero values
    non_zero_scores = validation_scores.values.flatten()
    non_zero_scores = non_zero_scores[non_zero_scores > 0]

    if len(non_zero_scores) == 0:
        threshold = 0
    else:
        threshold = np.percentile(non_zero_scores, score_threshold_percentile)

    print(
        f"Using threshold score of {threshold} (based on {score_threshold_percentile}th percentile)"
    )

    # Create binary coverage matrix
    coverage_matrix = (validation_scores >= threshold).astype(int)

    # Print coverage statistics
    total_pairs = coverage_matrix.shape[0]
    pairs_covered = (coverage_matrix.sum(axis=1) > 0).sum()
    print(
        f"Coverage matrix: {pairs_covered} of {total_pairs} pairs ({pairs_covered/total_pairs*100:.2f}%) "
        + f"are covered by at least one cell line"
    )

    return coverage_matrix


def select_optimal_cell_lines(coverage_matrix, weights, num_cell_lines=6):
    """
    Select the optimal set of cell lines using a weighted greedy algorithm.
    Optimized version with vectorized operations and progress reporting.

    Parameters:
    - coverage_matrix: Binary matrix where rows=gene pairs, columns=cell lines
    - weights: Series of importance weights for each gene pair
    - num_cell_lines: Number of cell lines to select

    Returns:
    - Dictionary with selected cell lines and coverage statistics
    """
    print("Starting optimized cell line selection...")
    start_time = time.time()

    # Make sure weights align with coverage_matrix index
    if weights is not None:
        weights = weights.reindex(coverage_matrix.index).fillna(0)
    else:
        weights = pd.Series(1.0, index=coverage_matrix.index)

    # Convert to numpy arrays for faster calculation
    coverage_array = coverage_matrix.values
    weights_array = weights.values
    cell_lines = coverage_matrix.columns.tolist()

    # Initialize variables
    selected_cell_lines = []
    selected_indices = []
    remaining_pairs = set(range(len(coverage_matrix)))

    # Track which pairs are covered by each selected cell line
    covered_by_cell_line = {}
    gene_pair_examples = {}

    # Pre-compute pair scores for each gene pair
    pair_validation_scores = {}
    for pair_idx in range(len(coverage_matrix)):
        target_gene, biomarker_gene = coverage_matrix.index[pair_idx]
        # Store index lookup for faster access later
        pair_validation_scores[pair_idx] = {
            cell_idx: coverage_matrix.iloc[pair_idx, cell_idx]
            for cell_idx in range(len(cell_lines))
            if coverage_array[pair_idx, cell_idx] == 1
        }

    # Main selection loop with progress bar
    pbar = tqdm(total=min(num_cell_lines, len(cell_lines)), desc="Selecting cell lines")

    for i in range(num_cell_lines):
        if not remaining_pairs:
            pbar.update(num_cell_lines - i)
            pbar.close()
            print(f"All gene pairs are covered. Only needed {i} cell lines.")
            break

        # Calculate weighted coverage for each cell line
        weighted_coverage = {}
        cell_line_pairs = {}

        # This is the bottleneck - optimize by using vectorized operations
        for cell_idx, cell_line in enumerate(cell_lines):
            if cell_idx in selected_indices:
                continue

            # Vectorized operation: multiply coverage by weights and sum
            cell_coverage = coverage_array[:, cell_idx]
            covered_indices = np.where(cell_coverage == 1)[0]
            covered_indices = [idx for idx in covered_indices if idx in remaining_pairs]

            if not covered_indices:
                weighted_coverage[cell_idx] = 0
                cell_line_pairs[cell_idx] = (set(), [])
                continue

            weighted_sum = sum(weights_array[idx] for idx in covered_indices)

            weighted_coverage[cell_idx] = weighted_sum

            # Store pairs covered by this cell line with their scores
            newly_covered_pairs = set(covered_indices)

            # Get pair details for examples, but limit to avoid excessive processing
            pair_scores = []
            for pair_idx in list(newly_covered_pairs)[
                :50
            ]:  # Limit to 50 for efficiency
                target_gene, biomarker_gene = coverage_matrix.index[pair_idx]
                # Use pre-computed validation scores
                validation_score = pair_validation_scores[pair_idx].get(cell_idx, 0)
                pair_weight = weights_array[pair_idx]
                pair_scores.append(
                    (target_gene, biomarker_gene, validation_score * pair_weight)
                )

            cell_line_pairs[cell_idx] = (newly_covered_pairs, pair_scores)

        # Select the cell line with highest coverage
        if not weighted_coverage or all(
            score == 0 for score in weighted_coverage.values()
        ):
            pbar.update(num_cell_lines - i)
            pbar.close()
            print(
                f"No more cell lines can cover remaining pairs. Stopping at {i} cell lines."
            )
            break

        best_cell_idx = max(weighted_coverage.items(), key=lambda x: x[1])[0]
        best_cell_line = cell_lines[best_cell_idx]

        selected_cell_lines.append(best_cell_line)
        selected_indices.append(best_cell_idx)

        # Store the covered pairs and example gene pairs
        newly_covered, pair_scores = cell_line_pairs[best_cell_idx]
        covered_by_cell_line[i] = newly_covered

        # Sort pair_scores by the validation score (higher is better)
        pair_scores.sort(key=lambda x: x[2], reverse=True)
        gene_pair_examples[i] = pair_scores

        # Update the remaining pairs
        remaining_pairs -= newly_covered

        # Log progress
        pbar.update(1)
        pbar.set_postfix({"covered": len(newly_covered)})

    pbar.close()

    # Calculate pairs covered per cell line
    pairs_covered_per_cell = {
        cell_line: len(covered_by_cell_line[i])
        for i, cell_line in enumerate(selected_cell_lines)
        if i in covered_by_cell_line
    }

    # Calculate coverage statistics
    total_pairs = len(coverage_matrix)
    all_covered_pairs = set()
    for covered_set in covered_by_cell_line.values():
        all_covered_pairs.update(covered_set)

    elapsed_time = time.time() - start_time
    print(f"Cell line selection completed in {elapsed_time:.2f} seconds")
    print(
        f"Selected {len(selected_cell_lines)} cell lines that collectively cover "
        + f"{len(all_covered_pairs)} of {total_pairs} gene pairs ({len(all_covered_pairs)/total_pairs*100:.2f}%)"
    )

    coverage_stats = {
        "selected_cell_lines": selected_cell_lines,
        "total_coverage_percent": len(all_covered_pairs) / total_pairs * 100,
        "covered_by_cell_line": covered_by_cell_line,
        "pairs_covered_per_cell": pairs_covered_per_cell,
        "gene_pair_examples": gene_pair_examples,
    }

    return coverage_stats


def evaluate_cell_line_selection(
    coverage_matrix,
    selected_cell_lines,
    weights,
    df_res_vae_annot_filtered_cap_top,
    top_n=100,
):
    """
    Evaluate the quality of the selected cell lines.

    Parameters:
    - coverage_matrix: Binary coverage matrix
    - selected_cell_lines: List of selected cell line names/indices
    - weights: Series of importance weights for each gene pair
    - df_res_vae_annot_filtered_cap_top: Original dataframe with gene pairs and significance metrics
    - top_n: Number of top gene pairs to consider for high-priority coverage

    Returns:
    - Dictionary with evaluation metrics
    """
    # Make sure weights align with coverage_matrix index
    if weights is not None:
        weights = weights.reindex(coverage_matrix.index).fillna(0)
    else:
        weights = pd.Series(1.0, index=coverage_matrix.index)

    # Get indices of top gene pairs by weight (most significant deltaFDR)
    top_pairs_idx = weights.nlargest(min(top_n, len(weights))).index

    # Calculate overall coverage
    covered_pairs = set()
    for cell_line in selected_cell_lines:
        cell_idx = coverage_matrix.columns.get_loc(cell_line)
        for pair_idx in range(len(coverage_matrix)):
            if coverage_matrix.iloc[pair_idx, cell_idx] == 1:
                covered_pairs.add(pair_idx)

    # Calculate coverage of top pairs
    top_pair_indices = [coverage_matrix.index.get_loc(pair) for pair in top_pairs_idx]
    top_pairs_covered = sum(1 for idx in top_pair_indices if idx in covered_pairs)

    # Find which top pairs are not covered
    uncovered_top_pairs = [
        coverage_matrix.index[idx]
        for idx in top_pair_indices
        if idx not in covered_pairs
    ]

    # Calculate weighted coverage
    weighted_coverage = sum(weights.iloc[pair_idx] for pair_idx in covered_pairs)

    # Calculate maximum possible weighted coverage
    max_weighted_coverage = weights.sum()

    # Store significant FDR values for covered and uncovered top pairs
    significant_pairs_covered = []
    for pair in top_pairs_idx:
        if coverage_matrix.index.get_loc(pair) in covered_pairs:
            target_gene, biomarker_gene = pair
            row = df_res_vae_annot_filtered_cap_top[
                (df_res_vae_annot_filtered_cap_top["y_id"] == target_gene)
                & (df_res_vae_annot_filtered_cap_top["x_id"] == biomarker_gene)
            ]
            if not row.empty:
                significant_pairs_covered.append(
                    (
                        target_gene,
                        biomarker_gene,
                        row["fdr_vae"].values[0],  # Use fdr_vae instead
                        row["log10fdr_vae"].values[0],
                    )
                )

    # Sort by significance (lower FDR = more significant)
    significant_pairs_covered.sort(
        key=lambda x: x[2]
    )  # Sort ascending (lower FDR first)

    return {
        "overall_coverage_percent": len(covered_pairs) / len(coverage_matrix) * 100,
        "top_pair_coverage_percent": top_pairs_covered / len(top_pairs_idx) * 100,
        "weighted_coverage_percent": weighted_coverage / max_weighted_coverage * 100,
        "total_pairs_covered": len(covered_pairs),
        "total_pairs": len(coverage_matrix),
        "top_pairs_covered": top_pairs_covered,
        "top_pairs_total": len(top_pairs_idx),
        "uncovered_top_pairs": uncovered_top_pairs,
        "significant_pairs_covered": significant_pairs_covered,
    }


def visualize_results(
    results,
    df_res_vae_annot_filtered_cap_top,
    gexp_df,
    cas9_df,
    output_dir=".",
    visualize_top_n=10,
):
    """
    Create visualizations of the cell line selection results with 3-criteria analysis.

    Parameters:
    - results: Dictionary returned by select_validation_cell_lines
    - df_res_vae_annot_filtered_cap_top: DataFrame with gene pairs and significance metrics
    - gexp_df: Gene expression DataFrame
    - cas9_df: CRISPR scores DataFrame
    - output_dir: Directory to save visualizations
    - visualize_top_n: Number of top gene pairs to visualize per cell line (capped at 4 for layout)
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # 1. Selected cell lines coverage heatmap
    plt.figure(figsize=(12, 8))
    selected_scores = results["validation_scores"][results["selected_cell_lines"]]

    # If there are many gene pairs, sample a subset for clearer visualization
    if len(selected_scores) > 50:
        # Sample top pairs by FDR significance (lower FDR = more significant) and some random pairs
        fdr_values = df_res_vae_annot_filtered_cap_top["fdr_vae"].values
        vis_weights = pd.Series(
            -np.log10(np.maximum(fdr_values, 1e-10)),  # Convert FDR to -log10 scale
            index=pd.MultiIndex.from_tuples(
                zip(
                    df_res_vae_annot_filtered_cap_top["y_id"],
                    df_res_vae_annot_filtered_cap_top["x_id"],
                ),
                names=["target_gene", "biomarker_gene"],
            ),
        )
        top_pairs = vis_weights.nlargest(30).index  # Top 30 most significant pairs
        random_pairs = selected_scores.sample(min(20, len(selected_scores))).index
        plot_pairs = list(set(top_pairs).union(set(random_pairs)))
        selected_scores = selected_scores.loc[plot_pairs]

    # Create the heatmap
    ax = sns.heatmap(selected_scores, cmap="viridis", linewidths=0.5)
    plt.title("3-Criteria Validation Scores for Selected Cell Lines", fontsize=14)
    plt.ylabel("Gene Pairs (Target, Biomarker)", fontsize=12)
    plt.xlabel("Cell Lines", fontsize=12)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/selected_cell_lines_heatmap.png", dpi=300)
    plt.close()

    # 2. Bar chart of gene pairs covered by each cell line
    plt.figure(figsize=(10, 6))
    cell_coverage = pd.Series(results["cell_line_coverage"])
    cell_coverage.plot(kind="bar", color="skyblue")
    plt.title("Number of Gene Pairs Covered by Each Selected Cell Line", fontsize=14)
    plt.ylabel("Number of Gene Pairs", fontsize=12)
    plt.xlabel("Cell Line", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/cell_line_coverage_bar.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Enhanced gene pair visualization with 3 criteria for each cell line
    for cell_idx, cell_line in enumerate(
        results["selected_cell_lines"][:3]
    ):  # Plot first 3 cell lines
        if cell_idx not in results["gene_pair_examples"]:
            continue

        examples = results["gene_pair_examples"][cell_idx]
        if not examples:
            continue

        # Plot top gene pairs for this cell line with 3-panel visualization
        num_pairs_to_plot = min(
            visualize_top_n, len(examples), 4
        )  # Cap at 4 for layout reasons
        fig = plt.figure(figsize=(18, 3 * num_pairs_to_plot))

        for i, (target_gene, biomarker_gene, score) in enumerate(
            examples[:num_pairs_to_plot]
        ):
            if i >= num_pairs_to_plot:
                break

            # Get CRISPR and expression values
            cas9_values = cas9_df[target_gene]
            target_gexp_values = gexp_df[target_gene]
            biomarker_gexp_values = gexp_df[biomarker_gene]

            # Handle missing values
            valid_cells = (
                cas9_values.notna()
                & target_gexp_values.notna()
                & biomarker_gexp_values.notna()
            )
            cas9_values = cas9_values[valid_cells]
            target_gexp_values = target_gexp_values[valid_cells]
            biomarker_gexp_values = biomarker_gexp_values[valid_cells]

            # Create 3 subplots for each gene pair (3 pairwise comparisons)
            base_idx = i * 3

            # Panel 1: CRISPR vs Target Expression
            ax1 = plt.subplot(num_pairs_to_plot, 3, base_idx + 1)
            ax1.scatter(target_gexp_values, cas9_values, alpha=0.5, c="lightgray")

            if cell_line in target_gexp_values.index and cell_line in cas9_values.index:
                highlighted_x = target_gexp_values[cell_line]
                highlighted_y = cas9_values[cell_line]
                ax1.scatter(
                    [highlighted_x],
                    [highlighted_y],
                    c="red",
                    s=100,
                    edgecolor="black",
                    zorder=10,
                )

            corr1 = np.corrcoef(target_gexp_values, cas9_values)[0, 1]
            ax1.set_title(f"{target_gene}\nCRISPR vs Target Expr (r={corr1:.3f})")
            ax1.set_xlabel(f"{target_gene} Expression")
            ax1.set_ylabel(f"{target_gene} CRISPR Score")

            # Add gene-specific threshold lines
            crispr_threshold = np.percentile(
                cas9_values, results["threshold_percentiles"]["crispr"]
            )
            target_threshold = np.percentile(
                target_gexp_values,
                results["threshold_percentiles"]["target_expression"],
            )
            ax1.axhline(
                y=crispr_threshold,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"CRISPR {results['threshold_percentiles']['crispr']}th %ile",
            )
            ax1.axvline(
                x=target_threshold,
                color="blue",
                linestyle="--",
                alpha=0.7,
                label=f"Target {results['threshold_percentiles']['target_expression']}th %ile",
            )

            # Panel 2: CRISPR vs Biomarker Expression
            ax2 = plt.subplot(num_pairs_to_plot, 3, base_idx + 2)
            ax2.scatter(biomarker_gexp_values, cas9_values, alpha=0.5, c="lightgray")

            if (
                cell_line in biomarker_gexp_values.index
                and cell_line in cas9_values.index
            ):
                highlighted_x = biomarker_gexp_values[cell_line]
                highlighted_y = cas9_values[cell_line]
                ax2.scatter(
                    [highlighted_x],
                    [highlighted_y],
                    c="red",
                    s=100,
                    edgecolor="black",
                    zorder=10,
                )

            corr2 = np.corrcoef(biomarker_gexp_values, cas9_values)[0, 1]
            ax2.set_title(
                f"{target_gene} vs {biomarker_gene}\nCRISPR vs Biomarker Expr (r={corr2:.3f})"
            )
            ax2.set_xlabel(f"{biomarker_gene} Expression")
            ax2.set_ylabel(f"{target_gene} CRISPR Score")

            # Add gene-specific threshold lines
            biomarker_threshold = np.percentile(
                biomarker_gexp_values,
                results["threshold_percentiles"]["biomarker_expression"],
            )
            ax2.axhline(
                y=crispr_threshold,
                color="red",
                linestyle="--",
                alpha=0.7,
            )
            ax2.axvline(
                x=biomarker_threshold,
                color="green",
                linestyle="--",
                alpha=0.7,
                label=f"Biomarker {results['threshold_percentiles']['biomarker_expression']}th %ile",
            )

            # Panel 3: Target Expression vs Biomarker Expression
            ax3 = plt.subplot(num_pairs_to_plot, 3, base_idx + 3)
            ax3.scatter(
                biomarker_gexp_values, target_gexp_values, alpha=0.5, c="lightgray"
            )

            if (
                cell_line in biomarker_gexp_values.index
                and cell_line in target_gexp_values.index
            ):
                highlighted_x = biomarker_gexp_values[cell_line]
                highlighted_y = target_gexp_values[cell_line]
                ax3.scatter(
                    [highlighted_x],
                    [highlighted_y],
                    c="red",
                    s=100,
                    edgecolor="black",
                    zorder=10,
                )

            corr3 = np.corrcoef(biomarker_gexp_values, target_gexp_values)[0, 1]
            ax3.set_title(f"Target vs Biomarker\nExpression (r={corr3:.3f})")
            ax3.set_xlabel(f"{biomarker_gene} Expression")
            ax3.set_ylabel(f"{target_gene} Expression")

            # Add gene-specific threshold lines
            ax3.axhline(
                y=target_threshold,
                color="blue",
                linestyle="--",
                alpha=0.7,
            )
            ax3.axvline(
                x=biomarker_threshold,
                color="green",
                linestyle="--",
                alpha=0.7,
            )

            # Find the gene pair in the original dataframe for additional info
            row = df_res_vae_annot_filtered_cap_top[
                (df_res_vae_annot_filtered_cap_top["y_id"] == target_gene)
                & (df_res_vae_annot_filtered_cap_top["x_id"] == biomarker_gene)
            ]
            # Gene pair info now uses fdr_vae as the primary significance measure

        plt.suptitle(
            f"Top {num_pairs_to_plot} Gene Pairs for Cell Line: {cell_line}\n3-Criteria Analysis",
            fontsize=16,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(
            f"{output_dir}/cell_line_{cell_line}_examples.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Coverage summary pie chart
    plt.figure(figsize=(8, 8))
    coverage_data = [
        results["evaluation"]["total_pairs_covered"],
        results["evaluation"]["total_pairs"]
        - results["evaluation"]["total_pairs_covered"],
    ]
    plt.pie(
        coverage_data,
        labels=["Covered", "Not Covered"],
        autopct="%1.1f%%",
        colors=["#66b3ff", "#ff9999"],
        startangle=90,
        shadow=True,
    )
    plt.axis("equal")
    plt.title("Overall Gene Pair Coverage\n(3-Criteria Selection)", fontsize=14)
    plt.savefig(f"{output_dir}/coverage_pie_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. New visualization: 3D scatter plot for top gene pair (if matplotlib supports it)
    try:
        from mpl_toolkits.mplot3d import Axes3D

        if results["gene_pair_examples"]:
            # Get the top gene pair from the first selected cell line
            examples = results["gene_pair_examples"][0]
            if examples:
                target_gene, biomarker_gene, _ = examples[0]

                # Get data for all cell lines
                cas9_values = cas9_df[target_gene].dropna()
                target_gexp_values = gexp_df[target_gene].dropna()
                biomarker_gexp_values = gexp_df[biomarker_gene].dropna()

                # Find common cell lines
                common_cells = cas9_values.index.intersection(
                    target_gexp_values.index
                ).intersection(biomarker_gexp_values.index)

                if len(common_cells) > 10:  # Only create if we have enough data points
                    cas9_common = cas9_values[common_cells]
                    target_gexp_common = target_gexp_values[common_cells]
                    biomarker_gexp_common = biomarker_gexp_values[common_cells]

                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.add_subplot(111, projection="3d")

                    # Plot all points
                    ax.scatter(
                        target_gexp_common,
                        biomarker_gexp_common,
                        cas9_common,
                        alpha=0.6,
                        c="lightgray",
                        s=50,
                    )

                    # Highlight selected cell lines
                    for selected_cell in results["selected_cell_lines"]:
                        if selected_cell in common_cells:
                            ax.scatter(
                                [target_gexp_common[selected_cell]],
                                [biomarker_gexp_common[selected_cell]],
                                [cas9_common[selected_cell]],
                                c="red",
                                s=100,
                                edgecolor="black",
                            )

                    ax.set_xlabel(f"{target_gene} Expression")
                    ax.set_ylabel(f"{biomarker_gene} Expression")
                    ax.set_zlabel(f"{target_gene} CRISPR Score")
                    ax.set_title(
                        f"3D View: Top Gene Pair\n{target_gene} (Target) vs {biomarker_gene} (Biomarker)"
                    )

                    plt.savefig(
                        f"{output_dir}/3d_scatter_top_pair.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
    except ImportError:
        print("3D plotting not available, skipping 3D visualization")

    print(f"Enhanced visualizations saved to {output_dir}")
