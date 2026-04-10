# Cell Line Selection — Results Tracker

This document tracks the output files produced by [notebooks/cell_line_selection_final.ipynb](../notebooks/cell_line_selection_final.ipynb), which selects cell lines for validation of synthetic-lethal (SL) target/biomarker pairs predicted by MOSA.

All result files live in [notebooks/validation_growth_tissue/](../notebooks/validation_growth_tissue/).

## Latest result

**[coverage_matrix_20251015_ID_mapped.csv](../notebooks/validation_growth_tissue/coverage_matrix_20251015_ID_mapped.csv)** — this is the current canonical output.

- 2000 SL pairs × 6 selected cell lines
- Produced at cell 47 of the notebook by `final_list_df.to_csv(...)` (see notebook line ~7729)
- Adds HGNC and Ensembl gene IDs to the previous `coverage_matrix_20251015.csv` via a lookup against [data/hgnc_complete_set.txt](../data/hgnc_complete_set.txt), with a fallback that resolves old/aliased symbols through HGNC `alias_symbol` and `prev_symbol` columns.

### Columns

| Column | Description |
|---|---|
| `target_gene` | SL target gene symbol |
| `target_hgnc_id` | HGNC ID of the target gene (e.g. `HGNC:18786`) |
| `target_ensembl_gene_id` | Ensembl gene ID of the target gene |
| `biomarker_gene` | SL biomarker (partner) gene symbol |
| `biomarker_hgnc_id` | HGNC ID of the biomarker gene |
| `biomarker_ensembl_gene_id` | Ensembl gene ID of the biomarker gene |
| `HT-29_Female_Large_Intestine` | Coverage indicator (0/1) for cell line HT-29 |
| `A375_Female_Skin` | Coverage indicator for A375 |
| `Hs-940-T_Male_Skin` | Coverage indicator for Hs-940-T |
| `MS751_Female_Cervix` | Coverage indicator for MS751 |
| `CCK-81_Female_Large_Intestine` | Coverage indicator for CCK-81 |
| `NCI-H1915_Female_Lung` | Coverage indicator for NCI-H1915 |
| `FDR_MOSA` | MOSA FDR for the pair |
| `Beta_MOSA` | MOSA beta (effect size) for the pair |

A `1` in a cell-line column means that the pair is "covered" (i.e. the cell line expresses / has the genotype required to test that pair) in the selected panel.

## File lineage (validation_growth_tissue/)

Files are listed newest-first by role in the pipeline. Earlier files are kept for reproducibility but are **superseded** by the ID-mapped version.

| File | Role | Status |
|---|---|---|
| [coverage_matrix_20251015_ID_mapped.csv](../notebooks/validation_growth_tissue/coverage_matrix_20251015_ID_mapped.csv) | Final coverage matrix with HGNC + Ensembl IDs annotated | **Current** |
| [coverage_matrix_20251015.csv](../notebooks/validation_growth_tissue/coverage_matrix_20251015.csv) | Coverage matrix with cell lines renamed to `<name>_<sex>_<tissue>` but without HGNC/Ensembl IDs | Superseded |
| [coverage_matrix_final_fix_top_pairs_df.csv](../notebooks/validation_growth_tissue/coverage_matrix_final_fix_top_pairs_df.csv) | Intermediate — top-N-pairs coverage matrix before cell-line column renaming | Intermediate |
| [coverage_matrix_df_filtered.csv](../notebooks/validation_growth_tissue/coverage_matrix_df_filtered.csv) | Earliest filtered coverage matrix dump | Legacy |

### SL pairs overlap (separate analysis)

These files come from [notebooks/sl_pairs_overlap_analysis.ipynb](../notebooks/sl_pairs_overlap_analysis.ipynb) and compare our predictions to the collaborator's list:

- [our_unique_sl_pairs.csv](../notebooks/validation_growth_tissue/our_unique_sl_pairs.csv)
- [collaborator_unique_sl_pairs.csv](../notebooks/validation_growth_tissue/collaborator_unique_sl_pairs.csv)
- [overlapping_sl_pairs.csv](../notebooks/validation_growth_tissue/overlapping_sl_pairs.csv)
- [sl_pairs_overlap_summary.txt](../notebooks/validation_growth_tissue/sl_pairs_overlap_summary.txt)

### Figures

- [cell_line_coverage_bar.png](../notebooks/validation_growth_tissue/cell_line_coverage_bar.png) — coverage per cell line
- [coverage_pie_chart.png](../notebooks/validation_growth_tissue/coverage_pie_chart.png) — overall coverage breakdown
- [selected_cell_lines_heatmap.png](../notebooks/validation_growth_tissue/selected_cell_lines_heatmap.png) — pair × cell line coverage heatmap
- [3d_scatter_top_pair.png](../notebooks/validation_growth_tissue/3d_scatter_top_pair.png)
- [top_scatter_plots.png](../notebooks/validation_growth_tissue/top_scatter_plots.png) / [top_scatter_plots.pdf](../notebooks/validation_growth_tissue/top_scatter_plots.pdf)
- Per-cell-line example plots: `cell_line_SIDM00136_examples.png`, `cell_line_SIDM00662_examples.png`, `cell_line_SIDM00795_examples.png`
- [sl_pairs_overlap_analysis.png](../notebooks/validation_growth_tissue/sl_pairs_overlap_analysis.png)

## Notebook configuration

Key constants set at the top of [cell_line_selection_final.ipynb](../notebooks/cell_line_selection_final.ipynb) (around line 1072):

```python
RESULT_DIR = "./validation_growth_tissue"
TIMESTAMP  = "20251015"
```

`TIMESTAMP` is what appears in the output filenames (`coverage_matrix_{TIMESTAMP}.csv`). To produce a new dated run, bump `TIMESTAMP` and re-execute the notebook end-to-end; the ID-mapping cell at the bottom then needs its hard-coded path updated to match.

## Update log

- **2025-11-24** — `coverage_matrix_20251015_ID_mapped.csv` regenerated (adds HGNC + Ensembl annotations).
- **2025-10-20** — SL-pairs overlap analysis outputs added.
- **2025-10-15** — Initial `coverage_matrix_20251015.csv` and top-pair figures generated.
