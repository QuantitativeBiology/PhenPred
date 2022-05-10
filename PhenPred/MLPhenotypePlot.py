#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


if __name__ == '__main__':
	"""
	Imports
	"""
	paper_scores = pd.read_csv(f"data/Figures_Source_Data - Fig3B_ml_benchmark.csv")

	grant_scores = pd.read_csv(f"data/ML_phenotype_prediction_drug_metabolomics.csv")

	"""
	Fig S3c-like plot
	"""
	# Dataset
	plot_df = pd.concat([
		paper_scores,
		grant_scores.rename(columns=dict(feature="drug_id", pearsonsr="score", ml_method="model", dataset_x="data"))
	])
	plot_df = plot_df.replace(dict(model=dict(rf="RF"), data=dict(metabolomics="Metabolomics")))

	# Plot
	order = ["WES", "Copy Number", "Tissue", "Methylation", "Transcriptome", "Proteome", "Metabolomics"]

	_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

	sns.boxplot(
		data=plot_df,
		x="score",
		y="data",
		hue="model",
		order=order,
		palette="Set2",
		linewidth=0.3,
		fliersize=1,
		notch=True,
		saturation=1.0,
		showcaps=False,
		boxprops=dict(linewidth=0.5),
		whiskerprops=dict(linewidth=0.5),
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

	ax.set_xlabel(f"Pearson's r")
	ax.set_ylabel(f"")

	ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.3))
	ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=-1)

	ax.legend(frameon=False, prop={"size": 5})

	plt.savefig(f"reports/PhenotypePrediction_Drug_Boxplot.png", bbox_inches="tight")
	plt.close("all")
