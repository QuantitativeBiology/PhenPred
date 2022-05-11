#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import sys

sys.path.extend(['/Volumes/GoogleDrive-108722195023672559969/My Drive/Grants/2023 ERC STG/PhenPred'])

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


if __name__ == '__main__':
	"""
	Imports
	"""
	scores = pd.read_csv(f"data/ML_phenotype_prediction.csv")

	"""
	Fig S3c-like plot
	"""
	order = ["genomics", "tissue", "methylation", "transcriptomics", "proteomics", "metabolomics"]

	_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

	sns.boxplot(
		data=scores,
		x="pearsonsr",
		y="dataset_x",
		orient="h",
		order=order,
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

	plt.savefig(f"reports/PhenotypePrediction_Drug_Boxplot.png", bbox_inches="tight")
	plt.close("all")

	"""
	
	"""
	order = ["(-3, -1]", "(-1, 3]"]
	order_hue = ["genomics", "tissue", "methylation", "transcriptomics", "proteomics", "metabolomics"]

	plot_df = pd.concat([
		scores,
		pd.cut(scores["skew"], [-3, -1, 3], labels=order).rename("skew_bin"),
	], axis=1)

	_, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=600)

	sns.boxplot(
		data=plot_df,
		x="pearsonsr",
		y="skew_bin",
		hue="dataset_x",
		orient="h",
		order=order,
		hue_order=order_hue,
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
	ax.set_ylabel(f"Skew bin")

	ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.3))
	ax.grid(axis="x", lw=0.1, color="#e1e1e1", zorder=-1)

	ax.legend(frameon=False, prop={"size": 5}, loc="center left", bbox_to_anchor=(1, 0.5))

	plt.savefig(f"reports/PhenotypePrediction_Drug_Boxplot_Skew.png", bbox_inches="tight")
	plt.close("all")

