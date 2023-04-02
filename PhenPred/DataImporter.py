#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import pandas as pd
from PhenPred.Utils import scale
from sklearn.preprocessing import StandardScaler


class DataImporter:
	def __init__(self, data_dir):
		self.data_dir = data_dir

	def read_dataset(self, dataset_name):
		"""
		Convenience function to input any dataset supported
		"""
		if dataset_name.lower() == "metabolomics":
			return self.read_metabolomics()

		elif dataset_name.lower() == "drug_response":
			return self.read_drug_response()

		elif dataset_name.lower() == "proteomics":
			return self.read_proteomics()

		elif dataset_name.lower() == "tissue":
			return self.read_tissue_types()

		elif dataset_name.lower() == "methylation":
			return self.read_methylation()

		elif dataset_name.lower() == "transcriptomics":
			return self.read_transcriptomics()

		elif dataset_name.lower() == "genomics":
			return self.read_genomics()

		elif dataset_name.lower() == "essentiality":
			return self.read_essentiality()

		else:
			raise Exception(f"Dataset {dataset_name}, not suported.")

	def read_proteomics(self):
		return pd.read_csv(f"{self.data_dir}/proteomics.csv", index_col=0).T

	def read_metabolomics(self):
		df = pd.read_csv(f"{self.data_dir}/metabolomics.csv", index_col=0).T
		df = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)
		return df

	def read_drug_response(self, only_gdsc2=True):
		df = pd.read_csv(f"{self.data_dir}/drugresponse.csv", index_col=0).T

		if only_gdsc2:
			df = df[[c for c in df if c.split(";")[2] == "GDSC2"]]

		return df

	def read_tissue_types(self):
		return pd.get_dummies(pd.read_csv(f"{self.data_dir}/samplesheet.csv", index_col=0)["tissue"])

	def read_methylation(self):
		return pd.read_csv(f"{self.data_dir}/methylation.csv", index_col=0).T

	def read_transcriptomics(self):
		return pd.read_csv(f"{self.data_dir}/transcriptomics.csv", index_col=0).T

	def read_genomics(self):
		return pd.read_csv(f"{self.data_dir}/genomics.csv", index_col=0).T

	def read_essentiality(self, only_context_fitness_genes=True):
		df = pd.read_csv(f"{self.data_dir}/crisprcas9.csv", index_col=0).dropna(axis=1)
		df = scale(df)
		df = df.T

		if only_context_fitness_genes:
			cf_genes = (df < -0.5).sum()
			cf_genes = cf_genes.loc[(cf_genes < df.shape[0] * 0.5) & (cf_genes >= 10)]
			df = df[cf_genes.index]

		return df
