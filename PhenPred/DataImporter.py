#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import pandas as pd


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

		else:
			raise Exception(f"Dataset {dataset_name}, not suported.")

	def read_proteomics(self):
		df = pd.read_csv(f"{self.data_dir}/proteomics.csv", index_col=0).T

		return df

	def read_metabolomics(self):
		df = pd.read_csv(f"{self.data_dir}/metabolomics.csv", index_col=0).T

		return df

	def read_drug_response(self):
		return pd.read_csv(f"{self.data_dir}/drugresponse.csv", index_col=0).T

	def read_tissue_types(self):
		return pd.get_dummies(pd.read_csv(f"{self.data_dir}/samplesheet.csv", index_col=0)["tissue"])
