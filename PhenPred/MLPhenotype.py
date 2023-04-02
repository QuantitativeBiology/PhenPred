#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

"""
Adapted from Simon https://github.com/EmanuelGoncalves/cancer_proteomics/tree/master/machine_learning
"""
import sys

sys.path.extend([
	"/Volumes/GoogleDrive-108722195023672559969/My Drive/Grants/2023 ERC STG/PhenPred",
	"/home/egoncalves/PhenPred",
])

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
from datetime import datetime
from PhenPred.Utils import read_yaml
from PhenPred.Utils import pearsonsr_scorer
from PhenPred.DataImporter import DataImporter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV


def get_ml_model(model_name):
	if model_name.lower() == "rf":
		return RandomForestRegressor()

	elif model_name.lower() == "en":
		return ElasticNet()

	else:
		raise Exception(f"ML Model {model_name} not supported")


def get_imputer(dataset_name):
	if dataset_name.lower() == "metabolomics":
		return SimpleImputer(missing_values=np.nan, strategy="mean")

	elif dataset_name.lower() == "proteomics":
		return SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-2.242616)

	else:
		return SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)


if __name__ == "__main__":
	"""
	Read argvs
	"""
	# configs = read_yaml("config/config_rf_drug.yml")
	configs = read_yaml(sys.argv[1])

	# dataset_x = "metabolomics"
	dataset_x = sys.argv[2]
	print(f"configs={sys.argv[1]}; dataset_x={dataset_x}")

	"""
	Data importer
	"""
	data = DataImporter(configs["DATA"]["dir"])

	"""
	Datasets
	"""
	# Create Y
	Y = data.read_dataset(configs["ML"]["dataset_y"])
	Y = Y.loc[:, Y.count() >= configs["DATA"]["min_count"]]

	# Create X
	X = data.read_dataset(dataset_x)

	if sys.argv[3] != "None":
		X2 = data.read_dataset(sys.argv[3]).reindex(X.index)
		X2 = pd.DataFrame(get_imputer(sys.argv[3]).fit_transform(X2), index=X2.index, columns=X2.columns)
		X = pd.concat([X2, X], axis=1)

	X = X.loc[:, X.count() >= configs["DATA"]["min_count"]]
	print(f"Datasets {configs['ML']['dataset_y']} and {dataset_x} imported")

	"""
	ML
	"""
	ml_results = []

	for feature in tqdm(Y.columns, miniters=1):
		# Regressor GridSearchCV
		regressor = GridSearchCV(
			get_ml_model(configs["ML"]["model"]),
			configs["ML"]["params_grid"],
			n_jobs=configs["ML"]["params_grid"]["n_jobs"][0],
			cv=KFold(n_splits=configs["ML"]["cv"]["n_splits"], shuffle=True),
			scoring=pearsonsr_scorer,
			refit=True,
		)

		# Overlapping observations without missing values
		samples = set(X.index).intersection(Y[feature].dropna().index)

		# If minimum number of samples not reached skip feature
		if len(samples) < configs["DATA"]["min_count"]:
			continue

		# Fit regressor
		regressor.fit(
			get_imputer(dataset_x).fit_transform(X.loc[samples]),
			Y.loc[samples, feature]
		)

		# Regressor results
		time = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

		ml_results.append(dict(
			feature=feature,
			dataset_y=configs["ML"]["dataset_y"],
			dataset_x=dataset_x,
			ml_method=configs["ML"]["model"],
			n_observations=len(samples),
			pearsonsr=regressor.best_score_,
			skew=skew(Y.loc[samples, feature]),
			time=time,
		))

		print(f"\n[{time}] {feature}; R2={regressor.best_score_:.2f}; Best params={regressor.best_params_}")

	"""
	Output
	"""
	ml_results_df = pd.DataFrame(ml_results)

	output_file = f"{configs['OUTPUT']['dir']}/ML_phenotype_prediction_{configs['ML']['dataset_y']}"

	if sys.argv[3] != "None":
		output_file += f"_{sys.argv[3]}"

	output_file += ".csv"

	with open(output_file, 'a') as f:
		ml_results_df.to_csv(f, mode='a', index=False, header=f.tell() == 0)
