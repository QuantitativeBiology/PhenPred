#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

"""
Adapted from Simon https://github.com/EmanuelGoncalves/cancer_proteomics/tree/master/machine_learning
"""
import sys

sys.path.extend(['/Volumes/GoogleDrive-108722195023672559969/My Drive/Grants/2023 ERC STG/PhenPred'])

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

	elif dataset_name.lower() == "tissue":
		return SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)

	else:
		raise Exception(f"Dataset {dataset_name}, not suported.")


if __name__ == "__main__":
	"""
	Read argvs
	"""
	configs = read_yaml(sys.argv[1])
	dataset_x = sys.argv[2]
	print(f"configs={sys.argv[1]}; dataset_x={dataset_x}")

	"""
	Data importer
	"""
	data = DataImporter(configs["DATA"]["dir"])

	"""
	Datasets
	"""
	Y = data.read_dataset(configs["ML"]["dataset_y"])
	Y = Y.loc[:, Y.count() >= configs["DATA"]["min_count"]]

	X = data.read_dataset(dataset_x)
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
			n_jobs=3,
			cv=KFold(n_splits=configs["ML"]["cv"]["n_splits"], shuffle=True),
			scoring=pearsonsr_scorer,
			refit=True,
		)

		# Overlapping observations without missing values
		samples = set(X.index).intersection(Y[feature].dropna().index)
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
	with open(configs["OUTPUT"]["file"], 'a') as f:
		ml_results_df.to_csv(f, mode='a', index=False, header=f.tell() == 0)
