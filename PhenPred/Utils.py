#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import yaml
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


def _pearsonr(y_true, y_pred):
    if y_true.ndim == 2:
        return [pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])]
    else:
        return pearsonr(y_true, y_pred)[0]


pearsonsr_scorer = make_scorer(_pearsonr, greater_is_better=True)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def get_essential_genes(dfile="data/EssentialGenes.csv", return_series=True):
    geneset = set(pd.read_csv(f"{dfile}", sep="\t")["gene"])
    if return_series:
        geneset = pd.Series(list(geneset)).rename("essential")
    return geneset


def get_non_essential_genes(dfile="data/NonessentialGenes.csv", return_series=True):
    geneset = set(pd.read_csv(f"{dfile}", sep="\t")["gene"])
    if return_series:
        geneset = pd.Series(list(geneset)).rename("non-essential")
    return geneset


def scale(df, essential=None, non_essential=None, metric=np.median):
    if essential is None:
        essential = get_essential_genes(return_series=False)

    if non_essential is None:
        non_essential = get_non_essential_genes(return_series=False)

    essential_metric = metric(df.reindex(essential).dropna(), axis=0)
    non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

    df = df.subtract(non_essential_metric).divide(non_essential_metric - essential_metric)

    return df
