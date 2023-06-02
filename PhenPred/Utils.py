#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr, spearmanr


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

    df = df.subtract(non_essential_metric).divide(
        non_essential_metric - essential_metric
    )

    return df


def two_vars_correlation(
    var1, var2, method="pearson", min_n=15, verbose=0, extra_fields=None
):
    if verbose > 0:
        print(f"Var1={var1.name}; Var2={var2.name}")

    nans_mask = np.logical_or(np.isnan(var1), np.isnan(var2))
    n = (~nans_mask).sum()

    if n <= min_n or np.std(var1[~nans_mask]) == 0 or np.std(var2[~nans_mask]) == 0:
        return dict(corr=np.nan, pval=np.nan, len=n)

    if method == "spearman":
        r, p = spearmanr(var1[~nans_mask], var2[~nans_mask], nan_policy="omit")
    else:
        r, p = pearsonr(var1[~nans_mask], var2[~nans_mask])

    res = dict(corr=r, pval=p, len=n)

    if extra_fields is not None:
        res.update(extra_fields)

    return res
