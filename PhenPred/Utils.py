#!/usr/bin/env python
# Copyright (C) 2022 Emanuel Goncalves

import yaml
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


def _pearsonr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


pearsonsr_scorer = make_scorer(_pearsonr, greater_is_better=True)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
