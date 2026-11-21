"""The three classifier families and how their probabilities are put on one scale.

Each family folder (trees/, logistic/, mlp/) trains one of these; train_all.py compares all
three. The settings are the ones used for the thesis models (v1.0.3).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def imputer():
    # keep_empty_features: an all-NaN column must stay in place, or the exported model's
    # feature indices would shift (it doesn't occur with these features, but be safe)
    return SimpleImputer(strategy="mean", keep_empty_features=True)


def trees():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, max_depth=8,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def logistic():
    return make_pipeline(imputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def mlp():
    return make_pipeline(imputer(), StandardScaler(),
                         MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=0))


@dataclass(frozen=True)
class Family:
    kind: str            # short name, used in file names and by hcpose (--classifier trees|logistic|mlp)
    title: str           # name in printed tables
    make: Callable[[], object]


FAMILIES = {
    "logistic": Family("logistic", "logistic regression", logistic),
    "trees": Family("trees", "gradient-boosted trees", trees),
    "mlp": Family("mlp", "MLP (64-32)", mlp),
}


def prior_offset(kind, y_train):
    """The MLP can't use class weights, so its probabilities follow the training prior.
    Shifting its logit by log((1-p)/p) gives the balanced probability the other two produce,
    so one set of analyzer thresholds works for all models."""
    if kind != "mlp":
        return 0.0
    p = float(np.mean(y_train))
    return float(np.log((1 - p) / p))


def shift(prob, offset):
    if offset == 0.0:
        return prob
    q = np.clip(prob, 1e-12, 1 - 1e-12)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + offset)))
