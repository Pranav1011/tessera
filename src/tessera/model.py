"""Thin wrapper around the CPU primitive-segmentation baseline (XGBoost on the
per-face geometric features). No GPU, no deep net — the model is a vehicle for the
eval harness, and gradient-boosted trees on hand-built geometry features is an
honest, defensible baseline."""

from __future__ import annotations

import numpy as np
import xgboost as xgb


def train(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.3,
        tree_method="hist", n_jobs=4, verbosity=0,
        objective="multi:softprob", num_class=n_classes, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def predict(clf: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    return clf.predict(X)
