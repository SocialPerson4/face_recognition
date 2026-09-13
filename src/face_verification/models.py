"""Classifier construction and numerically checked decision scoring."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_verification_classifier(
    family: str,
    *,
    c: float = 1.0,
    gamma: str | float = "scale",
    seed: int = 20260913,
) -> Pipeline:
    """Build a standardized binary classifier for absolute pair differences."""

    if c <= 0.0:
        raise ValueError("c must be positive")
    if family == "logistic_regression":
        estimator = LogisticRegression(
            C=c,
            penalty="l2",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed,
        )
    elif family == "linear_svm":
        estimator = SVC(C=c, kernel="linear")
    elif family == "rbf_svm":
        if isinstance(gamma, (int, float)) and gamma <= 0.0:
            raise ValueError("numeric gamma must be positive")
        estimator = SVC(C=c, kernel="rbf", gamma=gamma)
    else:
        raise ValueError(f"unknown classifier family: {family}")
    return make_pipeline(StandardScaler(), estimator)


def fit_and_score_classifier(
    model: Pipeline,
    train_features: np.ndarray,
    y_train: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a classifier and return finite train and validation decision scores."""

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(train_features, y_train)
        train_scores = np.asarray(model.decision_function(train_features))
        validation_scores = np.asarray(model.decision_function(validation_features))
    if not (
        np.all(np.isfinite(train_scores))
        and np.all(np.isfinite(validation_scores))
    ):
        raise FloatingPointError("classifier produced non-finite decision scores")
    return train_scores, validation_scores
