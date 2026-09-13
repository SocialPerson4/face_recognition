"""Metrics and threshold selection for one-to-one face verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class VerificationRates:
    """Confusion counts and rates at one accept/reject threshold."""

    threshold: float
    true_accepts: int
    false_rejects: int
    false_accepts: int
    true_rejects: int
    fmr: float
    fnmr: float
    true_accept_rate: float
    accuracy: float


@dataclass(frozen=True)
class EERResult:
    """Equal error rate and its linearly interpolated threshold."""

    eer: float
    threshold: float


@dataclass(frozen=True)
class ThresholdSelection:
    """Best observed threshold under a target false match rate."""

    target_fmr: float
    rates: VerificationRates


def _validated_inputs(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(values) or len(labels) == 0:
        raise ValueError("y_true and scores must have the same non-zero length")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must contain only finite values")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("y_true must contain both binary labels 0 and 1")
    return labels.astype(np.int64, copy=False), values


def rates_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> VerificationRates:
    """Compute verification errors when scores >= threshold are accepted."""

    labels, values = _validated_inputs(y_true, scores)
    if np.isnan(threshold):
        raise ValueError("threshold must not be NaN")

    accepted = values >= threshold
    genuine = labels == 1
    impostor = labels == 0
    true_accepts = int(np.sum(accepted & genuine))
    false_rejects = int(np.sum(~accepted & genuine))
    false_accepts = int(np.sum(accepted & impostor))
    true_rejects = int(np.sum(~accepted & impostor))

    genuine_count = int(np.sum(genuine))
    impostor_count = int(np.sum(impostor))
    total = len(labels)
    return VerificationRates(
        threshold=float(threshold),
        true_accepts=true_accepts,
        false_rejects=false_rejects,
        false_accepts=false_accepts,
        true_rejects=true_rejects,
        fmr=false_accepts / impostor_count,
        fnmr=false_rejects / genuine_count,
        true_accept_rate=true_accepts / genuine_count,
        accuracy=(true_accepts + true_rejects) / total,
    )


def select_threshold_at_fmr(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target_fmr: float = 0.01,
) -> ThresholdSelection:
    """Choose the lowest-FNMR observed threshold satisfying an FMR limit."""

    labels, values = _validated_inputs(y_true, scores)
    if not 0.0 <= target_fmr <= 1.0:
        raise ValueError("target_fmr must lie between 0 and 1")

    reject_all_threshold = np.nextafter(np.max(values), np.inf)
    thresholds = np.concatenate(
        ([reject_all_threshold], np.unique(values)[::-1])
    )
    feasible: list[VerificationRates] = []
    for threshold in thresholds:
        rates = rates_at_threshold(labels, values, float(threshold))
        if rates.fmr <= target_fmr:
            feasible.append(rates)
    selected = min(
        feasible,
        key=lambda rates: (rates.fnmr, rates.fmr, -rates.threshold),
    )
    return ThresholdSelection(target_fmr=float(target_fmr), rates=selected)


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> EERResult:
    """Compute EER by linear interpolation across the ROC operating points."""

    labels, values = _validated_inputs(y_true, scores)
    fmr, true_accept_rate, thresholds = roc_curve(
        labels, values, pos_label=1, drop_intermediate=False
    )
    fnmr = 1.0 - true_accept_rate
    difference = fmr - fnmr

    exact = np.flatnonzero(np.isclose(difference, 0.0))
    if len(exact):
        index = int(exact[0])
        return EERResult(eer=float(fmr[index]), threshold=float(thresholds[index]))

    crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
    if not len(crossings):
        index = int(np.argmin(np.abs(difference)))
        eer = float((fmr[index] + fnmr[index]) / 2.0)
        return EERResult(eer=eer, threshold=float(thresholds[index]))

    left = int(crossings[0])
    right = left + 1
    weight = float(-difference[left] / (difference[right] - difference[left]))
    eer = float(fmr[left] + weight * (fmr[right] - fmr[left]))
    threshold = float(
        thresholds[left] + weight * (thresholds[right] - thresholds[left])
    )
    return EERResult(eer=eer, threshold=threshold)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return ROC-AUC when larger scores mean more likely same identity."""

    labels, values = _validated_inputs(y_true, scores)
    return float(roc_auc_score(labels, values))
