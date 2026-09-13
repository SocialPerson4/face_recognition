"""Predeclared model-selection rules for cross-validation summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OneStandardErrorResult:
    """Outcome of choosing the simplest candidate within one best-model SE."""

    observed_best_k: int
    selected_k: int
    best_mean_loss: float
    best_standard_deviation: float
    best_standard_error: float
    eligibility_cutoff: float
    eligible_k: tuple[int, ...]


@dataclass(frozen=True)
class OneStandardErrorIndexResult:
    """One-SE result for configurations represented by caller-defined complexity."""

    observed_best_index: int
    selected_index: int
    best_mean_loss: float
    best_standard_deviation: float
    best_standard_error: float
    eligibility_cutoff: float
    eligible_indices: tuple[int, ...]


def select_k_one_standard_error(
    k_values: np.ndarray,
    mean_losses: np.ndarray,
    standard_deviations: np.ndarray,
    *,
    n_folds: int,
) -> OneStandardErrorResult:
    """Choose the smallest k within one standard error of the lowest mean loss."""

    k = np.asarray(k_values)
    means = np.asarray(mean_losses, dtype=np.float64)
    deviations = np.asarray(standard_deviations, dtype=np.float64)
    if k.ndim != 1 or means.ndim != 1 or deviations.ndim != 1:
        raise ValueError("selection inputs must be one-dimensional")
    if len(k) == 0 or not (len(k) == len(means) == len(deviations)):
        raise ValueError("selection inputs must have the same non-zero length")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not np.issubdtype(k.dtype, np.integer) or np.any(k < 1):
        raise ValueError("k_values must contain positive integers")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(deviations)):
        raise ValueError("loss statistics must be finite")
    if np.any(deviations < 0.0):
        raise ValueError("standard deviations must not be negative")

    best_index = min(
        range(len(k)),
        key=lambda index: (means[index], deviations[index], k[index]),
    )
    standard_error = float(deviations[best_index] / np.sqrt(n_folds))
    cutoff = float(means[best_index] + standard_error)
    eligible = tuple(sorted(int(value) for value in k[means <= cutoff]))
    return OneStandardErrorResult(
        observed_best_k=int(k[best_index]),
        selected_k=eligible[0],
        best_mean_loss=float(means[best_index]),
        best_standard_deviation=float(deviations[best_index]),
        best_standard_error=standard_error,
        eligibility_cutoff=cutoff,
        eligible_k=eligible,
    )


def select_configuration_one_standard_error(
    mean_losses: np.ndarray,
    standard_deviations: np.ndarray,
    complexity_keys: np.ndarray,
    *,
    n_folds: int,
) -> OneStandardErrorIndexResult:
    """Choose the least complex configuration within one best-model SE."""

    means = np.asarray(mean_losses, dtype=np.float64)
    deviations = np.asarray(standard_deviations, dtype=np.float64)
    complexity = np.asarray(complexity_keys, dtype=np.float64)
    if means.ndim != 1 or deviations.ndim != 1:
        raise ValueError("loss statistics must be one-dimensional")
    if complexity.ndim == 1:
        complexity = complexity[:, None]
    if complexity.ndim != 2:
        raise ValueError("complexity_keys must be one- or two-dimensional")
    if len(means) == 0 or not (
        len(means) == len(deviations) == len(complexity)
    ):
        raise ValueError("selection inputs must have the same non-zero length")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if not (
        np.all(np.isfinite(means))
        and np.all(np.isfinite(deviations))
        and np.all(np.isfinite(complexity))
    ):
        raise ValueError("selection statistics and complexity must be finite")
    if np.any(deviations < 0.0):
        raise ValueError("standard deviations must not be negative")

    def ranking_key(index: int) -> tuple[float, ...]:
        return (
            float(means[index]),
            float(deviations[index]),
            *(float(value) for value in complexity[index]),
        )

    observed = min(range(len(means)), key=ranking_key)
    standard_error = float(deviations[observed] / np.sqrt(n_folds))
    cutoff = float(means[observed] + standard_error)
    eligible = tuple(int(index) for index in np.flatnonzero(means <= cutoff))
    selected = min(
        eligible,
        key=lambda index: tuple(float(value) for value in complexity[index]),
    )
    return OneStandardErrorIndexResult(
        observed_best_index=observed,
        selected_index=selected,
        best_mean_loss=float(means[observed]),
        best_standard_deviation=float(deviations[observed]),
        best_standard_error=standard_error,
        eligibility_cutoff=cutoff,
        eligible_indices=eligible,
    )
