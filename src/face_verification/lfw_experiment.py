"""Shared helpers for identity-disjoint LFW experiments."""

from __future__ import annotations

import numpy as np

from .lfw_protocol import ProtocolPair
from .protocol import PairRecord, PairSet


def fold_roles(test_fold: int) -> tuple[tuple[int, ...], int, int]:
    if test_fold not in range(1, 11):
        raise ValueError("test_fold must lie between 1 and 10")
    calibration_fold = test_fold % 10 + 1
    train_folds = tuple(
        fold for fold in range(1, 11) if fold not in {test_fold, calibration_fold}
    )
    return train_folds, calibration_fold, test_fold


def to_pair_set(
    pairs: list[ProtocolPair], image_index: dict[str, int]
) -> PairSet:
    return PairSet(
        records=tuple(
            PairRecord(
                left_index=image_index[pair.left_image],
                right_index=image_index[pair.right_image],
                target=pair.target,
            )
            for pair in pairs
        )
    )


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }
