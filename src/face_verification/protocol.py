"""Reproducible, identity-disjoint face-verification protocols."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """Image indices and identities assigned to train, validation and test."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_identities: tuple[int, ...]
    validation_identities: tuple[int, ...]
    test_identities: tuple[int, ...]


@dataclass(frozen=True)
class PairRecord:
    """One unordered image pair and its same-identity target."""

    left_index: int
    right_index: int
    target: int


@dataclass(frozen=True)
class PairSet:
    """A deterministic collection of positive and negative pairs."""

    records: tuple[PairRecord, ...]

    @property
    def positive_count(self) -> int:
        return sum(record.target == 1 for record in self.records)

    @property
    def negative_count(self) -> int:
        return sum(record.target == 0 for record in self.records)


def split_by_identity(
    labels: np.ndarray,
    *,
    train_identities: int = 24,
    validation_identities: int = 8,
    test_identities: int = 8,
    seed: int = 20260913,
) -> DatasetSplit:
    """Split images so that each identity belongs to exactly one subset."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")

    unique_identities = np.unique(labels)
    requested = train_identities + validation_identities + test_identities
    if min(train_identities, validation_identities, test_identities) < 1:
        raise ValueError("each split must contain at least one identity")
    if requested != len(unique_identities):
        raise ValueError(
            f"requested {requested} identities but labels contain "
            f"{len(unique_identities)}"
        )

    shuffled = unique_identities.copy()
    np.random.default_rng(seed).shuffle(shuffled)

    train_stop = train_identities
    validation_stop = train_stop + validation_identities
    train_ids = np.sort(shuffled[:train_stop])
    validation_ids = np.sort(shuffled[train_stop:validation_stop])
    test_ids = np.sort(shuffled[validation_stop:])

    def indices_for(identities: np.ndarray) -> tuple[int, ...]:
        return tuple(np.flatnonzero(np.isin(labels, identities)).tolist())

    return DatasetSplit(
        train_indices=indices_for(train_ids),
        validation_indices=indices_for(validation_ids),
        test_indices=indices_for(test_ids),
        train_identities=tuple(int(value) for value in train_ids),
        validation_identities=tuple(int(value) for value in validation_ids),
        test_identities=tuple(int(value) for value in test_ids),
    )


def make_balanced_pairs(
    labels: np.ndarray,
    indices: tuple[int, ...],
    *,
    seed: int,
    negative_to_positive_ratio: float = 1.0,
) -> PairSet:
    """Create all positive pairs and a reproducible sample of negative pairs."""

    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if negative_to_positive_ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be positive")

    selected = tuple(sorted(set(int(index) for index in indices)))
    if not selected:
        raise ValueError("indices must not be empty")
    if selected[0] < 0 or selected[-1] >= len(labels):
        raise IndexError("pair index is outside the labels array")

    positive_candidates: list[tuple[int, int]] = []
    for identity in np.unique(labels[list(selected)]):
        identity_indices = [index for index in selected if labels[index] == identity]
        positive_candidates.extend(combinations(identity_indices, 2))

    if not positive_candidates:
        raise ValueError("at least one positive pair is required")

    negative_candidates = [
        (left, right)
        for left, right in combinations(selected, 2)
        if labels[left] != labels[right]
    ]
    requested_negatives = int(round(len(positive_candidates) * negative_to_positive_ratio))
    if requested_negatives > len(negative_candidates):
        raise ValueError(
            f"requested {requested_negatives} negatives but only "
            f"{len(negative_candidates)} are available"
        )

    rng = np.random.default_rng(seed)
    chosen_positions = rng.choice(
        len(negative_candidates), size=requested_negatives, replace=False
    )
    negative_pairs = [negative_candidates[int(position)] for position in chosen_positions]

    records = [
        PairRecord(left_index=left, right_index=right, target=1)
        for left, right in positive_candidates
    ]
    records.extend(
        PairRecord(left_index=left, right_index=right, target=0)
        for left, right in negative_pairs
    )
    permutation = rng.permutation(len(records))
    return PairSet(records=tuple(records[int(position)] for position in permutation))
