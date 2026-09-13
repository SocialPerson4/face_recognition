"""Tools for reproducible face-verification experiments."""

from .data import ORLDataset, load_orl, theoretical_pair_counts
from .protocol import DatasetSplit, PairRecord, PairSet, make_balanced_pairs, split_by_identity

__all__ = [
    "DatasetSplit",
    "ORLDataset",
    "PairRecord",
    "PairSet",
    "load_orl",
    "make_balanced_pairs",
    "split_by_identity",
    "theoretical_pair_counts",
]
