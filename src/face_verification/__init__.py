"""Tools for reproducible face-verification experiments."""

from .data import ORLDataset, load_orl, theoretical_pair_counts
from .features import (
    fit_pca,
    image_matrix,
    negative_euclidean_pair_scores,
    transform_images,
)
from .metrics import (
    EERResult,
    ThresholdSelection,
    VerificationRates,
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    select_threshold_at_fmr,
)
from .protocol import DatasetSplit, PairRecord, PairSet, make_balanced_pairs, split_by_identity

__all__ = [
    "DatasetSplit",
    "EERResult",
    "ORLDataset",
    "PairRecord",
    "PairSet",
    "ThresholdSelection",
    "VerificationRates",
    "equal_error_rate",
    "fit_pca",
    "image_matrix",
    "load_orl",
    "make_balanced_pairs",
    "negative_euclidean_pair_scores",
    "rates_at_threshold",
    "roc_auc",
    "select_threshold_at_fmr",
    "split_by_identity",
    "theoretical_pair_counts",
    "transform_images",
]
