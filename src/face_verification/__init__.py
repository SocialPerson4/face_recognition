"""Tools for reproducible face-verification experiments."""

from .data import ORLDataset, load_orl, theoretical_pair_counts
from .features import (
    fit_pca,
    fit_pca_components,
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
from .protocol import (
    DatasetSplit,
    IdentityFold,
    PairRecord,
    PairSet,
    make_balanced_pairs,
    make_identity_folds,
    split_by_identity,
)
from .selection import OneStandardErrorResult, select_k_one_standard_error

__all__ = [
    "DatasetSplit",
    "EERResult",
    "IdentityFold",
    "ORLDataset",
    "OneStandardErrorResult",
    "PairRecord",
    "PairSet",
    "ThresholdSelection",
    "VerificationRates",
    "equal_error_rate",
    "fit_pca",
    "fit_pca_components",
    "image_matrix",
    "load_orl",
    "make_balanced_pairs",
    "make_identity_folds",
    "negative_euclidean_pair_scores",
    "rates_at_threshold",
    "roc_auc",
    "select_threshold_at_fmr",
    "select_k_one_standard_error",
    "split_by_identity",
    "theoretical_pair_counts",
    "transform_images",
]
