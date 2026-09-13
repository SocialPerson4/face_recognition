"""Tools for reproducible face-verification experiments."""

from .data import ORLDataset, load_orl, theoretical_pair_counts
from .features import (
    absolute_pair_differences,
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
from .models import fit_and_score_classifier, make_verification_classifier
from .protocol import (
    DatasetSplit,
    IdentityFold,
    PairRecord,
    PairSet,
    make_balanced_pairs,
    make_identity_folds,
    split_by_identity,
)
from .selection import (
    OneStandardErrorIndexResult,
    OneStandardErrorResult,
    select_configuration_one_standard_error,
    select_k_one_standard_error,
)

__all__ = [
    "DatasetSplit",
    "EERResult",
    "IdentityFold",
    "ORLDataset",
    "OneStandardErrorResult",
    "OneStandardErrorIndexResult",
    "PairRecord",
    "PairSet",
    "ThresholdSelection",
    "VerificationRates",
    "absolute_pair_differences",
    "equal_error_rate",
    "fit_pca",
    "fit_pca_components",
    "fit_and_score_classifier",
    "image_matrix",
    "load_orl",
    "make_balanced_pairs",
    "make_identity_folds",
    "make_verification_classifier",
    "negative_euclidean_pair_scores",
    "rates_at_threshold",
    "roc_auc",
    "select_threshold_at_fmr",
    "select_k_one_standard_error",
    "select_configuration_one_standard_error",
    "split_by_identity",
    "theoretical_pair_counts",
    "transform_images",
]
