"""Feature extraction and distance scoring for face-verification baselines."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from .protocol import PairSet


def image_matrix(images: np.ndarray) -> np.ndarray:
    """Return finite images as a two-dimensional sample-by-pixel matrix."""

    # Use float64 for stable PCA projection and distance accumulation.  The ORL
    # loader stores float32 to save memory, but the experiment matrix is small
    # enough that the extra precision is inexpensive and avoids BLAS warnings.
    values = np.asarray(images, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("images must contain a sample axis and at least one feature axis")
    if values.shape[0] < 2:
        raise ValueError("at least two images are required")

    matrix = values.reshape(values.shape[0], -1)
    if matrix.shape[1] == 0:
        raise ValueError("images must contain at least one pixel or feature")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("images must contain only finite values")
    return matrix


def fit_pca(
    train_images: np.ndarray,
    *,
    retained_variance: float = 0.95,
) -> PCA:
    """Fit PCA only on the images explicitly supplied as training data."""

    if not 0.0 < retained_variance <= 1.0:
        raise ValueError("retained_variance must lie in (0, 1]")

    train_matrix = image_matrix(train_images)
    n_components: float | None = retained_variance if retained_variance < 1.0 else None
    model = PCA(n_components=n_components, svd_solver="full")
    model.fit(train_matrix)
    return model


def fit_pca_components(train_images: np.ndarray, *, n_components: int) -> PCA:
    """Fit a fixed number of PCA components on explicitly supplied images."""

    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be an integer")
    train_matrix = image_matrix(train_images)
    maximum = min(train_matrix.shape)
    if not 1 <= int(n_components) <= maximum:
        raise ValueError(f"n_components must lie between 1 and {maximum}")

    model = PCA(n_components=int(n_components), svd_solver="full")
    model.fit(train_matrix)
    return model


def transform_images(model: PCA, images: np.ndarray) -> np.ndarray:
    """Project images into an already fitted PCA coordinate system."""

    if not hasattr(model, "components_"):
        raise ValueError("PCA model must be fitted before transformation")
    matrix = image_matrix(images)
    if matrix.shape[1] != model.n_features_in_:
        raise ValueError(
            f"PCA expects {model.n_features_in_} features, found {matrix.shape[1]}"
        )

    # This is the same projection used by PCA.transform.  np.einsum avoids
    # spurious NumPy/Accelerate matmul overflow warnings observed on macOS;
    # the explicit finite-value check still rejects any real numerical failure.
    transformed = np.einsum(
        "ij,kj->ik",
        matrix - model.mean_,
        model.components_,
        optimize=False,
    )
    if model.whiten:
        transformed /= np.sqrt(model.explained_variance_)
    if not np.all(np.isfinite(transformed)):
        raise FloatingPointError("PCA transformation produced non-finite values")
    return transformed


def negative_euclidean_pair_scores(
    features: np.ndarray,
    pairs: PairSet,
) -> np.ndarray:
    """Score pairs with negative Euclidean distance so larger means more similar."""

    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix")
    if not pairs.records:
        raise ValueError("pairs must not be empty")

    left = np.fromiter(
        (record.left_index for record in pairs.records), dtype=np.int64
    )
    right = np.fromiter(
        (record.right_index for record in pairs.records), dtype=np.int64
    )
    if min(int(left.min()), int(right.min())) < 0:
        raise IndexError("pair index must not be negative")
    if max(int(left.max()), int(right.max())) >= matrix.shape[0]:
        raise IndexError("pair index is outside the feature matrix")
    if not (
        np.all(np.isfinite(matrix[left])) and np.all(np.isfinite(matrix[right]))
    ):
        raise ValueError("features selected by pairs must contain only finite values")

    distances = np.linalg.norm(matrix[left] - matrix[right], axis=1)
    return -distances
