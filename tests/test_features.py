import unittest

import numpy as np

from face_verification.features import (
    absolute_pair_differences,
    fit_pca,
    fit_pca_components,
    image_matrix,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.protocol import PairRecord, PairSet


class ImageFeatureTests(unittest.TestCase):
    def test_image_matrix_flattens_without_changing_sample_order(self) -> None:
        images = np.arange(24, dtype=np.float64).reshape(3, 2, 4)
        matrix = image_matrix(images)
        self.assertEqual(matrix.shape, (3, 8))
        np.testing.assert_array_equal(matrix[1], images[1].reshape(-1))

    def test_pca_is_fitted_from_supplied_training_images_only(self) -> None:
        train = np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [0.0, 0.0]],
            ]
        )
        unseen = np.full((2, 2, 2), 100.0)
        model = fit_pca(train, retained_variance=0.95)

        np.testing.assert_allclose(model.mean_, image_matrix(train).mean(axis=0))
        self.assertFalse(np.allclose(model.mean_, image_matrix(unseen).mean(axis=0)))
        self.assertEqual(transform_images(model, unseen).shape[0], 2)

    def test_negative_distance_gives_closer_pair_a_larger_score(self) -> None:
        features = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 4.0]])
        pairs = PairSet(
            records=(
                PairRecord(0, 1, 1),
                PairRecord(0, 2, 0),
            )
        )
        scores = negative_euclidean_pair_scores(features, pairs)
        np.testing.assert_allclose(scores, [-0.1, -5.0])
        self.assertGreater(scores[0], scores[1])

    def test_unselected_placeholder_rows_do_not_affect_pair_scores(self) -> None:
        features = np.array([[0.0], [1.0], [np.nan]])
        pairs = PairSet(records=(PairRecord(0, 1, 1),))
        np.testing.assert_allclose(
            negative_euclidean_pair_scores(features, pairs), [-1.0]
        )

    def test_absolute_pair_features_are_order_invariant(self) -> None:
        features = np.array([[1.0, 5.0], [4.0, 2.0]])
        forward = PairSet(records=(PairRecord(0, 1, 1),))
        backward = PairSet(records=(PairRecord(1, 0, 1),))
        np.testing.assert_allclose(
            absolute_pair_differences(features, forward), [[3.0, 3.0]]
        )
        np.testing.assert_allclose(
            absolute_pair_differences(features, forward),
            absolute_pair_differences(features, backward),
        )

    def test_invalid_feature_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            image_matrix(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            fit_pca(np.ones((3, 2)), retained_variance=0.0)
        with self.assertRaises(ValueError):
            fit_pca_components(np.ones((3, 2)), n_components=4)

    def test_fixed_pca_supports_deterministic_randomized_solver(self) -> None:
        images = np.random.default_rng(7).normal(size=(12, 10))
        first = fit_pca_components(
            images, n_components=3, svd_solver="randomized", random_state=7
        )
        second = fit_pca_components(
            images, n_components=3, svd_solver="randomized", random_state=7
        )
        np.testing.assert_allclose(first.components_, second.components_)

    def test_fixed_pca_supports_deterministic_randomized_solver(self) -> None:
        images = np.arange(120, dtype=np.float64).reshape(12, 10)
        first = fit_pca_components(
            images, n_components=3, svd_solver="randomized", random_state=7
        )
        second = fit_pca_components(
            images, n_components=3, svd_solver="randomized", random_state=7
        )
        np.testing.assert_allclose(first.components_, second.components_)


if __name__ == "__main__":
    unittest.main()
