import unittest

import numpy as np

from face_verification.selection import (
    select_configuration_one_standard_error,
    select_k_one_standard_error,
)


class OneStandardErrorSelectionTests(unittest.TestCase):
    def test_selects_smallest_candidate_within_best_standard_error(self) -> None:
        result = select_k_one_standard_error(
            np.array([20, 40, 80]),
            np.array([0.11, 0.10, 0.095]),
            np.array([0.02, 0.02, 0.03]),
            n_folds=5,
        )
        self.assertEqual(result.observed_best_k, 80)
        self.assertEqual(result.eligible_k, (40, 80))
        self.assertEqual(result.selected_k, 40)
        self.assertAlmostEqual(result.best_standard_error, 0.03 / np.sqrt(5))

    def test_tie_is_resolved_by_lower_dimension(self) -> None:
        result = select_k_one_standard_error(
            np.array([120, 80]),
            np.array([0.09, 0.09]),
            np.array([0.0, 0.0]),
            n_folds=5,
        )
        self.assertEqual(result.observed_best_k, 80)
        self.assertEqual(result.selected_k, 80)

    def test_invalid_statistics_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_k_one_standard_error(
                np.array([10, 20]),
                np.array([0.1, np.nan]),
                np.array([0.01, 0.02]),
                n_folds=5,
            )

    def test_configuration_selection_uses_declared_complexity_order(self) -> None:
        result = select_configuration_one_standard_error(
            np.array([0.10, 0.09, 0.095]),
            np.array([0.03, 0.03, 0.03]),
            np.array([[0.1, 1.0], [1.0, 0.1], [0.01, 10.0]]),
            n_folds=5,
        )
        self.assertEqual(result.observed_best_index, 1)
        self.assertEqual(result.eligible_indices, (0, 1, 2))
        self.assertEqual(result.selected_index, 2)


if __name__ == "__main__":
    unittest.main()
