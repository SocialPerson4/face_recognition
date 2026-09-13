import unittest

import numpy as np

from face_verification.metrics import (
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    select_threshold_at_fmr,
)


class VerificationMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.array([1, 1, 0, 0])
        self.perfect_scores = np.array([0.9, 0.7, 0.6, 0.2])

    def test_rates_at_perfect_threshold_are_hand_checkable(self) -> None:
        rates = rates_at_threshold(self.labels, self.perfect_scores, 0.65)
        self.assertEqual(
            (
                rates.true_accepts,
                rates.false_rejects,
                rates.false_accepts,
                rates.true_rejects,
            ),
            (2, 0, 0, 2),
        )
        self.assertEqual((rates.fmr, rates.fnmr, rates.accuracy), (0.0, 0.0, 1.0))

    def test_one_false_accept_is_counted(self) -> None:
        rates = rates_at_threshold(self.labels, self.perfect_scores, 0.5)
        self.assertEqual((rates.false_accepts, rates.true_rejects), (1, 1))
        self.assertEqual((rates.fmr, rates.fnmr, rates.accuracy), (0.5, 0.0, 0.75))

    def test_perfect_scores_have_zero_eer_and_unit_auc(self) -> None:
        eer = equal_error_rate(self.labels, self.perfect_scores)
        self.assertEqual(eer.eer, 0.0)
        self.assertEqual(roc_auc(self.labels, self.perfect_scores), 1.0)

    def test_overlapping_scores_have_expected_eer_and_auc(self) -> None:
        scores = np.array([0.9, 0.4, 0.8, 0.3])
        eer = equal_error_rate(self.labels, scores)
        self.assertAlmostEqual(eer.eer, 0.5)
        self.assertAlmostEqual(eer.threshold, 0.8)
        self.assertAlmostEqual(roc_auc(self.labels, scores), 0.75)

    def test_threshold_selection_obeys_fmr_constraint(self) -> None:
        selection = select_threshold_at_fmr(
            self.labels, self.perfect_scores, target_fmr=0.0
        )
        self.assertAlmostEqual(selection.rates.threshold, 0.7)
        self.assertEqual((selection.rates.fmr, selection.rates.fnmr), (0.0, 0.0))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc(np.array([1, 1]), np.array([0.9, 0.8]))
        with self.assertRaises(ValueError):
            rates_at_threshold(
                np.array([1, 0]), np.array([0.9, np.nan]), threshold=0.5
            )
        with self.assertRaises(ValueError):
            select_threshold_at_fmr(
                self.labels, self.perfect_scores, target_fmr=1.1
            )


if __name__ == "__main__":
    unittest.main()
