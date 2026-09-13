import unittest

import numpy as np

from face_verification.models import (
    fit_and_score_classifier,
    make_verification_classifier,
)


class VerificationClassifierTests(unittest.TestCase):
    def test_all_classifier_families_produce_finite_scores(self) -> None:
        features = np.array([[0.0], [0.1], [1.0], [1.1]])
        labels = np.array([1, 1, 0, 0])
        for family in ("logistic_regression", "linear_svm", "rbf_svm"):
            with self.subTest(family=family):
                model = make_verification_classifier(family, c=1.0)
                train_scores, validation_scores = fit_and_score_classifier(
                    model, features, labels, features
                )
                self.assertTrue(np.all(np.isfinite(train_scores)))
                self.assertEqual(validation_scores.shape, (4,))

    def test_invalid_classifier_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_verification_classifier("unknown")
        with self.assertRaises(ValueError):
            make_verification_classifier("linear_svm", c=0.0)
        with self.assertRaises(ValueError):
            make_verification_classifier("rbf_svm", gamma=0.0)


if __name__ == "__main__":
    unittest.main()
