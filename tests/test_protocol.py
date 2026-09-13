import unittest

import numpy as np

from face_verification.protocol import (
    make_balanced_pairs,
    make_identity_folds,
    split_by_identity,
)


class IdentitySplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(1, 41), 10)

    def test_orl_split_is_identity_disjoint_and_reproducible(self) -> None:
        first = split_by_identity(self.labels, seed=20260913)
        second = split_by_identity(self.labels, seed=20260913)

        self.assertEqual(first, second)
        self.assertEqual(
            tuple(
                map(
                    len,
                    (
                        first.train_identities,
                        first.validation_identities,
                        first.test_identities,
                    ),
                )
            ),
            (24, 8, 8),
        )
        self.assertEqual(
            tuple(
                map(
                    len,
                    (
                        first.train_indices,
                        first.validation_indices,
                        first.test_indices,
                    ),
                )
            ),
            (240, 80, 80),
        )

        identity_sets = [
            set(first.train_identities),
            set(first.validation_identities),
            set(first.test_identities),
        ]
        index_sets = [
            set(first.train_indices),
            set(first.validation_indices),
            set(first.test_indices),
        ]
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(identity_sets[left].isdisjoint(identity_sets[right]))
                self.assertTrue(index_sets[left].isdisjoint(index_sets[right]))

    def test_split_rejects_wrong_identity_total(self) -> None:
        with self.assertRaises(ValueError):
            split_by_identity(
                self.labels,
                train_identities=20,
                validation_identities=8,
                test_identities=8,
            )


class BalancedPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(1, 41), 10)
        self.split = split_by_identity(self.labels, seed=20260913)

    def test_pair_counts_and_pair_invariants(self) -> None:
        expected = {
            "train_indices": (1_080, 1_080, 2_160),
            "validation_indices": (360, 360, 720),
            "test_indices": (360, 360, 720),
        }
        for attribute, counts in expected.items():
            with self.subTest(split=attribute):
                pairs = make_balanced_pairs(
                    self.labels,
                    getattr(self.split, attribute),
                    seed=20261014,
                )
                self.assertEqual(
                    (pairs.positive_count, pairs.negative_count, len(pairs.records)),
                    counts,
                )
                pair_keys = {
                    (record.left_index, record.right_index) for record in pairs.records
                }
                self.assertEqual(len(pair_keys), len(pairs.records))
                for record in pairs.records:
                    self.assertLess(record.left_index, record.right_index)
                    self.assertEqual(
                        record.target,
                        int(self.labels[record.left_index] == self.labels[record.right_index]),
                    )

    def test_pair_sampling_is_reproducible(self) -> None:
        first = make_balanced_pairs(
            self.labels, self.split.validation_indices, seed=20261015
        )
        second = make_balanced_pairs(
            self.labels, self.split.validation_indices, seed=20261015
        )
        self.assertEqual(first, second)


class IdentityFoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.repeat(np.arange(1, 41), 10)
        self.outer = split_by_identity(self.labels, seed=20260913)

    def test_folds_are_reproducible_disjoint_and_cover_each_identity_once(self) -> None:
        first = make_identity_folds(
            self.labels, self.outer.train_indices, n_splits=5, seed=20261114
        )
        second = make_identity_folds(
            self.labels, self.outer.train_indices, n_splits=5, seed=20261114
        )
        self.assertEqual(first, second)
        self.assertEqual(sorted(len(fold.validation_identities) for fold in first), [4, 5, 5, 5, 5])

        seen_validation_identities: list[int] = []
        for fold in first:
            self.assertTrue(
                set(fold.train_identities).isdisjoint(fold.validation_identities)
            )
            self.assertTrue(
                set(fold.train_indices).isdisjoint(fold.validation_indices)
            )
            self.assertEqual(
                set(fold.train_indices) | set(fold.validation_indices),
                set(self.outer.train_indices),
            )
            seen_validation_identities.extend(fold.validation_identities)
        self.assertEqual(
            sorted(seen_validation_identities), sorted(self.outer.train_identities)
        )

    def test_invalid_fold_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_identity_folds(
                self.labels, self.outer.train_indices, n_splits=1, seed=20261114
            )


if __name__ == "__main__":
    unittest.main()
