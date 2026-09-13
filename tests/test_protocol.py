import unittest

import numpy as np

from face_verification.protocol import make_balanced_pairs, split_by_identity


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


if __name__ == "__main__":
    unittest.main()
