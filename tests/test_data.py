import unittest

from face_verification.data import theoretical_pair_counts


class PairCountTests(unittest.TestCase):
    def test_orl_theoretical_pair_counts(self) -> None:
        self.assertEqual(theoretical_pair_counts(40, 10), (1_800, 78_000))

    def test_pair_counts_reject_invalid_inputs(self) -> None:
        for people, images in [(0, 10), (40, 1)]:
            with self.subTest(people=people, images=images):
                with self.assertRaises(ValueError):
                    theoretical_pair_counts(people, images)


if __name__ == "__main__":
    unittest.main()
