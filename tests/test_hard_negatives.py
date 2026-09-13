import numpy as np

from scripts.analyze_orl_hard_negatives import (
    pair_records_for_identity_pairs,
    rank_identity_pairs,
)


def test_identity_pair_ranking_selects_closest_quarter() -> None:
    identities = tuple(range(8))
    centers = {identity: np.asarray([float(identity)]) for identity in identities}

    ranking = rank_identity_pairs(identities, centers)

    assert len(ranking) == 28
    assert sum(bool(row["is_hard_identity_pair"]) for row in ranking) == 7
    assert float(ranking[0]["centroid_distance"]) == 1.0


def test_identity_pair_expansion_uses_all_cross_image_combinations() -> None:
    labels = np.asarray([1] * 10 + [2] * 10 + [3] * 10)

    records = pair_records_for_identity_pairs(labels, {(1, 2), (2, 3)})

    assert len(records) == 200
    assert all(record.target == 0 for record in records)
    assert all(labels[record.left_index] != labels[record.right_index] for record in records)
