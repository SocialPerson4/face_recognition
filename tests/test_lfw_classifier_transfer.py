import pytest

from scripts.run_lfw_frozen_classifier_transfer import assert_distance_reproduction


def test_distance_reproduction_accepts_matching_saved_metrics() -> None:
    row = {
        "test_eer": {"eer": 0.2},
        "test_roc_auc": 0.8,
        "calibration_threshold_selection": {"rates": {"threshold": -3.0}},
    }

    assert_distance_reproduction(row, 0.2, 0.8, -3.0)


def test_distance_reproduction_rejects_changed_result() -> None:
    row = {
        "test_eer": {"eer": 0.2},
        "test_roc_auc": 0.8,
        "calibration_threshold_selection": {"rates": {"threshold": -3.0}},
    }

    with pytest.raises(ValueError, match="reproduction failed"):
        assert_distance_reproduction(row, 0.3, 0.8, -3.0)
