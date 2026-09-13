import pytest

from face_verification.lfw_experiment import aggregate, fold_roles


def test_fold_roles_rotate_calibration_after_test() -> None:
    train, calibration, test = fold_roles(10)

    assert calibration == 1
    assert test == 10
    assert train == (2, 3, 4, 5, 6, 7, 8, 9)


def test_fold_roles_reject_invalid_fold() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        fold_roles(0)


def test_aggregate_uses_sample_standard_deviation() -> None:
    result = aggregate([1.0, 2.0, 3.0])

    assert result["mean"] == 2.0
    assert result["standard_deviation"] == 1.0
