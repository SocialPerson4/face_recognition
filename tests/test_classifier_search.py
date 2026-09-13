from scripts.search_orl_classifier_hyperparameters import (
    BOUNDARY_C_VALUES,
    BOUNDARY_GAMMA_VALUES,
    COARSE_C_VALUES,
    COARSE_GAMMA_VALUES,
    SENTINEL_C_VALUES,
    SENTINEL_GAMMA_VALUES,
    configurations,
)


def test_classifier_search_grids_have_expected_size() -> None:
    assert len(configurations(COARSE_C_VALUES, COARSE_GAMMA_VALUES)) == 42
    assert len(configurations(BOUNDARY_C_VALUES, BOUNDARY_GAMMA_VALUES)) == 42
    assert len(configurations(SENTINEL_C_VALUES, SENTINEL_GAMMA_VALUES)) == 12


def test_boundary_search_overlaps_coarse_search() -> None:
    coarse = {
        config["config_id"]
        for config in configurations(COARSE_C_VALUES, COARSE_GAMMA_VALUES)
    }
    boundary = {
        config["config_id"]
        for config in configurations(BOUNDARY_C_VALUES, BOUNDARY_GAMMA_VALUES)
    }

    assert len(coarse & boundary) == 5


def test_sentinel_search_has_four_boundary_anchors() -> None:
    boundary = {
        config["config_id"]
        for config in configurations(BOUNDARY_C_VALUES, BOUNDARY_GAMMA_VALUES)
    }
    sentinel = {
        config["config_id"]
        for config in configurations(SENTINEL_C_VALUES, SENTINEL_GAMMA_VALUES)
    }

    assert len(boundary & sentinel) == 4
