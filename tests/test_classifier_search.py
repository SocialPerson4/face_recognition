from scripts.search_orl_classifier_hyperparameters import (
    BOUNDARY_C_VALUES,
    BOUNDARY_GAMMA_VALUES,
    COARSE_C_VALUES,
    COARSE_GAMMA_VALUES,
    configurations,
)


def test_classifier_search_grids_have_expected_size() -> None:
    assert len(configurations(COARSE_C_VALUES, COARSE_GAMMA_VALUES)) == 42
    assert len(configurations(BOUNDARY_C_VALUES, BOUNDARY_GAMMA_VALUES)) == 42


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
