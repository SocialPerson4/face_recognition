from scripts.evaluate_orl_frozen_candidates import (
    FROZEN_CONFIGURATIONS,
    MODEL_ORDER,
)


def test_all_frozen_models_have_a_configuration() -> None:
    assert tuple(FROZEN_CONFIGURATIONS) == MODEL_ORDER
    assert {config["k"] for config in FROZEN_CONFIGURATIONS.values()} == {80}


def test_frozen_classifier_parameters_match_preregistration() -> None:
    assert FROZEN_CONFIGURATIONS["logistic_regression"]["c"] == 0.00001
    assert FROZEN_CONFIGURATIONS["linear_svm"]["c"] == 0.00001
    assert FROZEN_CONFIGURATIONS["rbf_svm"]["c"] == 0.001
    assert FROZEN_CONFIGURATIONS["rbf_svm"]["gamma"] == 0.001
