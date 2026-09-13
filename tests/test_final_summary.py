from scripts.build_final_research_summary import build_summary


def test_final_summary_keeps_all_three_evidence_stages() -> None:
    models = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
    orl = {
        "results": [
            {
                "model": model,
                "validation_eer": 0.2,
                "test_eer": {"eer": 0.1},
                "test_roc_auc": 0.9,
                "test_rates_at_frozen_threshold": {"fmr": 0.01, "fnmr": 0.4},
            }
            for model in models
        ]
    }
    hard = {
        "model_results": [
            {
                "model": model,
                "groups": {
                    "ordinary": {"false_match_rate": 0.01},
                    "hard_identity_proxy": {"false_match_rate": 0.05},
                },
                "hard_to_ordinary_fmr_multiplier": 5.0,
            }
            for model in models
        ]
    }
    lfw = {
        "aggregate_test_metrics": {
            model: {"eer": {"mean": 0.4}} for model in models
        }
    }

    summary = build_summary(orl, hard, lfw)

    assert summary["project_status"] == "sealed"
    assert set(summary["models"]) == set(models)
    assert summary["models"]["pca_distance"]["orl"]["final_test_eer"] == 0.1
    assert summary["models"]["rbf_svm"]["orl_hard_negative_proxy"]["hard_fmr"] == 0.05
    assert summary["claim_boundary"]["twin_dataset_used"] is False
