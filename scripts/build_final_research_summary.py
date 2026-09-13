"""Build one machine-readable project summary from frozen result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orl",
        type=Path,
        default=Path("results/experiments/orl_final_test_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--hard-negatives",
        type=Path,
        default=Path("results/experiments/orl_hard_negative_proxy_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--lfw",
        type=Path,
        default=Path("results/experiments/lfw_frozen_classifier_transfer_8_1_1/summary.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/final/project_summary.json")
    )
    return parser.parse_args()


def build_summary(
    orl: dict[str, object],
    hard_negatives: dict[str, object],
    lfw: dict[str, object],
) -> dict[str, object]:
    orl_rows = {row["model"]: row for row in orl["results"]}
    hard_rows = {row["model"]: row for row in hard_negatives["model_results"]}
    lfw_rows = lfw["aggregate_test_metrics"]
    models = {}
    for model in MODEL_ORDER:
        orl_row = orl_rows[model]
        hard_row = hard_rows[model]
        models[model] = {
            "orl": {
                "development_eer": orl_row["validation_eer"],
                "final_test_eer": orl_row["test_eer"]["eer"],
                "final_test_roc_auc": orl_row["test_roc_auc"],
                "final_test_fmr": orl_row["test_rates_at_frozen_threshold"]["fmr"],
                "final_test_fnmr": orl_row["test_rates_at_frozen_threshold"]["fnmr"],
            },
            "orl_hard_negative_proxy": {
                "ordinary_fmr": hard_row["groups"]["ordinary"]["false_match_rate"],
                "hard_fmr": hard_row["groups"]["hard_identity_proxy"]["false_match_rate"],
                "risk_multiplier": hard_row["hard_to_ordinary_fmr_multiplier"],
            },
            "lfw_transfer": lfw_rows[model],
        }
    return {
        "artifact_type": "Final research synthesis from frozen experiment artifacts",
        "project_status": "sealed",
        "project_title": "Face verification under high identity similarity risk",
        "models": models,
        "main_findings": [
            "PCA distance achieved the best ORL final-test EER at 11.67%; the development-selected RBF advantage did not persist.",
            "Identity-level hard negative proxies increased frozen-threshold FMR from 0.14-0.57% to 5.86-6.86% across models.",
            "On identity-disjoint LFW folds, all ORL-frozen methods produced about 37-38% EER and above 90% FNMR at the strict operating point.",
            "The evidence supports a controlled-condition baseline and a high-similarity risk finding, not a deployable or twin-validated system.",
        ],
        "claim_boundary": {
            "twin_dataset_used": False,
            "deployable_access_control_claim": False,
            "lfw_hyperparameter_search_performed": False,
            "orl_test_reused_for_tuning": False,
        },
        "source_artifacts": {
            "orl": str(Path("results/experiments/orl_final_test_seed_20260913/summary.json")),
            "hard_negatives": str(Path("results/experiments/orl_hard_negative_proxy_seed_20260913/summary.json")),
            "lfw": str(Path("results/experiments/lfw_frozen_classifier_transfer_8_1_1/summary.json")),
        },
    }


def main() -> None:
    args = parse_args()
    summary = build_summary(
        json.loads(args.orl.read_text(encoding="utf-8")),
        json.loads(args.hard_negatives.read_text(encoding="utf-8")),
        json.loads(args.lfw.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["main_findings"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
