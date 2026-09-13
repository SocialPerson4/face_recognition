"""Compare ORL-frozen classifiers on the strict LFW 8/1/1 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from face_verification.data import load_lfw_preprocessed_images
from face_verification.features import (
    absolute_pair_differences,
    fit_pca_components,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.lfw_experiment import aggregate, fold_roles, targets, to_pair_set
from face_verification.lfw_protocol import parse_protocol
from face_verification.metrics import (
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    select_threshold_at_fmr,
)
from face_verification.models import fit_and_score_classifier, make_verification_classifier


MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
TARGET_FMR = 0.01
RANDOM_STATE = 20260913
EXPECTED_M4_SHA256 = "aeefff446ede1ebc9476988aad43ba2b6a1e76735f9d50fbb78d329501b1fa9c"
EXPECTED_DISTANCE_BASELINE_SHA256 = "ca7b67ba4a6f94a0712ae066c25b850e79c6705c69a1ec900c5d8a6986c1cead"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=Path("results/experiments/orl_frozen_validation_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--distance-baseline",
        type=Path,
        default=Path("results/experiments/lfw_pca_distance_8_1_1/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/lfw_frozen_classifier_transfer_8_1_1"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_distance_reproduction(
    baseline_row: dict[str, object],
    eer: float,
    auc: float,
    threshold: float,
) -> None:
    expected = (
        float(baseline_row["test_eer"]["eer"]),
        float(baseline_row["test_roc_auc"]),
        float(baseline_row["calibration_threshold_selection"]["rates"]["threshold"]),
    )
    actual = (eer, auc, threshold)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"distance baseline reproduction failed: expected {expected}, got {actual}"
        )


def main() -> None:
    args = parse_args()
    if sha256(args.frozen_summary) != EXPECTED_M4_SHA256:
        raise ValueError("ORL frozen configuration SHA-256 mismatch")
    if sha256(args.distance_baseline) != EXPECTED_DISTANCE_BASELINE_SHA256:
        raise ValueError("LFW distance baseline SHA-256 mismatch")
    frozen = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
    configurations = frozen["frozen_configurations"]
    distance_baseline = json.loads(args.distance_baseline.read_text(encoding="utf-8"))
    baseline_by_fold = {
        int(row["test_fold"]): row for row in distance_baseline["fold_results"]
    }
    _, official_pairs = parse_protocol(args.data_dir / "pairs.txt")
    relative_paths = sorted(
        {
            image
            for pair in official_pairs
            for image in (pair.left_image, pair.right_image)
        }
    )
    image_index = {path: index for index, path in enumerate(relative_paths)}
    images = load_lfw_preprocessed_images(
        args.data_dir / "lfw_funneled", relative_paths
    )

    fold_results: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for test_fold in range(1, 11):
        train_folds, calibration_fold, _ = fold_roles(test_fold)
        protocol_groups = {
            "train": [pair for pair in official_pairs if pair.fold in train_folds],
            "calibration": [
                pair for pair in official_pairs if pair.fold == calibration_fold
            ],
            "test": [pair for pair in official_pairs if pair.fold == test_fold],
        }
        pair_sets = {
            name: to_pair_set(pairs, image_index)
            for name, pairs in protocol_groups.items()
        }
        labels = {name: targets(pairs) for name, pairs in pair_sets.items()}
        train_image_indices = sorted(
            {
                image_index[image]
                for pair in protocol_groups["train"]
                for image in (pair.left_image, pair.right_image)
            }
        )
        pca = fit_pca_components(
            images[train_image_indices],
            n_components=80,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        features = transform_images(pca, images)
        scores: dict[str, dict[str, np.ndarray]] = {
            "pca_distance": {
                name: negative_euclidean_pair_scores(features, pair_sets[name])
                for name in ("calibration", "test")
            }
        }
        train_features = absolute_pair_differences(features, pair_sets["train"])
        evaluation_features = np.vstack(
            (
                absolute_pair_differences(features, pair_sets["calibration"]),
                absolute_pair_differences(features, pair_sets["test"]),
            )
        )
        for family in MODEL_ORDER[1:]:
            config = configurations[family]
            gamma = config["gamma"] if config["gamma"] is not None else "scale"
            classifier = make_verification_classifier(
                family,
                c=float(config["c"]),
                gamma=gamma,
                seed=RANDOM_STATE,
            )
            _, evaluation_scores = fit_and_score_classifier(
                classifier,
                train_features,
                labels["train"],
                evaluation_features,
            )
            calibration_count = len(pair_sets["calibration"].records)
            scores[family] = {
                "calibration": evaluation_scores[:calibration_count],
                "test": evaluation_scores[calibration_count:],
            }

        thresholds: dict[str, float] = {}
        for model in MODEL_ORDER:
            selection = select_threshold_at_fmr(
                labels["calibration"],
                scores[model]["calibration"],
                target_fmr=TARGET_FMR,
            )
            thresholds[model] = selection.rates.threshold
            test_eer = equal_error_rate(labels["test"], scores[model]["test"])
            test_auc = roc_auc(labels["test"], scores[model]["test"])
            test_rates = rates_at_threshold(
                labels["test"], scores[model]["test"], thresholds[model]
            )
            if model == "pca_distance":
                assert_distance_reproduction(
                    baseline_by_fold[test_fold],
                    test_eer.eer,
                    test_auc,
                    thresholds[model],
                )
            fold_results.append(
                {
                    "test_fold": test_fold,
                    "calibration_fold": calibration_fold,
                    "train_folds": list(train_folds),
                    "model": model,
                    "configuration": configurations[model],
                    "train_pair_count": len(pair_sets["train"].records),
                    "train_unique_image_count": len(train_image_indices),
                    "calibration_threshold_selection": asdict(selection),
                    "test_eer": asdict(test_eer),
                    "test_roc_auc": test_auc,
                    "test_rates_at_calibration_threshold": asdict(test_rates),
                }
            )
        for pair_id, pair in enumerate(protocol_groups["test"]):
            row: dict[str, object] = {
                "test_fold": test_fold,
                "pair_id": pair_id,
                "left_image": pair.left_image,
                "right_image": pair.right_image,
                "target": pair.target,
            }
            for model in MODEL_ORDER:
                score = float(scores[model]["test"][pair_id])
                row[f"{model}_score"] = score
                row[f"{model}_threshold"] = thresholds[model]
                row[f"{model}_accepted"] = int(score >= thresholds[model])
            score_rows.append(row)

    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODEL_ORDER:
        rows = [row for row in fold_results if row["model"] == model]
        aggregates[model] = {
            "eer": aggregate([float(row["test_eer"]["eer"]) for row in rows]),
            "roc_auc": aggregate([float(row["test_roc_auc"]) for row in rows]),
            "fmr": aggregate(
                [float(row["test_rates_at_calibration_threshold"]["fmr"]) for row in rows]
            ),
            "fnmr": aggregate(
                [float(row["test_rates_at_calibration_threshold"]["fnmr"]) for row in rows]
            ),
            "accuracy": aggregate(
                [float(row["test_rates_at_calibration_threshold"]["accuracy"]) for row in rows]
            ),
        }
    distance_eer_by_fold = {
        int(row["test_fold"]): float(row["test_eer"]["eer"])
        for row in fold_results
        if row["model"] == "pca_distance"
    }
    comparisons = {}
    for model in MODEL_ORDER[1:]:
        rows = [row for row in fold_results if row["model"] == model]
        deltas = [
            float(row["test_eer"]["eer"]) - distance_eer_by_fold[int(row["test_fold"])]
            for row in rows
        ]
        comparisons[model] = {
            "eer_difference_vs_distance": aggregate(deltas),
            "folds_better_than_distance": int(sum(delta < 0.0 for delta in deltas)),
            "folds_equal_to_distance": int(
                sum(bool(np.isclose(delta, 0.0)) for delta in deltas)
            ),
            "folds_worse_than_distance": int(sum(delta > 0.0 for delta in deltas)),
        }

    summary = {
        "artifact_type": "LFW ORL-frozen classifier transfer comparison",
        "protocol": "10 rotating identity-disjoint 8/1/1 rounds",
        "target_fmr": TARGET_FMR,
        "orl_frozen_summary_sha256": EXPECTED_M4_SHA256,
        "lfw_distance_baseline_sha256": EXPECTED_DISTANCE_BASELINE_SHA256,
        "distance_reproduction": "passed for EER, AUC, and calibration threshold in all folds",
        "hyperparameter_search_on_lfw": False,
        "configurations": configurations,
        "fold_results": fold_results,
        "aggregate_test_metrics": aggregates,
        "paired_comparison_to_distance": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = (
            "test_fold",
            "calibration_fold",
            "model",
            "test_eer",
            "test_roc_auc",
            "test_fmr",
            "test_fnmr",
            "test_accuracy",
            "false_accepts",
            "false_rejects",
        )
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in fold_results:
            rates = row["test_rates_at_calibration_threshold"]
            writer.writerow(
                {
                    "test_fold": row["test_fold"],
                    "calibration_fold": row["calibration_fold"],
                    "model": row["model"],
                    "test_eer": row["test_eer"]["eer"],
                    "test_roc_auc": row["test_roc_auc"],
                    "test_fmr": rates["fmr"],
                    "test_fnmr": rates["fnmr"],
                    "test_accuracy": rates["accuracy"],
                    "false_accepts": rates["false_accepts"],
                    "false_rejects": rates["false_rejects"],
                }
            )
    with (args.output_dir / "test_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(score_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(score_rows)
    print(json.dumps(aggregates, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
