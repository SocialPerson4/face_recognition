"""Run the one-time ORL final test with frozen models and thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from face_verification.data import load_orl
from face_verification.features import (
    absolute_pair_differences,
    fit_pca_components,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.metrics import equal_error_rate, rates_at_threshold, roc_auc
from face_verification.models import fit_and_score_classifier, make_verification_classifier
from face_verification.protocol import PairSet, make_balanced_pairs, split_by_identity


DEFAULT_SEED = 20260913
TRAIN_PAIR_SEED_OFFSET = 101
TEST_PAIR_SEED_OFFSET = 103
MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
EXPECTED_M4_SHA256 = "aeefff446ede1ebc9476988aad43ba2b6a1e76735f9d50fbb78d329501b1fa9c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--frozen-summary",
        type=Path,
        default=Path(
            "results/experiments/orl_frozen_validation_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/orl_final_test_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def write_test_scores(
    path: Path,
    pairs: PairSet,
    scores_by_model: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    fields = ["pair_id", "left_index", "right_index", "target"]
    for model in MODEL_ORDER:
        fields.extend((f"{model}_score", f"{model}_accepted"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pair_id, record in enumerate(pairs.records):
            row: dict[str, object] = {
                "pair_id": pair_id,
                "left_index": record.left_index,
                "right_index": record.right_index,
                "target": record.target,
            }
            for model in MODEL_ORDER:
                score = float(scores_by_model[model][pair_id])
                row[f"{model}_score"] = score
                row[f"{model}_accepted"] = int(score >= thresholds[model])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    frozen_hash = sha256(args.frozen_summary)
    if frozen_hash != EXPECTED_M4_SHA256:
        raise ValueError(
            "frozen M4 summary hash mismatch: "
            f"expected {EXPECTED_M4_SHA256}, found {frozen_hash}"
        )
    frozen = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
    if frozen["development_winner"] != "rbf_svm":
        raise ValueError("unexpected frozen development winner")

    configurations = frozen["frozen_configurations"]
    validation_results = {row["model"]: row for row in frozen["results"]}
    thresholds = {
        model: float(
            validation_results[model]["validation"]["working_point"]["rates"][
                "threshold"
            ]
        )
        for model in MODEL_ORDER
    }

    dataset = load_orl(args.data_dir)
    split = split_by_identity(dataset.labels, seed=args.seed)
    expected_test_identities = (2, 10, 11, 16, 18, 19, 30, 40)
    if split.test_identities != expected_test_identities:
        raise ValueError("test identities do not match the preregistered split")
    train_pairs = make_balanced_pairs(
        dataset.labels,
        split.train_indices,
        seed=args.seed + TRAIN_PAIR_SEED_OFFSET,
    )
    test_pairs = make_balanced_pairs(
        dataset.labels,
        split.test_indices,
        seed=args.seed + TEST_PAIR_SEED_OFFSET,
    )
    y_train = targets(train_pairs)
    y_test = targets(test_pairs)

    k_values = {int(config["k"]) for config in configurations.values()}
    if k_values != {80}:
        raise ValueError(f"unexpected frozen PCA dimensions: {sorted(k_values)}")
    pca = fit_pca_components(
        dataset.images[list(split.train_indices)], n_components=80
    )
    pca_features = np.full((len(dataset.images), 80), np.nan, dtype=np.float64)
    scored_indices = list(split.train_indices + split.test_indices)
    pca_features[scored_indices] = transform_images(
        pca, dataset.images[scored_indices]
    )

    train_scores: dict[str, np.ndarray] = {
        "pca_distance": negative_euclidean_pair_scores(pca_features, train_pairs)
    }
    test_scores: dict[str, np.ndarray] = {
        "pca_distance": negative_euclidean_pair_scores(pca_features, test_pairs)
    }
    train_features = absolute_pair_differences(pca_features, train_pairs)
    test_features = absolute_pair_differences(pca_features, test_pairs)
    for family in MODEL_ORDER[1:]:
        config = configurations[family]
        gamma = config["gamma"] if config["gamma"] is not None else "scale"
        model = make_verification_classifier(
            family,
            c=float(config["c"]),
            gamma=gamma,
            seed=args.seed,
        )
        train_scores[family], test_scores[family] = fit_and_score_classifier(
            model, train_features, y_train, test_features
        )

    results: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        test_eer = equal_error_rate(y_test, test_scores[model])
        fixed_rates = rates_at_threshold(y_test, test_scores[model], thresholds[model])
        validation = validation_results[model]["validation"]
        results.append(
            {
                "model": model,
                "configuration": configurations[model],
                "validation_eer": validation["eer"]["eer"],
                "validation_roc_auc": validation["roc_auc"],
                "frozen_validation_threshold": thresholds[model],
                "test_eer": asdict(test_eer),
                "test_roc_auc": roc_auc(y_test, test_scores[model]),
                "test_rates_at_frozen_threshold": asdict(fixed_rates),
            }
        )

    summary = {
        "artifact_type": "ORL one-time frozen final-test results",
        "dataset": "ORL Database of Faces",
        "split_seed": args.seed,
        "frozen_summary": str(args.frozen_summary),
        "frozen_summary_sha256": frozen_hash,
        "predeclared_development_winner": frozen["development_winner"],
        "post_test_model_selection": (
            "prohibited; test results are reporting evidence only"
        ),
        "training_identities": list(split.train_identities),
        "test_identities": list(split.test_identities),
        "train_pair_seed": args.seed + TRAIN_PAIR_SEED_OFFSET,
        "test_pair_seed": args.seed + TEST_PAIR_SEED_OFFSET,
        "test_positive_pairs": test_pairs.positive_count,
        "test_negative_pairs": test_pairs.negative_count,
        "test_pair_count": len(test_pairs.records),
        "threshold_source": "M4 development validation; never adjusted on test",
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_test_scores(
        args.output_dir / "test_scores.csv", test_pairs, test_scores, thresholds
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
