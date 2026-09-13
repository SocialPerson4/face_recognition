"""Evaluate preregistered ORL candidates on the development validation identities."""

from __future__ import annotations

import argparse
import csv
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
from face_verification.metrics import equal_error_rate, roc_auc, select_threshold_at_fmr
from face_verification.models import fit_and_score_classifier, make_verification_classifier
from face_verification.protocol import PairSet, make_balanced_pairs, split_by_identity


DEFAULT_SEED = 20260913
DEFAULT_K = 80
TRAIN_PAIR_SEED_OFFSET = 101
VALIDATION_PAIR_SEED_OFFSET = 102
MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
FROZEN_CONFIGURATIONS: dict[str, dict[str, object]] = {
    "pca_distance": {"k": 80, "classifier": None},
    "logistic_regression": {"k": 80, "c": 0.00001, "gamma": None},
    "linear_svm": {"k": 80, "c": 0.00001, "gamma": None},
    "rbf_svm": {"k": 80, "c": 0.001, "gamma": 0.001},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/orl_frozen_validation_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    return parser.parse_args()


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def metrics_for_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    return {
        "eer": asdict(equal_error_rate(y_true, scores)),
        "roc_auc": roc_auc(y_true, scores),
    }


def write_scores(
    path: Path,
    pairs: PairSet,
    scores_by_model: dict[str, np.ndarray],
) -> None:
    fieldnames = [
        "pair_id",
        "left_index",
        "right_index",
        "target",
        *(f"{model}_score" for model in MODEL_ORDER),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for pair_id, record in enumerate(pairs.records):
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "left_index": record.left_index,
                    "right_index": record.right_index,
                    "target": record.target,
                    **{
                        f"{model}_score": float(scores_by_model[model][pair_id])
                        for model in MODEL_ORDER
                    },
                }
            )


def main() -> None:
    args = parse_args()
    dataset = load_orl(args.data_dir)
    split = split_by_identity(dataset.labels, seed=args.seed)
    train_pairs = make_balanced_pairs(
        dataset.labels,
        split.train_indices,
        seed=args.seed + TRAIN_PAIR_SEED_OFFSET,
    )
    validation_pairs = make_balanced_pairs(
        dataset.labels,
        split.validation_indices,
        seed=args.seed + VALIDATION_PAIR_SEED_OFFSET,
    )
    y_train = targets(train_pairs)
    y_validation = targets(validation_pairs)

    pca = fit_pca_components(
        dataset.images[list(split.train_indices)], n_components=DEFAULT_K
    )
    pca_features = np.full(
        (len(dataset.images), DEFAULT_K), np.nan, dtype=np.float64
    )
    development_indices = list(split.train_indices + split.validation_indices)
    pca_features[development_indices] = transform_images(
        pca, dataset.images[development_indices]
    )

    train_scores: dict[str, np.ndarray] = {
        "pca_distance": negative_euclidean_pair_scores(pca_features, train_pairs)
    }
    validation_scores: dict[str, np.ndarray] = {
        "pca_distance": negative_euclidean_pair_scores(
            pca_features, validation_pairs
        )
    }
    train_features = absolute_pair_differences(pca_features, train_pairs)
    validation_features = absolute_pair_differences(pca_features, validation_pairs)
    for family in MODEL_ORDER[1:]:
        config = FROZEN_CONFIGURATIONS[family]
        gamma = config["gamma"] if config["gamma"] is not None else "scale"
        model = make_verification_classifier(
            family,
            c=float(config["c"]),
            gamma=gamma,
            seed=args.seed,
        )
        train_scores[family], validation_scores[family] = fit_and_score_classifier(
            model, train_features, y_train, validation_features
        )

    results: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        working_point = select_threshold_at_fmr(
            y_validation,
            validation_scores[model],
            target_fmr=args.target_fmr,
        )
        results.append(
            {
                "model": model,
                "configuration": FROZEN_CONFIGURATIONS[model],
                "train": metrics_for_scores(y_train, train_scores[model]),
                "validation": {
                    **metrics_for_scores(y_validation, validation_scores[model]),
                    "working_point": asdict(working_point),
                },
            }
        )

    ranking = sorted(
        results,
        key=lambda row: (
            float(row["validation"]["eer"]["eer"]),  # type: ignore[index]
            -float(row["validation"]["roc_auc"]),  # type: ignore[index]
            MODEL_ORDER.index(str(row["model"])),
        ),
    )
    for rank, row in enumerate(ranking, start=1):
        row["development_rank"] = rank

    summary = {
        "artifact_type": "ORL frozen-candidate development-validation results",
        "dataset": "ORL Database of Faces",
        "split_seed": args.seed,
        "pca_dimensions": DEFAULT_K,
        "pca_fit_scope": "outer-training identities only",
        "train_identities": list(split.train_identities),
        "validation_identities": list(split.validation_identities),
        "test_identity_count": len(split.test_identities),
        "test_status": (
            "test pixels were not fitted, transformed, paired, scored, or inspected"
        ),
        "validation_reuse_disclosure": (
            "these validation identities were viewed during the earlier M2B distance "
            "baseline; they were not used in M3 k/C/gamma searches"
        ),
        "train_pair_seed": args.seed + TRAIN_PAIR_SEED_OFFSET,
        "validation_pair_seed": args.seed + VALIDATION_PAIR_SEED_OFFSET,
        "train_pair_count": len(train_pairs.records),
        "validation_pair_count": len(validation_pairs.records),
        "primary_ranking": "validation EER, then AUC, then preregistered complexity",
        "target_fmr": args.target_fmr,
        "frozen_configurations": FROZEN_CONFIGURATIONS,
        "development_winner": str(ranking[0]["model"]),
        "winner_threshold_for_final_test": ranking[0]["validation"][
            "working_point"
        ],
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scores(args.output_dir / "train_scores.csv", train_pairs, train_scores)
    write_scores(
        args.output_dir / "validation_scores.csv",
        validation_pairs,
        validation_scores,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
