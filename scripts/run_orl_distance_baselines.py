"""Run development-only raw-pixel and PCA distance baselines on ORL."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from face_verification.data import load_orl
from face_verification.features import (
    fit_pca,
    image_matrix,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.metrics import (
    equal_error_rate,
    roc_auc,
    select_threshold_at_fmr,
)
from face_verification.protocol import PairSet, make_balanced_pairs, split_by_identity


DEFAULT_SEED = 20260913
PAIR_SEED_OFFSETS = {"train": 101, "validation": 102}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/orl_distance_baseline_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--retained-variance", type=float, default=0.95)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    return parser.parse_args()


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def score_summary(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    eer = equal_error_rate(y_true, scores)
    return {
        "pair_count": int(len(y_true)),
        "positive_pairs": int(np.sum(y_true == 1)),
        "negative_pairs": int(np.sum(y_true == 0)),
        "eer": asdict(eer),
        "roc_auc": roc_auc(y_true, scores),
    }


def write_scores(
    path: Path,
    pairs: PairSet,
    raw_scores: np.ndarray,
    pca_scores: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["pair_id", "left_index", "right_index", "target", "raw_score", "pca_score"]
        )
        for pair_id, (record, raw_score, pca_score) in enumerate(
            zip(pairs.records, raw_scores, pca_scores)
        ):
            writer.writerow(
                [
                    pair_id,
                    record.left_index,
                    record.right_index,
                    record.target,
                    float(raw_score),
                    float(pca_score),
                ]
            )


def main() -> None:
    args = parse_args()
    dataset = load_orl(args.data_dir)
    split = split_by_identity(dataset.labels, seed=args.seed)
    pairs = {
        "train": make_balanced_pairs(
            dataset.labels,
            split.train_indices,
            seed=args.seed + PAIR_SEED_OFFSETS["train"],
        ),
        "validation": make_balanced_pairs(
            dataset.labels,
            split.validation_indices,
            seed=args.seed + PAIR_SEED_OFFSETS["validation"],
        ),
    }

    raw_features = image_matrix(dataset.images)
    pca = fit_pca(
        dataset.images[list(split.train_indices)],
        retained_variance=args.retained_variance,
    )
    development_indices = np.asarray(
        sorted(split.train_indices + split.validation_indices), dtype=np.int64
    )
    pca_features = np.full(
        (len(dataset.images), int(pca.n_components_)), np.nan, dtype=np.float64
    )
    pca_features[development_indices] = transform_images(
        pca, dataset.images[development_indices]
    )

    scores: dict[str, dict[str, np.ndarray]] = {"raw_pixels": {}, "pca_95": {}}
    labels: dict[str, np.ndarray] = {}
    for split_name, pair_set in pairs.items():
        labels[split_name] = targets(pair_set)
        scores["raw_pixels"][split_name] = negative_euclidean_pair_scores(
            raw_features, pair_set
        )
        scores["pca_95"][split_name] = negative_euclidean_pair_scores(
            pca_features, pair_set
        )

    summary: dict[str, object] = {
        "artifact_type": "ORL development distance-baseline results",
        "dataset": "ORL Database of Faces",
        "split_seed": args.seed,
        "score_definition": "negative Euclidean distance; larger means more similar",
        "target_fmr": args.target_fmr,
        "test_status": (
            "test pairs were not constructed or scored; test images did not enter "
            "PCA fitting or transformation"
        ),
        "warning": (
            "Validation selects each threshold and is development evidence; "
            "it is not an untouched final-test estimate."
        ),
        "models": {},
    }

    model_metadata = {
        "raw_pixels": {
            "feature_dimensions": int(raw_features.shape[1]),
            "preprocessing": "flatten 112x92 pixels already scaled to [0,1]",
        },
        "pca_95": {
            "feature_dimensions": int(pca.n_components_),
            "requested_retained_variance": args.retained_variance,
            "actual_retained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "fit_images": len(split.train_indices),
            "preprocessing": "PCA fitted only on outer-training identities; no whitening",
        },
    }

    for model_name in ("raw_pixels", "pca_95"):
        train_result = score_summary(
            labels["train"], scores[model_name]["train"]
        )
        validation_result = score_summary(
            labels["validation"], scores[model_name]["validation"]
        )
        threshold_selection = select_threshold_at_fmr(
            labels["validation"],
            scores[model_name]["validation"],
            target_fmr=args.target_fmr,
        )
        validation_result["selected_working_point"] = {
            "selection_data": "outer validation",
            **asdict(threshold_selection),
        }
        summary["models"][model_name] = {
            **model_metadata[model_name],
            "train": train_result,
            "validation": validation_result,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation"):
        write_scores(
            args.output_dir / f"{split_name}_scores.csv",
            pairs[split_name],
            scores["raw_pixels"][split_name],
            scores["pca_95"][split_name],
        )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
