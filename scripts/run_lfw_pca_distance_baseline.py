"""Run the frozen 10-round LFW PCA-distance transfer baseline."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from face_verification.data import (
    LFW_CROP_BOX,
    LFW_OUTPUT_SIZE,
    load_lfw_preprocessed_images,
)
from face_verification.features import (
    fit_pca_components,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.lfw_protocol import ProtocolPair, parse_protocol
from face_verification.metrics import (
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    select_threshold_at_fmr,
)
from face_verification.protocol import PairRecord, PairSet


PCA_COMPONENTS = 80
TARGET_FMR = 0.01
RANDOM_STATE = 20260913


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/lfw_pca_distance_8_1_1"),
    )
    return parser.parse_args()


def fold_roles(test_fold: int) -> tuple[tuple[int, ...], int, int]:
    if test_fold not in range(1, 11):
        raise ValueError("test_fold must lie between 1 and 10")
    calibration_fold = test_fold % 10 + 1
    train_folds = tuple(
        fold for fold in range(1, 11) if fold not in {test_fold, calibration_fold}
    )
    return train_folds, calibration_fold, test_fold


def to_pair_set(
    pairs: list[ProtocolPair], image_index: dict[str, int]
) -> PairSet:
    return PairSet(
        records=tuple(
            PairRecord(
                left_index=image_index[pair.left_image],
                right_index=image_index[pair.right_image],
                target=pair.target,
            )
            for pair in pairs
        )
    )


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
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
        train_protocol = [pair for pair in official_pairs if pair.fold in train_folds]
        calibration_protocol = [
            pair for pair in official_pairs if pair.fold == calibration_fold
        ]
        test_protocol = [pair for pair in official_pairs if pair.fold == test_fold]
        train_image_indices = sorted(
            {
                image_index[image]
                for pair in train_protocol
                for image in (pair.left_image, pair.right_image)
            }
        )
        pca = fit_pca_components(
            images[train_image_indices],
            n_components=PCA_COMPONENTS,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        evaluation_indices = sorted(
            {
                image_index[image]
                for pair in (*calibration_protocol, *test_protocol)
                for image in (pair.left_image, pair.right_image)
            }
        )
        features = np.full((len(images), PCA_COMPONENTS), np.nan, dtype=np.float64)
        features[evaluation_indices] = transform_images(pca, images[evaluation_indices])
        calibration_pairs = to_pair_set(calibration_protocol, image_index)
        test_pairs = to_pair_set(test_protocol, image_index)
        y_calibration = targets(calibration_pairs)
        y_test = targets(test_pairs)
        calibration_scores = negative_euclidean_pair_scores(
            features, calibration_pairs
        )
        test_scores = negative_euclidean_pair_scores(features, test_pairs)
        threshold_selection = select_threshold_at_fmr(
            y_calibration, calibration_scores, target_fmr=TARGET_FMR
        )
        test_eer = equal_error_rate(y_test, test_scores)
        test_rates = rates_at_threshold(
            y_test, test_scores, threshold_selection.rates.threshold
        )
        fold_results.append(
            {
                "test_fold": test_fold,
                "calibration_fold": calibration_fold,
                "train_folds": list(train_folds),
                "train_pair_count": len(train_protocol),
                "train_unique_image_count": len(train_image_indices),
                "pca_explained_variance": float(
                    np.sum(pca.explained_variance_ratio_)
                ),
                "calibration_pair_count": len(calibration_protocol),
                "calibration_threshold_selection": asdict(threshold_selection),
                "test_pair_count": len(test_protocol),
                "test_eer": asdict(test_eer),
                "test_roc_auc": roc_auc(y_test, test_scores),
                "test_rates_at_calibration_threshold": asdict(test_rates),
            }
        )
        for pair_id, (pair, score) in enumerate(zip(test_protocol, test_scores)):
            score_rows.append(
                {
                    "test_fold": test_fold,
                    "pair_id": pair_id,
                    "left_image": pair.left_image,
                    "right_image": pair.right_image,
                    "target": pair.target,
                    "score": float(score),
                    "calibration_threshold": threshold_selection.rates.threshold,
                    "accepted": int(score >= threshold_selection.rates.threshold),
                }
            )

    metric_values = {
        "eer": [float(row["test_eer"]["eer"]) for row in fold_results],
        "roc_auc": [float(row["test_roc_auc"]) for row in fold_results],
        "fmr": [
            float(row["test_rates_at_calibration_threshold"]["fmr"])
            for row in fold_results
        ],
        "fnmr": [
            float(row["test_rates_at_calibration_threshold"]["fnmr"])
            for row in fold_results
        ],
        "accuracy": [
            float(row["test_rates_at_calibration_threshold"]["accuracy"])
            for row in fold_results
        ],
    }
    summary = {
        "artifact_type": "LFW frozen PCA-distance 8/1/1 transfer baseline",
        "dataset": "LFW Funneled",
        "protocol": "10 rotating rounds: 8 train folds, 1 threshold-calibration fold, 1 test fold",
        "data_isolation": "zero identity, image, and exact-pair overlap across official folds",
        "development_files_used": False,
        "preprocessing": {
            "color": "grayscale",
            "crop_box_left_top_right_bottom": list(LFW_CROP_BOX),
            "output_width_height": list(LFW_OUTPUT_SIZE),
            "pixel_range": [0.0, 1.0],
        },
        "model": {
            "family": "negative Euclidean distance in PCA space",
            "pca_components": PCA_COMPONENTS,
            "pca_solver": "randomized",
            "pca_random_state": RANDOM_STATE,
            "pca_whiten": False,
        },
        "target_fmr": TARGET_FMR,
        "unique_official_images_loaded": len(relative_paths),
        "fold_results": fold_results,
        "aggregate_test_metrics": {
            metric: aggregate(values) for metric, values in metric_values.items()
        },
        "interpretation_status": "external transfer result; no LFW hyperparameter search",
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
            "train_unique_image_count",
            "pca_explained_variance",
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
                    "train_unique_image_count": row["train_unique_image_count"],
                    "pca_explained_variance": row["pca_explained_variance"],
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
    print(json.dumps(summary["aggregate_test_metrics"], indent=2))


if __name__ == "__main__":
    main()
