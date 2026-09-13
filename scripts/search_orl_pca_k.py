"""Coarse-search PCA dimensions with identity-disjoint inner cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from face_verification.data import load_orl
from face_verification.features import (
    fit_pca_components,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.metrics import equal_error_rate, roc_auc, select_threshold_at_fmr
from face_verification.protocol import (
    PairSet,
    make_balanced_pairs,
    make_identity_folds,
    split_by_identity,
)


DEFAULT_SEED = 20260913
INNER_FOLD_SEED_OFFSET = 201
PAIR_SEED_OFFSET = 301
DEFAULT_K_VALUES = (10, 20, 40, 80, 120, 160, 180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/orl_pca_k_coarse_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    return parser.parse_args()


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def sample_standard_deviation(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def main() -> None:
    args = parse_args()
    k_values = tuple(sorted(set(args.k_values)))
    if not k_values or k_values[0] < 1:
        raise ValueError("k-values must contain positive integers")

    dataset = load_orl(args.data_dir)
    outer = split_by_identity(dataset.labels, seed=args.seed)
    inner_seed = args.seed + INNER_FOLD_SEED_OFFSET
    folds = make_identity_folds(
        dataset.labels,
        outer.train_indices,
        n_splits=args.folds,
        seed=inner_seed,
    )
    smallest_training_set = min(len(fold.train_indices) for fold in folds)
    if k_values[-1] > smallest_training_set - 1:
        raise ValueError(
            f"largest k={k_values[-1]} exceeds the shared non-zero PCA limit "
            f"{smallest_training_set - 1}"
        )

    rows: list[dict[str, object]] = []
    fold_protocols: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds, start=1):
        pair_seed = args.seed + PAIR_SEED_OFFSET + fold_index
        validation_pairs = make_balanced_pairs(
            dataset.labels, fold.validation_indices, seed=pair_seed
        )
        y_true = targets(validation_pairs)

        max_pca = fit_pca_components(
            dataset.images[list(fold.train_indices)], n_components=k_values[-1]
        )
        validation_features = transform_images(
            max_pca, dataset.images[list(fold.validation_indices)]
        )
        global_to_local = {
            global_index: local_index
            for local_index, global_index in enumerate(fold.validation_indices)
        }
        local_pairs = PairSet(
            records=tuple(
                type(record)(
                    left_index=global_to_local[record.left_index],
                    right_index=global_to_local[record.right_index],
                    target=record.target,
                )
                for record in validation_pairs.records
            )
        )

        fold_protocols.append(
            {
                "fold": fold_index,
                "train_identities": list(fold.train_identities),
                "validation_identities": list(fold.validation_identities),
                "train_images": len(fold.train_indices),
                "validation_images": len(fold.validation_indices),
                "validation_positive_pairs": validation_pairs.positive_count,
                "validation_negative_pairs": validation_pairs.negative_count,
                "pair_seed": pair_seed,
            }
        )

        cumulative_variance = np.cumsum(max_pca.explained_variance_ratio_)
        for k in k_values:
            scores = negative_euclidean_pair_scores(
                validation_features[:, :k], local_pairs
            )
            eer = equal_error_rate(y_true, scores)
            threshold = select_threshold_at_fmr(
                y_true, scores, target_fmr=args.target_fmr
            )
            rows.append(
                {
                    "k": k,
                    "fold": fold_index,
                    "train_identity_count": len(fold.train_identities),
                    "validation_identity_count": len(fold.validation_identities),
                    "validation_pair_count": len(validation_pairs.records),
                    "retained_variance": float(cumulative_variance[k - 1]),
                    "eer": eer.eer,
                    "eer_threshold": eer.threshold,
                    "roc_auc": roc_auc(y_true, scores),
                    "target_fmr": args.target_fmr,
                    "working_threshold": threshold.rates.threshold,
                    "working_fmr": threshold.rates.fmr,
                    "working_fnmr": threshold.rates.fnmr,
                }
            )

    aggregates: list[dict[str, object]] = []
    for k in k_values:
        selected = [row for row in rows if row["k"] == k]
        eers = [float(row["eer"]) for row in selected]
        aucs = [float(row["roc_auc"]) for row in selected]
        fnmrs = [float(row["working_fnmr"]) for row in selected]
        variances = [float(row["retained_variance"]) for row in selected]
        aggregates.append(
            {
                "k": k,
                "mean_eer": float(np.mean(eers)),
                "std_eer": sample_standard_deviation(eers),
                "mean_roc_auc": float(np.mean(aucs)),
                "std_roc_auc": sample_standard_deviation(aucs),
                "mean_fnmr_at_target_fmr": float(np.mean(fnmrs)),
                "std_fnmr_at_target_fmr": sample_standard_deviation(fnmrs),
                "mean_retained_variance": float(np.mean(variances)),
            }
        )

    ranked = sorted(
        aggregates,
        key=lambda row: (
            float(row["mean_eer"]),
            float(row["std_eer"]),
            -float(row["mean_roc_auc"]),
            int(row["k"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["coarse_rank"] = rank
    aggregates.sort(key=lambda row: int(row["k"]))

    summary = {
        "artifact_type": "ORL PCA-k coarse-search development results",
        "dataset": "ORL Database of Faces",
        "outer_split_seed": args.seed,
        "outer_data_used": "training identities only",
        "outer_validation_status": "not used",
        "outer_test_status": "not used",
        "inner_fold_seed": inner_seed,
        "inner_folds": args.folds,
        "k_values": list(k_values),
        "score_definition": "negative Euclidean distance in the first k PCA coordinates",
        "primary_ranking": "ascending mean inner-validation EER",
        "standard_deviation": "sample standard deviation across identity folds (ddof=1)",
        "selection_status": "coarse best is provisional; fine search is required",
        "coarse_best_k": int(ranked[0]["k"]),
        "fold_protocols": fold_protocols,
        "aggregate_results": aggregates,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    row_path = args.output_dir / "fold_results.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
