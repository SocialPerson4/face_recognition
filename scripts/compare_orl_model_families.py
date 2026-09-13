"""Screen four verification model families on fixed identity-disjoint folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from face_verification.data import load_orl
from face_verification.features import (
    absolute_pair_differences,
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
DEFAULT_K = 80
INNER_FOLD_SEED_OFFSET = 201
VALIDATION_PAIR_SEED_OFFSET = 301
TRAIN_PAIR_SEED_OFFSET = 401
MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/orl_model_screening_k80_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    return parser.parse_args()


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def model_factories(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=5_000,
                random_state=seed,
            ),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="linear"),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", gamma="scale"),
        ),
    }


def score_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    *,
    target_fmr: float,
) -> dict[str, float]:
    train_eer = equal_error_rate(y_train, train_scores)
    validation_eer = equal_error_rate(y_validation, validation_scores)
    working_point = select_threshold_at_fmr(
        y_validation, validation_scores, target_fmr=target_fmr
    )
    return {
        "train_eer": train_eer.eer,
        "train_roc_auc": roc_auc(y_train, train_scores),
        "validation_eer": validation_eer.eer,
        "validation_roc_auc": roc_auc(y_validation, validation_scores),
        "validation_eer_threshold": validation_eer.threshold,
        "target_fmr": target_fmr,
        "working_threshold": working_point.rates.threshold,
        "working_fmr": working_point.rates.fmr,
        "working_fnmr": working_point.rates.fnmr,
    }


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def fit_and_score_classifier(
    model: object,
    train_features: np.ndarray,
    y_train: np.ndarray,
    validation_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and score while rejecting any genuine non-finite numerical result."""

    # NumPy 2 + macOS Accelerate can emit spurious matmul floating-point
    # warnings even when every input and output is finite.  Keep suppression
    # local, then explicitly fail if the resulting scores are not finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(train_features, y_train)
        train_scores = np.asarray(model.decision_function(train_features))
        validation_scores = np.asarray(
            model.decision_function(validation_features)
        )
    if not (
        np.all(np.isfinite(train_scores))
        and np.all(np.isfinite(validation_scores))
    ):
        raise FloatingPointError("classifier produced non-finite decision scores")
    return train_scores, validation_scores


def main() -> None:
    args = parse_args()
    dataset = load_orl(args.data_dir)
    outer = split_by_identity(dataset.labels, seed=args.seed)
    inner_seed = args.seed + INNER_FOLD_SEED_OFFSET
    folds = make_identity_folds(
        dataset.labels,
        outer.train_indices,
        n_splits=args.folds,
        seed=inner_seed,
    )

    rows: list[dict[str, object]] = []
    validation_score_rows: list[dict[str, object]] = []
    fold_protocols: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_pair_seed = args.seed + TRAIN_PAIR_SEED_OFFSET + fold_index
        validation_pair_seed = (
            args.seed + VALIDATION_PAIR_SEED_OFFSET + fold_index
        )
        train_pairs = make_balanced_pairs(
            dataset.labels, fold.train_indices, seed=train_pair_seed
        )
        validation_pairs = make_balanced_pairs(
            dataset.labels, fold.validation_indices, seed=validation_pair_seed
        )
        y_train = targets(train_pairs)
        y_validation = targets(validation_pairs)

        pca = fit_pca_components(
            dataset.images[list(fold.train_indices)], n_components=args.k
        )
        pca_features = np.full(
            (len(dataset.images), args.k), np.nan, dtype=np.float64
        )
        pca_features[list(fold.train_indices)] = transform_images(
            pca, dataset.images[list(fold.train_indices)]
        )
        pca_features[list(fold.validation_indices)] = transform_images(
            pca, dataset.images[list(fold.validation_indices)]
        )

        all_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        all_scores["pca_distance"] = (
            negative_euclidean_pair_scores(pca_features, train_pairs),
            negative_euclidean_pair_scores(pca_features, validation_pairs),
        )

        train_pair_features = absolute_pair_differences(pca_features, train_pairs)
        validation_pair_features = absolute_pair_differences(
            pca_features, validation_pairs
        )
        for model_name, model in model_factories(args.seed + fold_index).items():
            all_scores[model_name] = fit_and_score_classifier(
                model,
                train_pair_features,
                y_train,
                validation_pair_features,
            )

        fold_protocols.append(
            {
                "fold": fold_index,
                "train_identities": list(fold.train_identities),
                "validation_identities": list(fold.validation_identities),
                "train_pairs": len(train_pairs.records),
                "validation_pairs": len(validation_pairs.records),
                "train_pair_seed": train_pair_seed,
                "validation_pair_seed": validation_pair_seed,
            }
        )
        for model_name in MODEL_ORDER:
            train_scores, validation_scores = all_scores[model_name]
            metrics = score_metrics(
                y_train,
                train_scores,
                y_validation,
                validation_scores,
                target_fmr=args.target_fmr,
            )
            rows.append({"model": model_name, "fold": fold_index, **metrics})

        for pair_id, record in enumerate(validation_pairs.records):
            validation_score_rows.append(
                {
                    "fold": fold_index,
                    "pair_id": pair_id,
                    "left_index": record.left_index,
                    "right_index": record.right_index,
                    "target": record.target,
                    **{
                        f"{model_name}_score": float(all_scores[model_name][1][pair_id])
                        for model_name in MODEL_ORDER
                    },
                }
            )

    aggregates: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model_name]
        train_eers = [float(row["train_eer"]) for row in selected]
        validation_eers = [float(row["validation_eer"]) for row in selected]
        validation_aucs = [float(row["validation_roc_auc"]) for row in selected]
        validation_fnmrs = [float(row["working_fnmr"]) for row in selected]
        aggregates.append(
            {
                "model": model_name,
                "mean_train_eer": float(np.mean(train_eers)),
                "std_train_eer": sample_std(train_eers),
                "mean_validation_eer": float(np.mean(validation_eers)),
                "std_validation_eer": sample_std(validation_eers),
                "mean_generalization_gap": float(
                    np.mean(np.asarray(validation_eers) - np.asarray(train_eers))
                ),
                "mean_validation_roc_auc": float(np.mean(validation_aucs)),
                "std_validation_roc_auc": sample_std(validation_aucs),
                "mean_fnmr_at_target_fmr": float(np.mean(validation_fnmrs)),
                "std_fnmr_at_target_fmr": sample_std(validation_fnmrs),
            }
        )
    ranked = sorted(
        aggregates,
        key=lambda row: (
            float(row["mean_validation_eer"]),
            float(row["std_validation_eer"]),
            MODEL_ORDER.index(str(row["model"])),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["screening_rank"] = rank

    summary = {
        "artifact_type": "ORL fixed-k model-family screening development results",
        "dataset": "ORL Database of Faces",
        "outer_split_seed": args.seed,
        "outer_data_used": "training identities only",
        "outer_validation_status": "not used",
        "outer_test_status": "not used",
        "inner_fold_seed": inner_seed,
        "inner_folds": args.folds,
        "pca_dimensions": args.k,
        "pair_representation": "elementwise absolute PCA-coordinate difference",
        "score_direction": "larger means more likely same identity",
        "model_configurations": {
            "pca_distance": "negative Euclidean distance; no learned classifier",
            "logistic_regression": "StandardScaler + L2 logistic regression, C=1",
            "linear_svm": "StandardScaler + linear-kernel SVC, C=1",
            "rbf_svm": "StandardScaler + RBF-kernel SVC, C=1, gamma=scale",
        },
        "selection_status": (
            "screening only; default-parameter ranking is not a final model choice"
        ),
        "observed_screening_best": str(ranked[0]["model"]),
        "fold_protocols": fold_protocols,
        "aggregate_results": aggregates,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "validation_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(validation_score_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(validation_score_rows)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
