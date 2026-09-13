"""Search classifier hyperparameters on fixed identity-disjoint ORL folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from face_verification.data import load_orl
from face_verification.features import (
    absolute_pair_differences,
    fit_pca_components,
    transform_images,
)
from face_verification.metrics import equal_error_rate, roc_auc, select_threshold_at_fmr
from face_verification.models import fit_and_score_classifier, make_verification_classifier
from face_verification.protocol import (
    PairSet,
    make_balanced_pairs,
    make_identity_folds,
    split_by_identity,
)
from face_verification.selection import select_configuration_one_standard_error


DEFAULT_SEED = 20260913
DEFAULT_K = 80
INNER_FOLD_SEED_OFFSET = 201
VALIDATION_PAIR_SEED_OFFSET = 301
TRAIN_PAIR_SEED_OFFSET = 401
COARSE_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
COARSE_GAMMA_VALUES = (0.0001, 0.001, 0.01, 0.1, 1.0)
BOUNDARY_C_VALUES = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003)
BOUNDARY_GAMMA_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1)
SENTINEL_C_VALUES = (0.0000001, 0.000001, 0.00001)
SENTINEL_GAMMA_VALUES = (0.001, 0.01)
FAMILIES = ("logistic_regression", "linear_svm", "rbf_svm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--stage", choices=("coarse", "boundary", "sentinel"), default="coarse"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    return parser.parse_args()


def configurations(
    c_values: tuple[float, ...], gamma_values: tuple[float, ...]
) -> tuple[dict[str, object], ...]:
    configs: list[dict[str, object]] = []
    for family in ("logistic_regression", "linear_svm"):
        for c in c_values:
            configs.append(
                {
                    "config_id": f"{family}__c_{c:g}",
                    "family": family,
                    "c": c,
                    "gamma": None,
                }
            )
    for c in c_values:
        for gamma in gamma_values:
            configs.append(
                {
                    "config_id": f"rbf_svm__c_{c:g}__gamma_{gamma:g}",
                    "family": "rbf_svm",
                    "c": c,
                    "gamma": gamma,
                }
            )
    return tuple(configs)


def targets(pairs: PairSet) -> np.ndarray:
    return np.fromiter((record.target for record in pairs.records), dtype=np.int64)


def fold_metrics(
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
        "validation_eer": validation_eer.eer,
        "validation_roc_auc": roc_auc(y_validation, validation_scores),
        "working_fmr": working_point.rates.fmr,
        "working_fnmr": working_point.rates.fnmr,
    }


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def main() -> None:
    args = parse_args()
    if args.stage == "sentinel":
        c_values = SENTINEL_C_VALUES
        gamma_values = SENTINEL_GAMMA_VALUES
        default_output_dir = Path(
            "results/experiments/orl_classifier_sentinel_k80_seed_20260913"
        )
    elif args.stage == "boundary":
        c_values = BOUNDARY_C_VALUES
        gamma_values = BOUNDARY_GAMMA_VALUES
        default_output_dir = Path(
            "results/experiments/orl_classifier_boundary_k80_seed_20260913"
        )
    else:
        c_values = COARSE_C_VALUES
        gamma_values = COARSE_GAMMA_VALUES
        default_output_dir = Path(
            "results/experiments/orl_classifier_search_k80_seed_20260913"
        )
    output_dir = args.output_dir or default_output_dir
    configs = configurations(c_values, gamma_values)
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
    fold_protocols: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_pair_seed = args.seed + TRAIN_PAIR_SEED_OFFSET + fold_index
        validation_pair_seed = args.seed + VALIDATION_PAIR_SEED_OFFSET + fold_index
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
        train_features = absolute_pair_differences(pca_features, train_pairs)
        validation_features = absolute_pair_differences(
            pca_features, validation_pairs
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
        for config in configs:
            gamma = config["gamma"] if config["gamma"] is not None else "scale"
            model = make_verification_classifier(
                str(config["family"]),
                c=float(config["c"]),
                gamma=gamma,
                seed=args.seed + fold_index,
            )
            train_scores, validation_scores = fit_and_score_classifier(
                model, train_features, y_train, validation_features
            )
            rows.append(
                {
                    **config,
                    "fold": fold_index,
                    **fold_metrics(
                        y_train,
                        train_scores,
                        y_validation,
                        validation_scores,
                        target_fmr=args.target_fmr,
                    ),
                }
            )

    aggregates: list[dict[str, object]] = []
    for config in configs:
        selected = [row for row in rows if row["config_id"] == config["config_id"]]
        train_eers = [float(row["train_eer"]) for row in selected]
        validation_eers = [float(row["validation_eer"]) for row in selected]
        aucs = [float(row["validation_roc_auc"]) for row in selected]
        fnmrs = [float(row["working_fnmr"]) for row in selected]
        aggregates.append(
            {
                **config,
                "mean_train_eer": float(np.mean(train_eers)),
                "mean_validation_eer": float(np.mean(validation_eers)),
                "std_validation_eer": sample_std(validation_eers),
                "mean_generalization_gap": float(
                    np.mean(np.asarray(validation_eers) - np.asarray(train_eers))
                ),
                "mean_validation_roc_auc": float(np.mean(aucs)),
                "std_validation_roc_auc": sample_std(aucs),
                "mean_fnmr_at_target_fmr": float(np.mean(fnmrs)),
                "std_fnmr_at_target_fmr": sample_std(fnmrs),
            }
        )

    family_selections: dict[str, object] = {}
    for family in FAMILIES:
        family_rows = [row for row in aggregates if row["family"] == family]
        if family == "rbf_svm":
            complexity = np.asarray(
                [[float(row["gamma"]), float(row["c"])] for row in family_rows]
            )
            complexity_rule = "ascending gamma, then ascending C"
        else:
            complexity = np.asarray([float(row["c"]) for row in family_rows])
            complexity_rule = "ascending C"
        selection = select_configuration_one_standard_error(
            np.asarray([row["mean_validation_eer"] for row in family_rows]),
            np.asarray([row["std_validation_eer"] for row in family_rows]),
            complexity,
            n_folds=args.folds,
        )
        observed = family_rows[selection.observed_best_index]
        chosen = family_rows[selection.selected_index]
        eligible = [
            str(family_rows[index]["config_id"])
            for index in selection.eligible_indices
        ]
        family_selections[family] = {
            "observed_best_config": str(observed["config_id"]),
            "selected_config": str(chosen["config_id"]),
            "selected_parameters": {
                "c": chosen["c"],
                "gamma": chosen["gamma"],
            },
            "best_mean_eer": selection.best_mean_loss,
            "best_std_eer": selection.best_standard_deviation,
            "best_standard_error": selection.best_standard_error,
            "eligibility_cutoff": selection.eligibility_cutoff,
            "complexity_rule": complexity_rule,
            "eligible_configs": eligible,
            "selected_aggregate_result": chosen,
        }

    sentinel_diagnostics: dict[str, object] | None = None
    if args.stage == "sentinel":
        anchor_c = SENTINEL_C_VALUES[-1]
        diagnostic_rows: list[dict[str, object]] = []
        for family in FAMILIES:
            gammas: tuple[float | None, ...] = (
                SENTINEL_GAMMA_VALUES if family == "rbf_svm" else (None,)
            )
            for gamma in gammas:
                series = [
                    row
                    for row in aggregates
                    if row["family"] == family and row["gamma"] == gamma
                ]
                anchor = next(row for row in series if row["c"] == anchor_c)
                anchor_folds = {
                    int(row["fold"]): float(row["validation_eer"])
                    for row in rows
                    if row["family"] == family
                    and row["gamma"] == gamma
                    and row["c"] == anchor_c
                }
                for candidate in series:
                    if candidate["c"] == anchor_c:
                        continue
                    improved_folds = sum(
                        float(row["validation_eer"])
                        < anchor_folds[int(row["fold"])]
                        for row in rows
                        if row["family"] == family
                        and row["gamma"] == gamma
                        and row["c"] == candidate["c"]
                    )
                    eer_improvement = float(anchor["mean_validation_eer"]) - float(
                        candidate["mean_validation_eer"]
                    )
                    auc_improvement = float(candidate["mean_validation_roc_auc"]) - float(
                        anchor["mean_validation_roc_auc"]
                    )
                    diagnostic_rows.append(
                        {
                            "family": family,
                            "gamma": gamma,
                            "anchor_c": anchor_c,
                            "candidate_c": candidate["c"],
                            "eer_improvement": eer_improvement,
                            "auc_improvement": auc_improvement,
                            "improved_fold_count": improved_folds,
                            "stable_material_improvement": (
                                eer_improvement >= 0.005
                                and auc_improvement >= 0.005
                                and improved_folds >= 4
                            ),
                        }
                    )
        sentinel_diagnostics = {
            "purpose": "boundary diagnosis, not renewed winner selection",
            "anchor_c": anchor_c,
            "minimum_eer_improvement": 0.005,
            "minimum_auc_improvement": 0.005,
            "minimum_improved_folds": 4,
            "comparisons": diagnostic_rows,
            "any_stable_material_improvement": any(
                bool(row["stable_material_improvement"])
                for row in diagnostic_rows
            ),
        }

    summary = {
        "artifact_type": f"ORL classifier hyperparameter {args.stage}-search results",
        "search_stage": args.stage,
        "dataset": "ORL Database of Faces",
        "outer_split_seed": args.seed,
        "outer_data_used": "training identities only",
        "outer_validation_status": "not used",
        "outer_test_status": "not used",
        "inner_fold_seed": inner_seed,
        "inner_folds": args.folds,
        "pca_dimensions": args.k,
        "c_values": list(c_values),
        "gamma_values": list(gamma_values),
        "configuration_count": len(configs),
        "classifier_fit_count": len(configs) * args.folds,
        "primary_metric": "mean identity-fold validation EER",
        "selection_rule": (
            "sentinel diagnostic thresholds; one-SE summaries are descriptive only"
            if args.stage == "sentinel"
            else "one standard error, then predeclared family complexity"
        ),
        "family_selections_status": (
            "descriptive only; do not replace preregistered frozen candidates"
            if args.stage == "sentinel"
            else "used for within-family candidate selection"
        ),
        "fold_protocols": fold_protocols,
        "family_selections": family_selections,
        "sentinel_diagnostics": sentinel_diagnostics,
        "aggregate_results": aggregates,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
