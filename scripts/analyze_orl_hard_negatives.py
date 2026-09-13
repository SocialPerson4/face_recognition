"""Compare ordinary and identity-level hard negative proxies on ORL validation."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

from face_verification.data import load_orl
from face_verification.features import (
    absolute_pair_differences,
    fit_pca_components,
    negative_euclidean_pair_scores,
    transform_images,
)
from face_verification.models import fit_and_score_classifier, make_verification_classifier
from face_verification.protocol import PairRecord, PairSet, make_balanced_pairs, split_by_identity


DEFAULT_SEED = 20260913
TRAIN_PAIR_SEED_OFFSET = 101
ORDINARY_PAIR_SEED_OFFSET = 501
HARD_IDENTITY_FRACTION = 0.25
MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")


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
        default=Path("results/experiments/orl_hard_negative_proxy_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def rank_identity_pairs(
    identities: tuple[int, ...], centers: dict[int, np.ndarray]
) -> list[dict[str, object]]:
    rows = [
        {
            "left_identity": left,
            "right_identity": right,
            "centroid_distance": float(np.linalg.norm(centers[left] - centers[right])),
        }
        for left, right in combinations(identities, 2)
    ]
    rows.sort(
        key=lambda row: (
            float(row["centroid_distance"]),
            int(row["left_identity"]),
            int(row["right_identity"]),
        )
    )
    hard_count = max(1, int(np.ceil(len(rows) * HARD_IDENTITY_FRACTION)))
    for rank, row in enumerate(rows, start=1):
        row["similarity_rank"] = rank
        row["is_hard_identity_pair"] = rank <= hard_count
    return rows


def pair_records_for_identity_pairs(
    labels: np.ndarray,
    identity_pairs: set[tuple[int, int]],
) -> list[PairRecord]:
    indices = {
        identity: tuple(np.flatnonzero(labels == identity).tolist())
        for pair in identity_pairs
        for identity in pair
    }
    return [
        PairRecord(left_index=left_index, right_index=right_index, target=0)
        for left_identity, right_identity in sorted(identity_pairs)
        for left_index, right_index in product(
            indices[left_identity], indices[right_identity]
        )
    ]


def score_groups(
    pca_features: np.ndarray,
    train_pairs: PairSet,
    groups: dict[str, PairSet],
    configurations: dict[str, dict[str, object]],
    *,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    results = {
        group: {
            "pca_distance": negative_euclidean_pair_scores(pca_features, pairs)
        }
        for group, pairs in groups.items()
    }
    train_features = absolute_pair_differences(pca_features, train_pairs)
    y_train = np.fromiter(
        (record.target for record in train_pairs.records), dtype=np.int64
    )
    for family in MODEL_ORDER[1:]:
        config = configurations[family]
        gamma = config["gamma"] if config["gamma"] is not None else "scale"
        model = make_verification_classifier(
            family, c=float(config["c"]), gamma=gamma, seed=seed
        )
        validation_features = np.vstack(
            [absolute_pair_differences(pca_features, groups[group]) for group in groups]
        )
        _, combined_scores = fit_and_score_classifier(
            model, train_features, y_train, validation_features
        )
        start = 0
        for group, pairs in groups.items():
            stop = start + len(pairs.records)
            results[group][family] = combined_scores[start:stop]
            start = stop
    return results


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
    configurations = frozen["frozen_configurations"]
    thresholds = {
        row["model"]: float(
            row["validation"]["working_point"]["rates"]["threshold"]
        )
        for row in frozen["results"]
    }

    dataset = load_orl(args.data_dir)
    split = split_by_identity(dataset.labels, seed=args.seed)
    train_pairs = make_balanced_pairs(
        dataset.labels,
        split.train_indices,
        seed=args.seed + TRAIN_PAIR_SEED_OFFSET,
    )
    pca = fit_pca_components(
        dataset.images[list(split.train_indices)], n_components=80
    )
    pca_features = np.full((len(dataset.images), 80), np.nan, dtype=np.float64)
    development_indices = list(split.train_indices + split.validation_indices)
    pca_features[development_indices] = transform_images(
        pca, dataset.images[development_indices]
    )

    centers = {
        identity: np.mean(
            pca_features[np.flatnonzero(dataset.labels == identity)], axis=0
        )
        for identity in split.validation_identities
    }
    identity_ranking = rank_identity_pairs(split.validation_identities, centers)
    hard_identity_pairs = {
        (int(row["left_identity"]), int(row["right_identity"]))
        for row in identity_ranking
        if row["is_hard_identity_pair"]
    }
    ordinary_identity_pairs = {
        (int(row["left_identity"]), int(row["right_identity"]))
        for row in identity_ranking
        if not row["is_hard_identity_pair"]
    }
    hard_records = pair_records_for_identity_pairs(
        dataset.labels, hard_identity_pairs
    )
    ordinary_pool = pair_records_for_identity_pairs(
        dataset.labels, ordinary_identity_pairs
    )
    rng = np.random.default_rng(args.seed + ORDINARY_PAIR_SEED_OFFSET)
    ordinary_indices = rng.choice(
        len(ordinary_pool), size=len(hard_records), replace=False
    )
    ordinary_records = [ordinary_pool[int(index)] for index in ordinary_indices]
    groups = {
        "ordinary": PairSet(records=tuple(ordinary_records)),
        "hard_identity_proxy": PairSet(records=tuple(hard_records)),
    }
    scores = score_groups(
        pca_features,
        train_pairs,
        groups,
        configurations,
        seed=args.seed,
    )

    model_results: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        group_results = {}
        for group in groups:
            values = scores[group][model]
            accepted = values >= thresholds[model]
            group_results[group] = {
                "pair_count": len(values),
                "false_accept_count": int(np.sum(accepted)),
                "false_match_rate": float(np.mean(accepted)),
                "median_score": float(np.median(values)),
                "score_90th_percentile": float(np.quantile(values, 0.9)),
            }
        ordinary_fmr = group_results["ordinary"]["false_match_rate"]
        hard_fmr = group_results["hard_identity_proxy"]["false_match_rate"]
        risk_multiplier = None if ordinary_fmr == 0.0 else hard_fmr / ordinary_fmr
        model_results.append(
            {
                "model": model,
                "frozen_threshold": thresholds[model],
                "groups": group_results,
                "hard_to_ordinary_fmr_multiplier": risk_multiplier,
                "hard_minus_ordinary_fmr": hard_fmr - ordinary_fmr,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "identity_pair_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(identity_ranking[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(identity_ranking)

    score_fields = [
        "group",
        "pair_id",
        "left_index",
        "right_index",
        "left_identity",
        "right_identity",
        *(f"{model}_score" for model in MODEL_ORDER),
        *(f"{model}_accepted" for model in MODEL_ORDER),
    ]
    with (args.output_dir / "negative_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields, lineterminator="\n")
        writer.writeheader()
        for group, pairs in groups.items():
            for pair_id, record in enumerate(pairs.records):
                writer.writerow(
                    {
                        "group": group,
                        "pair_id": pair_id,
                        "left_index": record.left_index,
                        "right_index": record.right_index,
                        "left_identity": int(dataset.labels[record.left_index]),
                        "right_identity": int(dataset.labels[record.right_index]),
                        **{
                            f"{model}_score": float(scores[group][model][pair_id])
                            for model in MODEL_ORDER
                        },
                        **{
                            f"{model}_accepted": int(
                                scores[group][model][pair_id] >= thresholds[model]
                            )
                            for model in MODEL_ORDER
                        },
                    }
                )

    summary = {
        "artifact_type": "ORL post-test identity-level hard-negative proxy analysis",
        "analysis_status": "post-test exploratory; cannot modify M5 conclusions",
        "data_scope": "outer validation identities only",
        "selection_proxy": "Euclidean distance between identity mean PCA embeddings",
        "selection_bias_disclosure": (
            "hard groups are selected in PCA space, so this is a directed stress test, "
            "not an unbiased model-family comparison"
        ),
        "twin_label_status": "ORL has no twin or kinship labels",
        "hard_identity_fraction": HARD_IDENTITY_FRACTION,
        "identity_pair_count": len(identity_ranking),
        "hard_identity_pair_count": len(hard_identity_pairs),
        "ordinary_identity_pair_count": len(ordinary_identity_pairs),
        "ordinary_sampling_seed": args.seed + ORDINARY_PAIR_SEED_OFFSET,
        "pairs_per_group": len(hard_records),
        "hard_identity_pairs": [list(pair) for pair in sorted(hard_identity_pairs)],
        "model_results": model_results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
