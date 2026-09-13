"""Verify that repeated classifier configurations reproduce exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_FIELDS = (
    "mean_train_eer",
    "mean_validation_eer",
    "std_validation_eer",
    "mean_generalization_gap",
    "mean_validation_roc_auc",
    "std_validation_roc_auc",
    "mean_fnmr_at_target_fmr",
    "std_fnmr_at_target_fmr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        "--coarse",
        dest="reference",
        type=Path,
        default=Path(
            "results/experiments/orl_classifier_search_k80_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--candidate",
        "--boundary",
        dest="candidate",
        type=Path,
        default=Path(
            "results/experiments/orl_classifier_boundary_k80_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/experiments/orl_classifier_boundary_k80_seed_20260913/overlap_audit.json"
        ),
    )
    return parser.parse_args()


def indexed_results(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row["config_id"]): row
        for row in payload["aggregate_results"]  # type: ignore[index,union-attr]
    }


def main() -> None:
    args = parse_args()
    reference = indexed_results(
        json.loads(args.reference.read_text(encoding="utf-8"))
    )
    candidate = indexed_results(
        json.loads(args.candidate.read_text(encoding="utf-8"))
    )
    shared_ids = sorted(reference.keys() & candidate.keys())
    comparisons = []
    for config_id in shared_ids:
        differing_fields = [
            field
            for field in METRIC_FIELDS
            if reference[config_id][field] != candidate[config_id][field]
        ]
        comparisons.append(
            {
                "config_id": config_id,
                "exact_match": not differing_fields,
                "differing_fields": differing_fields,
            }
        )

    payload = {
        "artifact_type": "classifier search overlap reproducibility audit",
        "reference_result": str(args.reference),
        "candidate_result": str(args.candidate),
        "compared_metric_fields": list(METRIC_FIELDS),
        "shared_configuration_count": len(shared_ids),
        "all_exact": bool(shared_ids) and all(
            row["exact_match"] for row in comparisons
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["all_exact"]:
        raise SystemExit("Repeated configurations did not reproduce exactly")


if __name__ == "__main__":
    main()
