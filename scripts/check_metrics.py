"""Save hand-checkable synthetic verification metric examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from face_verification.metrics import (
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    select_threshold_at_fmr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/audit/metric_sanity.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = np.array([1, 1, 0, 0])
    perfect_scores = np.array([0.9, 0.7, 0.6, 0.2])
    overlapping_scores = np.array([0.9, 0.4, 0.8, 0.3])

    summary = {
        "artifact_type": "synthetic metric sanity check",
        "warning": "These values are hand-checkable examples, not ORL model results.",
        "label_meaning": {"1": "same identity", "0": "different identity"},
        "accept_rule": "score >= threshold",
        "perfect_example": {
            "labels": labels.tolist(),
            "scores": perfect_scores.tolist(),
            "rates_at_0.65": asdict(
                rates_at_threshold(labels, perfect_scores, threshold=0.65)
            ),
            "selected_at_fmr_0": asdict(
                select_threshold_at_fmr(labels, perfect_scores, target_fmr=0.0)
            ),
            "eer": asdict(equal_error_rate(labels, perfect_scores)),
            "roc_auc": roc_auc(labels, perfect_scores),
        },
        "overlapping_example": {
            "labels": labels.tolist(),
            "scores": overlapping_scores.tolist(),
            "eer": asdict(equal_error_rate(labels, overlapping_scores)),
            "roc_auc": roc_auc(labels, overlapping_scores),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
