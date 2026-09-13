"""Analyze identity, image, and pair overlap in the official LFW protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from face_verification.lfw_protocol import (
    overlap,
    parse_protocol,
    summarize_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/audit/lfw_protocol_overlap.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filenames = ("pairsDevTrain.txt", "pairsDevTest.txt", "pairs.txt")
    parsed = {
        filename: parse_protocol(args.data_dir / filename) for filename in filenames
    }
    summaries = {
        filename: summarize_protocol(header, pairs)
        for filename, (header, pairs) in parsed.items()
    }
    dev_train = parsed["pairsDevTrain.txt"][1]
    dev_test = parsed["pairsDevTest.txt"][1]
    benchmark = parsed["pairs.txt"][1]
    cross_protocol = {
        "dev_train_vs_dev_test": overlap(dev_train, dev_test),
        "dev_train_vs_benchmark": overlap(dev_train, benchmark),
        "dev_test_vs_benchmark": overlap(dev_test, benchmark),
    }

    fold_rows = []
    for fold in range(1, 11):
        held_out = [pair for pair in benchmark if pair.fold == fold]
        remaining = [pair for pair in benchmark if pair.fold != fold]
        held_out_summary = summarize_protocol([10, 300], held_out)
        fold_rows.append(
            {
                "held_out_fold": fold,
                "held_out_pair_count": len(held_out),
                "held_out_identity_count": held_out_summary["unique_identity_count"],
                "held_out_image_count": held_out_summary["unique_image_count"],
                **overlap(held_out, remaining),
            }
        )
    fold_overlap_summary = {
        key: {
            "minimum": min(row[key] for row in fold_rows),
            "maximum": max(row[key] for row in fold_rows),
            "mean": mean(row[key] for row in fold_rows),
        }
        for key in ("shared_identities_count", "shared_images_count", "shared_pairs_count")
    }
    summary = {
        "artifact_type": "LFW official protocol overlap audit",
        "protocols": summaries,
        "cross_protocol_overlap": cross_protocol,
        "benchmark_fold_overlap": fold_rows,
        "benchmark_fold_overlap_summary": fold_overlap_summary,
        "note": "This analysis inspects protocol membership only and fits no model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
