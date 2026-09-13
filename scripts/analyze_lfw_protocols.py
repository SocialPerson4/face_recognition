"""Analyze identity, image, and pair overlap in the official LFW protocols."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class ProtocolPair:
    left_image: str
    right_image: str
    target: int
    fold: int | None

    @property
    def identities(self) -> frozenset[str]:
        return frozenset(
            (self.left_image.split("/", 1)[0], self.right_image.split("/", 1)[0])
        )

    @property
    def canonical_key(self) -> tuple[int, str, str]:
        left, right = sorted((self.left_image, self.right_image))
        return self.target, left, right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/audit/lfw_protocol_overlap.json")
    )
    return parser.parse_args()


def image_name(identity: str, index: str) -> str:
    return f"{identity}/{identity}_{int(index):04d}.jpg"


def parse_protocol(path: Path) -> tuple[list[int], list[ProtocolPair]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    header = [int(value) for value in lines[0].split()]
    raw_rows = [line.split() for line in lines[1:] if line]
    if len(header) == 1:
        pairs_per_class = header[0]
        expected = 2 * pairs_per_class
        folds = [None] * expected
    elif len(header) == 2:
        fold_count, pairs_per_class = header
        expected = 2 * fold_count * pairs_per_class
        folds = [
            fold
            for fold in range(1, fold_count + 1)
            for _ in range(2 * pairs_per_class)
        ]
    else:
        raise ValueError(f"Unsupported protocol header: {header}")
    if len(raw_rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, got {len(raw_rows)}")

    pairs = []
    for row, fold in zip(raw_rows, folds):
        if len(row) == 3:
            identity, left_index, right_index = row
            pairs.append(
                ProtocolPair(
                    image_name(identity, left_index),
                    image_name(identity, right_index),
                    1,
                    fold,
                )
            )
        elif len(row) == 4:
            left_identity, left_index, right_identity, right_index = row
            pairs.append(
                ProtocolPair(
                    image_name(left_identity, left_index),
                    image_name(right_identity, right_index),
                    0,
                    fold,
                )
            )
        else:
            raise ValueError(f"Malformed protocol row in {path}: {row}")
    return header, pairs


def pair_sets(pairs: list[ProtocolPair]) -> dict[str, set[object]]:
    return {
        "identities": {identity for pair in pairs for identity in pair.identities},
        "images": {
            image for pair in pairs for image in (pair.left_image, pair.right_image)
        },
        "pairs": {pair.canonical_key for pair in pairs},
    }


def summarize_protocol(header: list[int], pairs: list[ProtocolPair]) -> dict[str, object]:
    sets = pair_sets(pairs)
    return {
        "header": header,
        "pair_count": len(pairs),
        "positive_pair_count": sum(pair.target == 1 for pair in pairs),
        "negative_pair_count": sum(pair.target == 0 for pair in pairs),
        "unique_identity_count": len(sets["identities"]),
        "unique_image_count": len(sets["images"]),
        "unique_pair_count": len(sets["pairs"]),
        "duplicate_pair_count": len(pairs) - len(sets["pairs"]),
    }


def overlap(left: list[ProtocolPair], right: list[ProtocolPair]) -> dict[str, int]:
    left_sets = pair_sets(left)
    right_sets = pair_sets(right)
    return {
        f"shared_{name}_count": len(left_sets[name] & right_sets[name])
        for name in ("identities", "images", "pairs")
    }


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
