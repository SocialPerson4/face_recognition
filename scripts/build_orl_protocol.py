"""Build and save the frozen ORL identity split and balanced pair lists."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from face_verification.data import load_orl
from face_verification.protocol import make_balanced_pairs, split_by_identity


DEFAULT_SEED = 20260913
PAIR_SEED_OFFSETS = {"train": 101, "validation": 102, "test": 103}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/protocols/orl_seed_20260913"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def write_pairs(path: Path, pair_set, dataset, data_dir: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "pair_id",
                "left_index",
                "right_index",
                "left_path",
                "right_path",
                "left_identity",
                "right_identity",
                "target",
            ]
        )
        for pair_id, record in enumerate(pair_set.records):
            left_path = dataset.paths[record.left_index].relative_to(data_dir)
            right_path = dataset.paths[record.right_index].relative_to(data_dir)
            writer.writerow(
                [
                    pair_id,
                    record.left_index,
                    record.right_index,
                    left_path.as_posix(),
                    right_path.as_posix(),
                    int(dataset.labels[record.left_index]),
                    int(dataset.labels[record.right_index]),
                    record.target,
                ]
            )


def main() -> None:
    args = parse_args()
    dataset = load_orl(args.data_dir)
    split = split_by_identity(dataset.labels, seed=args.seed)
    split_indices = {
        "train": split.train_indices,
        "validation": split.validation_indices,
        "test": split.test_indices,
    }
    split_identities = {
        "train": split.train_identities,
        "validation": split.validation_identities,
        "test": split.test_identities,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "ORL Database of Faces",
        "strategy": "identity-disjoint 60/20/20 split, then balanced pairing",
        "split_seed": args.seed,
        "negative_to_positive_ratio": 1.0,
        "test_usage": "reserved for one final evaluation after all choices are frozen",
        "splits": {},
    }

    for split_name in ("train", "validation", "test"):
        pair_seed = args.seed + PAIR_SEED_OFFSETS[split_name]
        pair_set = make_balanced_pairs(
            dataset.labels,
            split_indices[split_name],
            seed=pair_seed,
        )
        write_pairs(
            args.output_dir / f"{split_name}_pairs.csv",
            pair_set,
            dataset,
            args.data_dir,
        )
        summary["splits"][split_name] = {
            "identity_count": len(split_identities[split_name]),
            "image_count": len(split_indices[split_name]),
            "identities": list(split_identities[split_name]),
            "pair_seed": pair_seed,
            "positive_pairs": pair_set.positive_count,
            "negative_pairs": pair_set.negative_count,
            "total_pairs": len(pair_set.records),
            "pair_file": f"{split_name}_pairs.csv",
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
