"""Validate the local ORL dataset and save a reproducible summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from face_verification.data import load_orl, theoretical_pair_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/orl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/audit/orl_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_orl(args.data_dir)
    people, counts = np.unique(dataset.labels, return_counts=True)
    same_pairs, different_pairs = theoretical_pair_counts(
        num_people=len(people),
        images_per_person=int(counts[0]),
    )

    summary = {
        "dataset": "ORL Database of Faces",
        "num_images": int(dataset.images.shape[0]),
        "num_identities": int(len(people)),
        "images_per_identity": sorted({int(value) for value in counts}),
        "image_height": int(dataset.images.shape[1]),
        "image_width": int(dataset.images.shape[2]),
        "flattened_dimension": int(dataset.flattened.shape[1]),
        "pixel_min": float(dataset.images.min()),
        "pixel_max": float(dataset.images.max()),
        "pixel_mean": float(dataset.images.mean()),
        "theoretical_same_identity_pairs": same_pairs,
        "theoretical_different_identity_pairs": different_pairs,
        "note": "Pair counts cover the full dataset before train/validation/test splitting.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

