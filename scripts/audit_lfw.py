"""Audit the local LFW Funneled archive, images, and official pair protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


EXPECTED_SHA256 = {
    "lfw-funneled.tgz": "b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a",
    "pairsDevTrain.txt": "1d454dada7dfeca0e7eab6f65dc4e97a6312d44cf142207be28d688be92aabfa",
    "pairsDevTest.txt": "7cb06600ea8b2814ac26e946201cdb304296262aad67d046a16a7ec85d0ff87c",
    "pairs.txt": "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/lfw"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/audit/lfw_funneled_summary.json")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_image_path(image_root: Path, identity: str, index: str) -> Path:
    return image_root / identity / f"{identity}_{int(index):04d}.jpg"


def audit_pair_protocol(path: Path, image_root: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines or not lines[0]:
        raise ValueError(f"Empty pair protocol: {path}")
    header = [int(value) for value in lines[0].split()]
    if len(header) == 1:
        expected_pairs = 2 * header[0]
        folds = None
        pairs_per_class_per_fold = None
    elif len(header) == 2:
        folds, pairs_per_class_per_fold = header
        expected_pairs = 2 * folds * pairs_per_class_per_fold
    else:
        raise ValueError(f"Unsupported pair header in {path}: {header}")

    positive_count = 0
    negative_count = 0
    referenced_paths: set[Path] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) == 3:
            identity, left_index, right_index = fields
            positive_count += 1
            referenced_paths.add(pair_image_path(image_root, identity, left_index))
            referenced_paths.add(pair_image_path(image_root, identity, right_index))
        elif len(fields) == 4:
            left_identity, left_index, right_identity, right_index = fields
            negative_count += 1
            referenced_paths.add(
                pair_image_path(image_root, left_identity, left_index)
            )
            referenced_paths.add(
                pair_image_path(image_root, right_identity, right_index)
            )
        else:
            raise ValueError(f"Malformed pair row {path}:{line_number}: {line}")

    actual_pairs = positive_count + negative_count
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"Pair count mismatch in {path}: expected {expected_pairs}, got {actual_pairs}"
        )
    missing = sorted(str(item) for item in referenced_paths if not item.is_file())
    return {
        "header": header,
        "folds": folds,
        "pairs_per_class_per_fold": pairs_per_class_per_fold,
        "pair_count": actual_pairs,
        "positive_pair_count": positive_count,
        "negative_pair_count": negative_count,
        "unique_referenced_image_count": len(referenced_paths),
        "missing_referenced_images": missing,
    }


def main() -> None:
    args = parse_args()
    image_root = args.data_dir / "lfw_funneled"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing extracted image directory: {image_root}")

    file_hashes = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = args.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing required LFW file: {path}")
        actual = sha256_file(path)
        file_hashes[filename] = {
            "sha256": actual,
            "expected_sha256": expected,
            "matches_expected": actual == expected,
        }
    if not all(item["matches_expected"] for item in file_hashes.values()):
        raise ValueError("At least one LFW file failed SHA-256 verification")

    identity_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    counts = {path.name: len(list(path.glob("*.jpg"))) for path in identity_dirs}
    image_paths = sorted(path for path in image_root.rglob("*.jpg") if path.is_file())
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    corrupt_images: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                image.verify()
        except Exception as error:  # report every decoder failure in the artifact
            corrupt_images.append(f"{path}: {type(error).__name__}: {error}")

    count_distribution = Counter(counts.values())
    protocols = {
        filename: audit_pair_protocol(args.data_dir / filename, image_root)
        for filename in ("pairsDevTrain.txt", "pairsDevTest.txt", "pairs.txt")
    }
    summary = {
        "dataset": "Labeled Faces in the Wild, Funneled",
        "audit_status": "passed"
        if not corrupt_images
        and all(not item["missing_referenced_images"] for item in protocols.values())
        else "failed",
        "file_hashes": file_hashes,
        "num_identities": len(identity_dirs),
        "num_images": len(image_paths),
        "identity_image_count_distribution": {
            str(count): identities for count, identities in sorted(count_distribution.items())
        },
        "single_image_identities": sum(count == 1 for count in counts.values()),
        "identities_with_at_least_two_images": sum(
            count >= 2 for count in counts.values()
        ),
        "maximum_images_for_one_identity": max(counts.values()),
        "identities_with_maximum_images": sorted(
            identity for identity, count in counts.items() if count == max(counts.values())
        ),
        "image_dimensions": dict(sorted(dimensions.items())),
        "image_modes": dict(sorted(modes.items())),
        "corrupt_images": corrupt_images,
        "official_pair_protocols": protocols,
        "note": "This audit validates data only; it does not fit or evaluate a model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["audit_status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
