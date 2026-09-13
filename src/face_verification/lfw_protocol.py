"""Parsing and set operations for official LFW verification protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
