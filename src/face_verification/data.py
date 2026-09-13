"""Dataset loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ORLDataset:
    """In-memory representation of the ORL face dataset."""

    images: np.ndarray
    labels: np.ndarray
    paths: tuple[Path, ...]

    @property
    def flattened(self) -> np.ndarray:
        """Return images as one row per image and one column per pixel."""

        return self.images.reshape(self.images.shape[0], -1)


def _numeric_suffix(path: Path, prefix: str = "") -> int:
    stem = path.stem if path.is_file() else path.name
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return int(stem)


def theoretical_pair_counts(num_people: int, images_per_person: int) -> tuple[int, int]:
    """Return total same-person and different-person unordered pair counts."""

    if num_people < 1 or images_per_person < 2:
        raise ValueError("pair counts require at least one person and two images per person")

    same_person = num_people * images_per_person * (images_per_person - 1) // 2
    identity_pairs = num_people * (num_people - 1) // 2
    different_person = identity_pairs * images_per_person * images_per_person
    return same_person, different_person


def load_orl(root: str | Path, *, strict: bool = True) -> ORLDataset:
    """Load ORL images after checking identities, counts, modes and dimensions."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ORL directory does not exist: {root}")

    subject_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.startswith("s")),
        key=lambda path: _numeric_suffix(path, "s"),
    )
    if strict and len(subject_dirs) != 40:
        raise ValueError(f"expected 40 ORL identities, found {len(subject_dirs)}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []
    expected_shape: tuple[int, int] | None = None

    for subject_dir in subject_dirs:
        label = _numeric_suffix(subject_dir, "s")
        image_paths = sorted(subject_dir.glob("*.pgm"), key=_numeric_suffix)
        if strict and len(image_paths) != 10:
            raise ValueError(f"expected 10 images for {subject_dir.name}, found {len(image_paths)}")

        for image_path in image_paths:
            with Image.open(image_path) as image:
                gray = image.convert("L")
                array = np.asarray(gray, dtype=np.float32) / 255.0

            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError(
                    f"inconsistent image shape for {image_path}: "
                    f"expected {expected_shape}, found {array.shape}"
                )

            images.append(array)
            labels.append(label)
            paths.append(image_path)

    if not images:
        raise ValueError(f"no PGM images found under {root}")

    dataset = ORLDataset(
        images=np.stack(images),
        labels=np.asarray(labels, dtype=np.int64),
        paths=tuple(paths),
    )
    if strict and dataset.images.shape != (400, 112, 92):
        raise ValueError(f"expected ORL shape (400, 112, 92), found {dataset.images.shape}")
    return dataset

