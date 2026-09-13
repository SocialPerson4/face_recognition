from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from face_verification.data import load_lfw_preprocessed_images


def test_lfw_preprocessing_produces_frozen_grayscale_shape(tmp_path: Path) -> None:
    root = tmp_path / "lfw_funneled"
    person = root / "Person_A"
    person.mkdir(parents=True)
    pixels = np.zeros((250, 250, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    Image.fromarray(pixels).save(person / "Person_A_0001.jpg")

    images = load_lfw_preprocessed_images(
        root, ["Person_A/Person_A_0001.jpg"]
    )

    assert images.shape == (1, 62, 47)
    assert images.dtype == np.float32
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0


def test_lfw_preprocessing_rejects_unexpected_source_size(tmp_path: Path) -> None:
    root = tmp_path / "lfw_funneled"
    person = root / "Person_A"
    person.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(person / "Person_A_0001.jpg")

    with pytest.raises(ValueError, match="expected 250x250"):
        load_lfw_preprocessed_images(root, ["Person_A/Person_A_0001.jpg"])
