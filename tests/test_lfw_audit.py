from pathlib import Path

from scripts.audit_lfw import audit_pair_protocol, pair_image_path


def test_pair_image_path_uses_lfw_four_digit_naming() -> None:
    assert pair_image_path(Path("images"), "Ada_Lovelace", "3") == Path(
        "images/Ada_Lovelace/Ada_Lovelace_0003.jpg"
    )


def test_audit_pair_protocol_counts_and_resolves_images(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    for identity, indices in {"Person_A": (1, 2), "Person_B": (1,)}.items():
        directory = image_root / identity
        directory.mkdir(parents=True)
        for index in indices:
            (directory / f"{identity}_{index:04d}.jpg").touch()
    protocol = tmp_path / "pairs.txt"
    protocol.write_text(
        "1\nPerson_A\t1\t2\nPerson_A\t1\tPerson_B\t1\n", encoding="utf-8"
    )

    result = audit_pair_protocol(protocol, image_root)

    assert result["pair_count"] == 2
    assert result["positive_pair_count"] == 1
    assert result["negative_pair_count"] == 1
    assert result["missing_referenced_images"] == []
