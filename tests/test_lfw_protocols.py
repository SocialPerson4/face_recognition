from pathlib import Path

from scripts.analyze_lfw_protocols import overlap, parse_protocol, summarize_protocol


def test_parse_ten_fold_protocol_assigns_rows_to_folds(tmp_path: Path) -> None:
    protocol = tmp_path / "pairs.txt"
    protocol.write_text(
        "2\t1\n"
        "A\t1\t2\nA\t1\tB\t1\n"
        "C\t1\t2\nC\t1\tD\t1\n",
        encoding="utf-8",
    )

    header, pairs = parse_protocol(protocol)

    assert header == [2, 1]
    assert [pair.fold for pair in pairs] == [1, 1, 2, 2]
    assert pairs[0].identities == frozenset({"A"})
    assert pairs[1].identities == frozenset({"A", "B"})
    assert summarize_protocol(header, pairs)["positive_pair_count"] == 2


def test_overlap_distinguishes_identity_image_and_pair_reuse(tmp_path: Path) -> None:
    left_path = tmp_path / "left.txt"
    right_path = tmp_path / "right.txt"
    left_path.write_text("1\nA\t1\t2\nA\t1\tB\t1\n", encoding="utf-8")
    right_path.write_text("1\nA\t2\t3\nA\t2\tB\t1\n", encoding="utf-8")
    left = parse_protocol(left_path)[1]
    right = parse_protocol(right_path)[1]

    result = overlap(left, right)

    assert result == {
        "shared_identities_count": 2,
        "shared_images_count": 2,
        "shared_pairs_count": 0,
    }
