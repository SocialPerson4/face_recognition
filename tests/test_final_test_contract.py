from pathlib import Path

from scripts.run_orl_final_test import EXPECTED_M4_SHA256, sha256


def test_frozen_m4_summary_matches_preregistered_hash() -> None:
    path = Path(
        "results/experiments/orl_frozen_validation_seed_20260913/summary.json"
    )
    assert sha256(path) == EXPECTED_M4_SHA256


def test_sha256_changes_when_contract_changes(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"c": 0.001}\n', encoding="utf-8")
    second.write_text('{"c": 0.002}\n', encoding="utf-8")

    assert sha256(first) != sha256(second)
