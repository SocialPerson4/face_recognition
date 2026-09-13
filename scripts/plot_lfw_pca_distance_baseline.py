"""Plot saved LFW PCA-distance transfer results without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

_MATPLOTLIB_CACHE = Path("results/tmp/matplotlib").resolve()
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiments/lfw_pca_distance_8_1_1"),
    )
    parser.add_argument(
        "--orl-summary",
        type=Path,
        default=Path("results/experiments/orl_final_test_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/lfw_pca_distance_8_1_1"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_clean_svg(figure: plt.Figure, png_path: Path, svg_path: Path) -> None:
    figure.savefig(png_path, dpi=320)
    figure.savefig(svg_path)
    plt.close(figure)
    clean_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(clean_svg + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    folds = read_csv(args.input_dir / "fold_results.csv")
    scores = read_csv(args.input_dir / "test_scores.csv")
    orl = json.loads(args.orl_summary.read_text(encoding="utf-8"))
    orl_eer = next(
        row["test_eer"]["eer"] for row in orl["results"] if row["model"] == "pca_distance"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fold_numbers = np.asarray([int(row["test_fold"]) for row in folds])
    eer = 100 * np.asarray([float(row["test_eer"]) for row in folds])
    fmr = 100 * np.asarray([float(row["test_fmr"]) for row in folds])
    fnmr = 100 * np.asarray([float(row["test_fnmr"]) for row in folds])
    mean_eer = 100 * summary["aggregate_test_metrics"]["eer"]["mean"]
    std_eer = 100 * summary["aggregate_test_metrics"]["eer"]["standard_deviation"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 5.9))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.77, wspace=0.28)
    left.plot(fold_numbers, eer, color="#356D9A", marker="o", linewidth=2)
    left.axhspan(mean_eer - std_eer, mean_eer + std_eer, color="#356D9A", alpha=0.14)
    left.axhline(mean_eer, color="#356D9A", linestyle="--", linewidth=1.5)
    left.axhline(100 * orl_eer, color="#D5523A", linestyle=":", linewidth=2)
    left.text(10.2, mean_eer, f"LFW mean {mean_eer:.2f}%", va="center", color="#234E70")
    left.text(10.2, 100 * orl_eer, f"ORL final {100 * orl_eer:.2f}%", va="center", color="#A33C2C")
    left.set_xlim(0.7, 11.9)
    left.set_ylim(8, 44)
    left.set_xticks(fold_numbers)
    left.set_xlabel("Held-out identity fold")
    left.set_ylabel("Equal error rate (%)")
    left.set_title("Transfer error rises on unconstrained faces", fontweight="bold")
    left.grid(axis="y", color="#D9DEE7", linewidth=0.8)

    width = 0.36
    right.bar(fold_numbers - width / 2, fmr, width, color="#356D9A", label="FMR")
    right.bar(fold_numbers + width / 2, fnmr, width, color="#D5523A", label="FNMR")
    right.axhline(1, color="#111827", linestyle="--", linewidth=1.2, label="1% target")
    right.set_xticks(fold_numbers)
    right.set_xlabel("Held-out identity fold")
    right.set_ylabel("Error rate at calibration threshold (%)")
    right.set_title("Low false accepts require rejecting most genuine pairs", fontweight="bold")
    right.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    right.legend(frameon=False, loc="center right")
    right.text(
        0.98,
        0.95,
        f"mean FMR {np.mean(fmr):.2f}%\nmean FNMR {np.mean(fnmr):.2f}%",
        transform=right.transAxes,
        ha="right",
        va="top",
        fontweight="bold",
    )

    figure.suptitle("Frozen PCA-distance baseline fails to transfer from ORL to LFW", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "10 identity-disjoint rounds · k=80 frozen from ORL · thresholds selected on separate calibration identities",
        ha="center",
        color="#4B5563",
        fontsize=9.4,
    )
    save_clean_svg(
        figure,
        args.output_dir / "lfw_pca_distance_results.png",
        args.output_dir / "lfw_pca_distance_results.svg",
    )

    genuine_margins = np.asarray(
        [float(row["score"]) - float(row["calibration_threshold"]) for row in scores if row["target"] == "1"]
    )
    impostor_margins = np.asarray(
        [float(row["score"]) - float(row["calibration_threshold"]) for row in scores if row["target"] == "0"]
    )
    combined = np.concatenate((genuine_margins, impostor_margins))
    lower, upper = np.quantile(combined, [0.005, 0.995])
    bins = np.linspace(float(lower), float(upper), 55)
    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.76)
    axis.hist(impostor_margins, bins=bins, density=True, histtype="step", linewidth=2.2, color="#356D9A", label="Different identity")
    axis.hist(genuine_margins, bins=bins, density=True, histtype="stepfilled", alpha=0.32, linewidth=2, color="#D5523A", label="Same identity")
    axis.axvline(0, color="#111827", linestyle="--", linewidth=1.7, label="Calibration threshold")
    axis.set_xlabel("Score minus that fold's calibration threshold")
    axis.set_ylabel("Density")
    axis.set_title("Most genuine pairs fall below the strict access threshold", fontweight="bold")
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(frameon=False, loc="upper left")
    figure.suptitle("Why the LFW false rejection rate reaches 92.8%", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "Values ≥ 0 are accepted · 3,000 genuine and 3,000 impostor test pairs across 10 held-out folds",
        ha="center",
        color="#4B5563",
        fontsize=9.4,
    )
    save_clean_svg(
        figure,
        args.output_dir / "lfw_threshold_margin_distribution.png",
        args.output_dir / "lfw_threshold_margin_distribution.svg",
    )


if __name__ == "__main__":
    main()
