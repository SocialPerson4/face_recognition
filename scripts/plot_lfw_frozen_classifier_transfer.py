"""Plot the saved LFW frozen-classifier transfer comparison."""

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


MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
MODEL_LABELS = ("PCA\ndistance", "Logistic\nregression", "Linear\nSVM", "RBF\nSVM")
COLORS = ("#64748B", "#2A9D8F", "#356D9A", "#D98E32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiments/lfw_frozen_classifier_transfer_8_1_1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/lfw_frozen_classifier_transfer_8_1_1"),
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
    rows = read_csv(args.input_dir / "fold_results.csv")
    aggregates = summary["aggregate_test_metrics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    positions = np.arange(len(MODEL_ORDER))
    eer_means = 100 * np.asarray([aggregates[model]["eer"]["mean"] for model in MODEL_ORDER])
    eer_stds = 100 * np.asarray([aggregates[model]["eer"]["standard_deviation"] for model in MODEL_ORDER])
    fnmr_means = 100 * np.asarray([aggregates[model]["fnmr"]["mean"] for model in MODEL_ORDER])
    fnmr_stds = 100 * np.asarray([aggregates[model]["fnmr"]["standard_deviation"] for model in MODEL_ORDER])
    fmr_means = 100 * np.asarray([aggregates[model]["fmr"]["mean"] for model in MODEL_ORDER])

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.8, 5.8))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.76, wspace=0.3)
    bars = left.bar(positions, eer_means, yerr=eer_stds, capsize=5, color=COLORS)
    left.set_xticks(positions, MODEL_LABELS)
    left.set_ylabel("10-fold mean EER (%)")
    left.set_ylim(0, 46)
    left.set_title("Supervision does not improve overall separation", fontweight="bold")
    left.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for bar, value in zip(bars, eer_means):
        left.text(bar.get_x() + bar.get_width() / 2, value + 4.1, f"{value:.2f}%", ha="center", fontweight="bold")

    bars = right.bar(positions, fnmr_means, yerr=fnmr_stds, capsize=5, color=COLORS)
    right.set_xticks(positions, MODEL_LABELS)
    right.set_ylabel("FNMR at independently calibrated threshold (%)")
    right.set_ylim(0, 102)
    right.set_title("Strict access remains unusable for every model", fontweight="bold")
    right.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for bar, fnmr, fmr in zip(bars, fnmr_means, fmr_means):
        right.text(
            bar.get_x() + bar.get_width() / 2,
            fnmr - 8,
            f"FNMR {fnmr:.1f}%\nFMR {fmr:.2f}%",
            ha="center",
            color="white",
            fontsize=8.8,
            fontweight="bold",
        )

    figure.suptitle("ORL-frozen classifiers do not rescue the LFW PCA representation", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "Same 10 identity-disjoint folds · same k=80 representation · no LFW hyperparameter search",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )
    save_clean_svg(
        figure,
        args.output_dir / "lfw_frozen_model_comparison.png",
        args.output_dir / "lfw_frozen_model_comparison.svg",
    )

    distance = {
        int(row["test_fold"]): float(row["test_eer"])
        for row in rows
        if row["model"] == "pca_distance"
    }
    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.76)
    for model, label, color in zip(MODEL_ORDER[1:], MODEL_LABELS[1:], COLORS[1:]):
        model_rows = [row for row in rows if row["model"] == model]
        folds = np.asarray([int(row["test_fold"]) for row in model_rows])
        deltas = 100 * np.asarray(
            [float(row["test_eer"]) - distance[int(row["test_fold"])] for row in model_rows]
        )
        axis.plot(folds, deltas, marker="o", linewidth=1.8, color=color, label=label.replace("\n", " "))
    axis.axhline(0, color="#111827", linestyle="--", linewidth=1.4)
    axis.fill_between([0.7, 10.3], -0.15, 0, color="#2A9D8F", alpha=0.08)
    axis.set_xlim(0.7, 10.3)
    axis.set_xticks(range(1, 11))
    axis.set_xlabel("Held-out identity fold")
    axis.set_ylabel("Classifier EER minus distance EER (percentage points)")
    axis.set_title(
        "Small gains in a few folds do not persist across identities",
        fontweight="bold",
    )
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.suptitle("Paired fold evidence: no stable classifier advantage", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "Below zero favors the classifier · all 10 folds are retained",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )
    save_clean_svg(
        figure,
        args.output_dir / "lfw_paired_eer_differences.png",
        args.output_dir / "lfw_paired_eer_differences.svg",
    )


if __name__ == "__main__":
    main()
