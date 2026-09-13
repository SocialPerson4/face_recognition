"""Create the final three-stage research evidence figure from frozen summaries."""

from __future__ import annotations

import argparse
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
        "--input", type=Path, default=Path("results/final/project_summary.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/final_research_summary"),
    )
    return parser.parse_args()


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
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    models = summary["models"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.4, 5.8))
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.23, top=0.74, wspace=0.32)
    positions = np.arange(4)

    development = 100 * np.asarray(
        [models[model]["orl"]["development_eer"] for model in MODEL_ORDER]
    )
    final = 100 * np.asarray(
        [models[model]["orl"]["final_test_eer"] for model in MODEL_ORDER]
    )
    width = 0.36
    axes[0].bar(positions - width / 2, development, width, color="#B8C1CC", label="Development")
    bars = axes[0].bar(positions + width / 2, final, width, color=COLORS, label="Final test")
    axes[0].set_xticks(positions, MODEL_LABELS)
    axes[0].set_ylabel("EER (%)")
    axes[0].set_ylim(0, 22)
    axes[0].set_title("1  Controlled baseline", fontweight="bold")
    axes[0].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8.5)
    for bar, value in zip(bars, final):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ordinary = 100 * np.asarray(
        [models[model]["orl_hard_negative_proxy"]["ordinary_fmr"] for model in MODEL_ORDER]
    )
    hard = 100 * np.asarray(
        [models[model]["orl_hard_negative_proxy"]["hard_fmr"] for model in MODEL_ORDER]
    )
    multipliers = [
        models[model]["orl_hard_negative_proxy"]["risk_multiplier"] for model in MODEL_ORDER
    ]
    axes[1].bar(positions - width / 2, ordinary, width, color="#B8C1CC", label="Ordinary")
    bars = axes[1].bar(positions + width / 2, hard, width, color="#D5523A", label="Hard proxy")
    axes[1].set_xticks(positions, MODEL_LABELS)
    axes[1].set_ylabel("FMR at frozen threshold (%)")
    axes[1].set_ylim(0, 8.2)
    axes[1].set_title("2  Similarity stress test", fontweight="bold")
    axes[1].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8.5)
    for bar, value, multiplier in zip(bars, hard, multipliers):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}%\n×{multiplier:.0f}",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
        )

    lfw_mean = 100 * np.asarray(
        [models[model]["lfw_transfer"]["eer"]["mean"] for model in MODEL_ORDER]
    )
    lfw_std = 100 * np.asarray(
        [models[model]["lfw_transfer"]["eer"]["standard_deviation"] for model in MODEL_ORDER]
    )
    bars = axes[2].bar(positions, lfw_mean, yerr=lfw_std, capsize=4, color=COLORS)
    axes[2].set_xticks(positions, MODEL_LABELS)
    axes[2].set_ylabel("10-fold mean EER (%)")
    axes[2].set_ylim(0, 44)
    axes[2].set_title("3  Unconstrained transfer", fontweight="bold")
    axes[2].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for bar, value in zip(bars, lfw_mean):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 3.8, f"{value:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    figure.suptitle("Traditional face verification: useful baseline, exposed risk, failed transfer", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "ORL final test → identity-level hard-negative proxy → identity-disjoint LFW transfer",
        ha="center",
        color="#4B5563",
        fontsize=9.8,
    )
    figure.text(
        0.5,
        0.025,
        "ORL contains no twin or kinship labels · no deployment claim",
        ha="center",
        color="#7C2D12",
        fontsize=8.8,
    )
    save_clean_svg(
        figure,
        args.output_dir / "final_research_evidence.png",
        args.output_dir / "final_research_evidence.svg",
    )


if __name__ == "__main__":
    main()
