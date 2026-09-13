"""Plot the reproducible PCA-k coarse-search development results."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/experiments/orl_pca_k_coarse_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_pca_k_coarse_seed_20260913"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["aggregate_results"]

    k = np.asarray([row["k"] for row in rows])
    mean_eer = 100.0 * np.asarray([row["mean_eer"] for row in rows])
    std_eer = 100.0 * np.asarray([row["std_eer"] for row in rows])
    mean_auc = np.asarray([row["mean_roc_auc"] for row in rows])
    std_auc = np.asarray([row["std_roc_auc"] for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue = "#2878B5"
    orange = "#E07A2D"
    grid = "#D9DEE7"
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.78, wspace=0.18)

    axes[0].plot(k, mean_eer, color=blue, marker="o", linewidth=2.2)
    axes[0].fill_between(
        k, mean_eer - std_eer, mean_eer + std_eer, color=blue, alpha=0.16
    )
    axes[0].set_title("(a) Equal error rate")
    axes[0].set_xlabel("PCA dimensions (k)")
    axes[0].set_ylabel("Mean EER across identity folds (%)")
    axes[0].set_xticks(k)
    axes[0].grid(axis="y", color=grid, linewidth=0.8)
    axes[0].annotate(
        "performance plateau",
        xy=(120, mean_eer[k.tolist().index(120)]),
        xytext=(75, 14.7),
        arrowprops={"arrowstyle": "->", "color": "#4B5563"},
        color="#374151",
    )

    axes[1].plot(k, mean_auc, color=orange, marker="o", linewidth=2.2)
    axes[1].fill_between(
        k, mean_auc - std_auc, mean_auc + std_auc, color=orange, alpha=0.16
    )
    axes[1].set_title("(b) ROC-AUC")
    axes[1].set_xlabel("PCA dimensions (k)")
    axes[1].set_ylabel("Mean ROC-AUC across identity folds")
    axes[1].set_xticks(k)
    axes[1].grid(axis="y", color=grid, linewidth=0.8)

    figure.suptitle(
        "PCA dimension sensitivity on ORL (5 identity-disjoint folds)",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Shaded bands: ±1 fold standard deviation · outer validation/test not used",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "pca_k_coarse.png"
    svg_path = args.output_dir / "pca_k_coarse.svg"
    figure.savefig(png_path, dpi=320)
    figure.savefig(svg_path)
    plt.close(figure)

    # Matplotlib's SVG writer leaves spaces at line endings.  Removing them
    # keeps the generated artifact friendly to repository whitespace checks.
    clean_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(clean_svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
