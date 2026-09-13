"""Plot fixed-k ORL model-family screening results."""

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
            "results/experiments/orl_model_screening_k80_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_model_screening_k80_seed_20260913"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["aggregate_results"]
    labels = ["PCA\ndistance", "Logistic\nregression", "Linear\nSVM", "RBF\nSVM"]
    x = np.arange(len(rows))

    train_eer = 100.0 * np.asarray([row["mean_train_eer"] for row in rows])
    train_std = 100.0 * np.asarray([row["std_train_eer"] for row in rows])
    validation_eer = 100.0 * np.asarray(
        [row["mean_validation_eer"] for row in rows]
    )
    validation_std = 100.0 * np.asarray(
        [row["std_validation_eer"] for row in rows]
    )
    validation_auc = np.asarray([row["mean_validation_roc_auc"] for row in rows])
    validation_auc_std = np.asarray(
        [row["std_validation_roc_auc"] for row in rows]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.2, top=0.76, wspace=0.24)
    width = 0.36

    axes[0].bar(
        x - width / 2,
        train_eer,
        width,
        yerr=train_std,
        capsize=4,
        color="#9CA3AF",
        label="inner train",
    )
    axes[0].bar(
        x + width / 2,
        validation_eer,
        width,
        yerr=validation_std,
        capsize=4,
        color="#2878B5",
        label="unseen-identity validation",
    )
    axes[0].set_title("(a) Train–validation behavior")
    axes[0].set_ylabel("Mean EER across folds (%) · lower is better")
    axes[0].set_xticks(x, labels)
    axes[0].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper left")

    colors = ["#2878B5", "#B8C0CC", "#B8C0CC", "#E07A2D"]
    axes[1].bar(
        x,
        validation_auc,
        yerr=validation_auc_std,
        capsize=4,
        color=colors,
        width=0.62,
    )
    axes[1].set_title("(b) Unseen-identity ranking")
    axes[1].set_ylabel("Mean validation ROC-AUC · higher is better")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.84, 1.0)
    axes[1].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for index, value in enumerate(validation_auc):
        axes[1].text(index, value + 0.004, f"{value:.3f}", ha="center", fontsize=9)

    figure.suptitle(
        "Fixed-k model-family screening on ORL",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "k = 80 · 5 identity-disjoint inner folds · classifier defaults not tuned · outer validation/test not used",
        ha="center",
        color="#4B5563",
        fontsize=9.2,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "model_family_screening.png"
    svg_path = args.output_dir / "model_family_screening.svg"
    figure.savefig(png_path, dpi=320)
    figure.savefig(svg_path)
    plt.close(figure)

    clean_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(clean_svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
