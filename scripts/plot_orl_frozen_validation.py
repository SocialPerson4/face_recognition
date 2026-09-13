"""Plot frozen-candidate ORL development-validation results."""

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


MODEL_LABELS = {
    "pca_distance": "PCA\ndistance",
    "logistic_regression": "Logistic\nregression",
    "linear_svm": "Linear\nSVM",
    "rbf_svm": "RBF\nSVM",
}
COLORS = ("#31945B", "#2878B5", "#E07A2D", "#7953A9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/experiments/orl_frozen_validation_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_frozen_validation_seed_20260913"),
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
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["results"]
    labels = [MODEL_LABELS[row["model"]] for row in results]
    validation_eer = np.asarray(
        [100.0 * row["validation"]["eer"]["eer"] for row in results]
    )
    validation_auc = np.asarray(
        [row["validation"]["roc_auc"] for row in results]
    )
    low_fmr_fnmr = np.asarray(
        [
            100.0 * row["validation"]["working_point"]["rates"]["fnmr"]
            for row in results
        ]
    )
    positions = np.arange(len(results))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.76, wspace=0.3)

    axes[0].bar(positions, validation_eer, color=COLORS, width=0.68)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Validation EER (%) · lower is better")
    axes[0].set_title("Overall verification trade-off", fontweight="bold")
    axes[0].set_ylim(0.0, max(validation_eer) + 4.2)
    axes[0].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for index, (eer, auc) in enumerate(zip(validation_eer, validation_auc)):
        marker = "★ " if results[index]["model"] == payload["development_winner"] else ""
        axes[0].text(
            index,
            eer + 0.35,
            f"{marker}{eer:.2f}%\nAUC {auc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if marker else "normal",
        )

    axes[1].bar(positions, low_fmr_fnmr, color=COLORS, width=0.68)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("FNMR at FMR ≤ 1% (%) · lower is better")
    axes[1].set_title("Strict access-control operating point", fontweight="bold")
    axes[1].set_ylim(0.0, max(low_fmr_fnmr) + 10.0)
    axes[1].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    best_fnmr_index = int(np.argmin(low_fmr_fnmr))
    for index, value in enumerate(low_fmr_fnmr):
        marker = "★ " if index == best_fnmr_index else ""
        axes[1].text(
            index,
            value + 0.8,
            f"{marker}{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if marker else "normal",
        )

    figure.suptitle(
        "Frozen-model development validation on ORL",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.07,
        "24 training identities → 8 validation identities · frozen k/C/gamma · final 8 test identities untouched",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )
    figure.text(
        0.5,
        0.035,
        "Disclosure: validation identities were viewed in the early M2B distance exploration; not used for M3 tuning",
        ha="center",
        color="#7C2D12",
        fontsize=8.7,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_clean_svg(
        figure,
        args.output_dir / "frozen_model_validation.png",
        args.output_dir / "frozen_model_validation.svg",
    )


if __name__ == "__main__":
    main()
