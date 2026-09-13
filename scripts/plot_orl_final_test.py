"""Plot final ORL test comparisons and ROC curves from saved scores."""

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
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = ("pca_distance", "logistic_regression", "linear_svm", "rbf_svm")
MODEL_LABELS = {
    "pca_distance": "PCA distance",
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF-SVM",
}
SHORT_LABELS = {
    "pca_distance": "PCA\ndistance",
    "logistic_regression": "Logistic\nregression",
    "linear_svm": "Linear\nSVM",
    "rbf_svm": "RBF\nSVM",
}
COLORS = {
    "pca_distance": "#31945B",
    "logistic_regression": "#2878B5",
    "linear_svm": "#E07A2D",
    "rbf_svm": "#7953A9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/experiments/orl_final_test_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path(
            "results/experiments/orl_final_test_seed_20260913/test_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_final_test_seed_20260913"),
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


def load_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    scores = {
        model: np.asarray(
            [float(row[f"{model}_score"]) for row in rows], dtype=np.float64
        )
        for model in MODEL_ORDER
    }
    return labels, scores


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    by_model = {row["model"]: row for row in payload["results"]}
    labels, scores = load_scores(args.scores)
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
    validation_eer = np.asarray(
        [100.0 * by_model[model]["validation_eer"] for model in MODEL_ORDER]
    )
    test_eer = np.asarray(
        [100.0 * by_model[model]["test_eer"]["eer"] for model in MODEL_ORDER]
    )
    test_fnmr = np.asarray(
        [
            100.0 * by_model[model]["test_rates_at_frozen_threshold"]["fnmr"]
            for model in MODEL_ORDER
        ]
    )
    test_fmr = np.asarray(
        [
            100.0 * by_model[model]["test_rates_at_frozen_threshold"]["fmr"]
            for model in MODEL_ORDER
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.8))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.76, wspace=0.3)
    width = 0.34
    axes[0].bar(
        positions - width / 2,
        validation_eer,
        width,
        color="#B8C0CC",
        label="Development validation",
    )
    test_bars = axes[0].bar(
        positions + width / 2,
        test_eer,
        width,
        color=[COLORS[model] for model in MODEL_ORDER],
        label="Final test",
    )
    axes[0].set_xticks(positions, [SHORT_LABELS[model] for model in MODEL_ORDER])
    axes[0].set_ylabel("EER (%) · lower is better")
    axes[0].set_title("Development ranking did not persist", fontweight="bold")
    axes[0].set_ylim(0, max(validation_eer) + 4.5)
    axes[0].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")
    for bar, value in zip(test_bars, test_eer):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            fontsize=9,
        )

    axes[1].bar(
        positions,
        test_fnmr,
        color=[COLORS[model] for model in MODEL_ORDER],
        width=0.68,
    )
    axes[1].set_xticks(positions, [SHORT_LABELS[model] for model in MODEL_ORDER])
    axes[1].set_ylabel("Test FNMR at frozen threshold (%)")
    axes[1].set_title("Validation-set threshold transferred to test", fontweight="bold")
    axes[1].set_ylim(0, max(test_fnmr) + 10)
    axes[1].grid(axis="y", color="#D9DEE7", linewidth=0.8)
    for index, (fnmr, fmr) in enumerate(zip(test_fnmr, test_fmr)):
        axes[1].text(
            index,
            fnmr + 0.8,
            f"FNMR {fnmr:.2f}%\nFMR {fmr:.2f}%",
            ha="center",
            fontsize=8.8,
            fontweight="bold" if index == 0 else "normal",
        )

    figure.suptitle("One-time frozen final test on ORL", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.065,
        "360 genuine + 360 impostor pairs · models and thresholds frozen before test",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )
    figure.text(
        0.5,
        0.03,
        "RBF-SVM remains the preregistered development candidate; test ranking is not used for reselection",
        ha="center",
        color="#7C2D12",
        fontsize=8.8,
    )
    save_clean_svg(
        figure,
        args.output_dir / "final_test_comparison.png",
        args.output_dir / "final_test_comparison.svg",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.6))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.76, wspace=0.27)
    for model in MODEL_ORDER:
        fmr, true_accept_rate, _ = roc_curve(labels, scores[model])
        auc = by_model[model]["test_roc_auc"]
        label = f"{MODEL_LABELS[model]} · AUC {auc:.4f}"
        for axis in axes:
            axis.plot(fmr, true_accept_rate, color=COLORS[model], linewidth=2, label=label)
        fixed = by_model[model]["test_rates_at_frozen_threshold"]
        axes[1].scatter(
            fixed["fmr"],
            fixed["true_accept_rate"],
            color=COLORS[model],
            s=48,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    axes[0].plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", linewidth=1.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Full ROC curve", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[1].set_xlim(0, 0.05)
    axes[1].set_ylim(0.35, 0.75)
    axes[1].set_title("Low-FMR region · dots use frozen thresholds", fontweight="bold")
    for axis in axes:
        axis.set_xlabel("False match rate (FMR)")
        axis.set_ylabel("True accept rate (1 − FNMR)")
        axis.grid(color="#D9DEE7", linewidth=0.8)
    figure.suptitle("Final-test ROC and access-control operating points", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.045,
        "Curves describe test ranking; operating-point dots use thresholds selected only on M4 validation data",
        ha="center",
        color="#4B5563",
        fontsize=9.2,
    )
    save_clean_svg(
        figure,
        args.output_dir / "final_test_roc.png",
        args.output_dir / "final_test_roc.svg",
    )


if __name__ == "__main__":
    main()
