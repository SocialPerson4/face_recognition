"""Plot linear-model C sensitivity and the RBF C-gamma search surface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_MATPLOTLIB_CACHE = Path("results/tmp/matplotlib").resolve()
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib
import matplotlib.patches as patches
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/experiments/orl_classifier_search_k80_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--distance-baseline",
        type=Path,
        default=Path(
            "results/experiments/orl_model_screening_k80_seed_20260913/summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_classifier_search_k80_seed_20260913"),
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
    baseline_payload = json.loads(args.distance_baseline.read_text(encoding="utf-8"))
    aggregates = payload["aggregate_results"]
    distance_eer = 100.0 * next(
        row["mean_validation_eer"]
        for row in baseline_payload["aggregate_results"]
        if row["model"] == "pca_distance"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9.5, 5.5))
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.2, top=0.78)
    styles = {
        "logistic_regression": ("Logistic regression", "#2878B5", "o"),
        "linear_svm": ("Linear SVM", "#E07A2D", "s"),
    }
    for family, (label, color, marker) in styles.items():
        rows = [row for row in aggregates if row["family"] == family]
        c_values = np.asarray([row["c"] for row in rows])
        means = 100.0 * np.asarray([row["mean_validation_eer"] for row in rows])
        deviations = 100.0 * np.asarray([row["std_validation_eer"] for row in rows])
        axis.errorbar(
            c_values,
            means,
            yerr=deviations,
            color=color,
            marker=marker,
            capsize=4,
            linewidth=2,
            label=label,
        )
    axis.axhline(
        distance_eer,
        color="#31945B",
        linestyle="--",
        linewidth=1.8,
        label=f"PCA distance baseline ({distance_eer:.2f}%)",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Regularization parameter C (log scale)")
    axis.set_ylabel("Mean validation EER (%) · lower is better")
    axis.set_title("Stronger regularization improves linear classifiers", fontweight="bold")
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Classifier hyperparameter sensitivity on ORL",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "k = 80 · 5 identity-disjoint inner folds · error bars: ±1 fold SD · outer validation/test not used",
        ha="center",
        color="#4B5563",
        fontsize=9.2,
    )
    save_clean_svg(
        figure,
        args.output_dir / "linear_c_sensitivity.png",
        args.output_dir / "linear_c_sensitivity.svg",
    )

    rbf_rows = [row for row in aggregates if row["family"] == "rbf_svm"]
    c_values = payload["c_values"]
    gamma_values = payload["gamma_values"]
    heatmap = np.empty((len(gamma_values), len(c_values)), dtype=np.float64)
    for row in rbf_rows:
        gamma_index = gamma_values.index(row["gamma"])
        c_index = c_values.index(row["c"])
        heatmap[gamma_index, c_index] = 100.0 * row["mean_validation_eer"]

    selection = payload["family_selections"]["rbf_svm"]
    selected = selection["selected_parameters"]
    observed_id = selection["observed_best_config"]
    observed = next(row for row in rbf_rows if row["config_id"] == observed_id)

    figure, axis = plt.subplots(figsize=(9.5, 5.9))
    figure.subplots_adjust(left=0.12, right=0.9, bottom=0.19, top=0.78)
    image = axis.imshow(heatmap, cmap="YlGnBu_r", aspect="auto")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("Mean validation EER (%) · lower is better")
    axis.set_xticks(np.arange(len(c_values)), [f"{value:g}" for value in c_values])
    axis.set_yticks(
        np.arange(len(gamma_values)), [f"{value:g}" for value in gamma_values]
    )
    axis.set_xlabel("C")
    axis.set_ylabel("gamma")
    axis.set_title("RBF-SVM C–gamma response surface", fontweight="bold")
    for row_index in range(len(gamma_values)):
        for column_index in range(len(c_values)):
            value = heatmap[row_index, column_index]
            text_color = "white" if value < 16.0 else "#111827"
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    selected_column = c_values.index(selected["c"])
    selected_row = gamma_values.index(selected["gamma"])
    observed_column = c_values.index(observed["c"])
    observed_row = gamma_values.index(observed["gamma"])
    axis.add_patch(
        patches.Rectangle(
            (selected_column - 0.48, selected_row - 0.48),
            0.96,
            0.96,
            fill=False,
            edgecolor="#00BDE3",
            linewidth=3,
            label="one-SE selected",
        )
    )
    axis.add_patch(
        patches.Rectangle(
            (observed_column - 0.4, observed_row - 0.4),
            0.8,
            0.8,
            fill=False,
            edgecolor="#E07A2D",
            linewidth=2.5,
            linestyle="--",
            label="observed minimum",
        )
    )
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2)
    figure.suptitle(
        "RBF-SVM coarse hyperparameter search on ORL",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Cell values: five-fold mean EER (%) · k = 80 · outer validation/test not used",
        ha="center",
        color="#4B5563",
        fontsize=9.2,
    )
    save_clean_svg(
        figure,
        args.output_dir / "rbf_c_gamma_heatmap.png",
        args.output_dir / "rbf_c_gamma_heatmap.svg",
    )


if __name__ == "__main__":
    main()
