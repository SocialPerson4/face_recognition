"""Plot identity-level hard-negative proxy evidence from saved ORL results."""

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
import matplotlib.patches as patches
import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiments/orl_hard_negative_proxy_seed_20260913"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_hard_negative_proxy_seed_20260913"),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    ranking = read_csv(args.input_dir / "identity_pair_ranking.csv")
    score_rows = read_csv(args.input_dir / "negative_scores.csv")
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
    width = 0.34
    ordinary = np.asarray(
        [
            100.0 * row["groups"]["ordinary"]["false_match_rate"]
            for row in summary["model_results"]
        ]
    )
    hard = np.asarray(
        [
            100.0 * row["groups"]["hard_identity_proxy"]["false_match_rate"]
            for row in summary["model_results"]
        ]
    )
    multipliers = [
        row["hard_to_ordinary_fmr_multiplier"] for row in summary["model_results"]
    ]
    figure, axis = plt.subplots(figsize=(9.8, 5.8))
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.21, top=0.77)
    ordinary_bars = axis.bar(
        positions - width / 2,
        ordinary,
        width,
        color="#9CA3AF",
        label="Ordinary negatives",
    )
    hard_bars = axis.bar(
        positions + width / 2,
        hard,
        width,
        color="#D5523A",
        label="Hard identity proxy",
    )
    axis.set_xticks(positions, [SHORT_LABELS[model] for model in MODEL_ORDER])
    axis.set_ylabel("False match rate at frozen threshold (%)")
    axis.set_title("Identity-level similarity concentrates false accepts", fontweight="bold")
    axis.set_ylim(0, max(hard) + 2.4)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(frameon=False, loc="upper left")
    for bar, value in zip(ordinary_bars, ordinary):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.14,
            f"{value:.2f}%",
            ha="center",
            fontsize=8.5,
        )
    for bar, value, multiplier in zip(hard_bars, hard, multipliers):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.14,
            f"{value:.2f}%\n×{multiplier:.1f}",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
        )
    figure.suptitle("ORL high-similarity negative proxy stress test", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "700 pairs per group · frozen M4 thresholds · validation identities only",
        ha="center",
        color="#4B5563",
        fontsize=9.4,
    )
    figure.text(
        0.5,
        0.025,
        "Directed PCA-space stress test · ORL has no twin or kinship labels",
        ha="center",
        color="#7C2D12",
        fontsize=8.8,
    )
    save_clean_svg(
        figure,
        args.output_dir / "hard_negative_fmr_risk.png",
        args.output_dir / "hard_negative_fmr_risk.svg",
    )

    identities = sorted(
        {int(row["left_identity"]) for row in ranking}
        | {int(row["right_identity"]) for row in ranking}
    )
    identity_index = {identity: index for index, identity in enumerate(identities)}
    matrix = np.full((len(identities), len(identities)), np.nan)
    hard_cells: list[tuple[int, int]] = []
    for row in ranking:
        left = identity_index[int(row["left_identity"])]
        right = identity_index[int(row["right_identity"])]
        distance = float(row["centroid_distance"])
        matrix[left, right] = distance
        matrix[right, left] = distance
        if row["is_hard_identity_pair"] == "True":
            hard_cells.extend(((left, right), (right, left)))
    figure, axis = plt.subplots(figsize=(8.4, 6.8))
    figure.subplots_adjust(left=0.13, right=0.88, bottom=0.15, top=0.79)
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="equal")
    colorbar = figure.colorbar(image, ax=axis, pad=0.03)
    colorbar.set_label("Distance between identity mean PCA embeddings")
    axis.set_xticks(np.arange(len(identities)), identities)
    axis.set_yticks(np.arange(len(identities)), identities)
    axis.set_xlabel("Validation identity")
    axis.set_ylabel("Validation identity")
    axis.set_title("Closest identity pairs define the stress-test tail", fontweight="bold")
    for row, column in hard_cells:
        axis.add_patch(
            patches.Rectangle(
                (column - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#E64B35",
                linewidth=2.2,
            )
        )
    figure.suptitle("Identity-level appearance similarity matrix", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.045,
        "Red outlines: closest 7 of 28 identity pairs (top similarity quartile) · lower distance means more similar",
        ha="center",
        color="#4B5563",
        fontsize=9.1,
    )
    save_clean_svg(
        figure,
        args.output_dir / "identity_similarity_matrix.png",
        args.output_dir / "identity_similarity_matrix.svg",
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.12, top=0.84, hspace=0.42, wspace=0.25)
    for axis, model in zip(axes.flat, MODEL_ORDER):
        ordinary_scores = np.asarray(
            [
                float(row[f"{model}_score"])
                for row in score_rows
                if row["group"] == "ordinary"
            ]
        )
        hard_scores = np.asarray(
            [
                float(row[f"{model}_score"])
                for row in score_rows
                if row["group"] == "hard_identity_proxy"
            ]
        )
        combined = np.concatenate((ordinary_scores, hard_scores))
        bins = np.linspace(float(np.min(combined)), float(np.max(combined)), 31)
        axis.hist(
            ordinary_scores,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color="#4B5563",
            label="Ordinary",
        )
        axis.hist(
            hard_scores,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.3,
            linewidth=2,
            color="#D5523A",
            label="Hard proxy",
        )
        threshold = next(
            row["frozen_threshold"]
            for row in summary["model_results"]
            if row["model"] == model
        )
        axis.axvline(
            threshold,
            color="#111827",
            linestyle="--",
            linewidth=1.5,
            label="Frozen threshold",
        )
        axis.set_title(MODEL_LABELS[model], fontweight="bold")
        axis.set_xlabel("Similarity score · higher is more likely same person")
        axis.set_ylabel("Density")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Hard identity proxies shift impostor scores toward acceptance", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.035,
        "Distributions use different score scales by model; compare ordinary vs hard within each panel",
        ha="center",
        color="#4B5563",
        fontsize=9.2,
    )
    save_clean_svg(
        figure,
        args.output_dir / "hard_negative_score_distributions.png",
        args.output_dir / "hard_negative_score_distributions.svg",
    )


if __name__ == "__main__":
    main()
