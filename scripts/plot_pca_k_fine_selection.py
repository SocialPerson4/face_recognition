"""Visualize the one-standard-error decision for the PCA distance baseline."""

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
        default=Path("results/experiments/orl_pca_k_fine_seed_20260913/summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/orl_pca_k_fine_seed_20260913"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["aggregate_results"]
    selection = payload["one_standard_error_selection"]
    if selection is None:
        raise ValueError("input does not contain a one-standard-error selection")

    k = np.asarray([row["k"] for row in rows], dtype=np.int64)
    mean = 100.0 * np.asarray([row["mean_eer"] for row in rows])
    standard_error = 100.0 * np.asarray(
        [row["std_eer"] for row in rows]
    ) / np.sqrt(payload["inner_folds"])
    cutoff = 100.0 * float(selection["eligibility_cutoff"])
    selected_k = int(selection["selected_k"])
    observed_best_k = int(selection["observed_best_k"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axis = plt.subplots(figsize=(10.8, 5.4))
    figure.subplots_adjust(left=0.1, right=0.97, bottom=0.19, top=0.78)

    axis.axhspan(0.0, cutoff, color="#4CAF76", alpha=0.10)
    axis.axhline(
        cutoff,
        color="#31945B",
        linestyle="--",
        linewidth=1.8,
        label=f"one-SE eligibility cutoff = {cutoff:.2f}%",
    )
    axis.errorbar(
        k,
        mean,
        yerr=standard_error,
        color="#6B7280",
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=4,
        label="mean EER ± 1 standard error",
    )

    selected_index = int(np.flatnonzero(k == selected_k)[0])
    best_index = int(np.flatnonzero(k == observed_best_k)[0])
    axis.scatter(
        [selected_k],
        [mean[selected_index]],
        marker="*",
        s=230,
        color="#2878B5",
        zorder=5,
        label=f"selected simplest k = {selected_k}",
    )
    axis.scatter(
        [observed_best_k],
        [mean[best_index]],
        marker="D",
        s=75,
        color="#E07A2D",
        zorder=5,
        label=f"observed minimum starts at k = {observed_best_k}",
    )

    axis.set_xlabel("PCA dimensions (k)")
    axis.set_ylabel("Mean EER across identity folds (%)")
    axis.set_xticks(k)
    axis.set_ylim(7.2, 11.35)
    axis.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    axis.legend(loc="lower left", frameon=False, ncol=2)
    figure.suptitle(
        "PCA dimension selection using the one-standard-error rule",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Fine search on 5 identity-disjoint inner folds · outer validation/test not used",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "pca_k_one_standard_error.png"
    svg_path = args.output_dir / "pca_k_one_standard_error.svg"
    figure.savefig(png_path, dpi=320)
    figure.savefig(svg_path)
    plt.close(figure)

    clean_svg = "\n".join(
        line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()
    )
    svg_path.write_text(clean_svg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
