"""Plot LFW protocol overlap evidence and the frozen rotating evaluation design."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/audit/lfw_protocol_overlap.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/lfw_protocol_design"),
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
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.2, 5.8))
    figure.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.79, wspace=0.28)

    cross = data["cross_protocol_overlap"]
    labels = ["Dev train\nvs dev test", "Dev train\nvs 10-fold", "Dev test\nvs 10-fold"]
    keys = ["dev_train_vs_dev_test", "dev_train_vs_benchmark", "dev_test_vs_benchmark"]
    identities = [cross[key]["shared_identities_count"] for key in keys]
    images = [cross[key]["shared_images_count"] for key in keys]
    positions = range(3)
    width = 0.36
    bars_a = left.bar(
        [value - width / 2 for value in positions],
        identities,
        width,
        color="#356D9A",
        label="Shared identities",
    )
    bars_b = left.bar(
        [value + width / 2 for value in positions],
        images,
        width,
        color="#D98E32",
        label="Shared images",
    )
    left.set_xticks(list(positions), labels)
    left.set_ylabel("Overlap count")
    left.set_title("Development files overlap the benchmark", fontweight="bold")
    left.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    left.legend(frameon=False, loc="upper left")
    for bar in (*bars_a, *bars_b):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 45,
            f"{int(bar.get_height()):,}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    right.set_xlim(0.3, 10.7)
    right.set_ylim(0, 3.4)
    right.axis("off")
    right.set_title("Frozen 8 / 1 / 1 identity-disjoint rotation", fontweight="bold")
    colors = {"train": "#356D9A", "calibrate": "#E9B44C", "test": "#D5523A"}
    roles = ["test", "calibrate", *("train" for _ in range(8))]
    for fold, role in enumerate(roles, start=1):
        right.add_patch(
            patches.FancyBboxPatch(
                (fold - 0.42, 1.55),
                0.84,
                0.72,
                boxstyle="round,pad=0.03,rounding_size=0.06",
                facecolor=colors[role],
                edgecolor="white",
                linewidth=1.5,
            )
        )
        right.text(fold, 1.91, str(fold), color="white", ha="center", va="center", fontweight="bold")
    right.annotate(
        "rotate roles across 10 rounds",
        xy=(9.9, 1.25),
        xytext=(1.0, 1.25),
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.6},
        ha="left",
        color="#374151",
    )
    legend_items = (
        ("Train: 8 folds", "train"),
        ("Threshold: 1 fold", "calibrate"),
        ("Final score: 1 fold", "test"),
    )
    for index, (label, role) in enumerate(legend_items):
        x = 1.1 + index * 3.15
        right.add_patch(patches.Rectangle((x, 0.48), 0.34, 0.22, color=colors[role]))
        right.text(x + 0.45, 0.59, label, va="center", fontsize=9.4)
    right.text(
        5.5,
        2.75,
        "No identity, image, or exact-pair overlap\nbetween a held-out fold and the other nine",
        ha="center",
        va="center",
        color="#1F5132",
        fontsize=10.3,
        fontweight="bold",
    )

    figure.suptitle("LFW evaluation protocol: preserve unseen-identity evidence", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.055,
        "Primary experiment uses pairs.txt only · ORL-selected hyperparameters remain frozen",
        ha="center",
        color="#4B5563",
        fontsize=9.6,
    )
    save_clean_svg(
        figure,
        args.output_dir / "lfw_protocol_design.png",
        args.output_dir / "lfw_protocol_design.svg",
    )


if __name__ == "__main__":
    main()
