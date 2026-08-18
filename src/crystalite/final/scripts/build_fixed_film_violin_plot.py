from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
PER_SAMPLE_CSV = ROOT / "outputs" / "final report output" / "evaluation" / "per_sample_evaluation.csv"
OUT_PATH = (
    ROOT
    / "outputs"
    / "final report output"
    / "figures_prediction_complete"
    / "fixed_film"
    / "fixed_film_violin_by_guidance_panels.png"
)

TARGETS = ["2", "4", "8", "12"]
GUIDANCES = ["1", "2", "3", "4", "5"]
TARGET_COLORS = {
    "2": "#d1495b",
    "4": "#5e548e",
    "8": "#2a9d8f",
    "12": "#f4a261",
}


def load_values() -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with PER_SAMPLE_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row["group"].startswith("fixed_film__g"):
                continue
            if row["guidance_factor"] not in GUIDANCES or row["target"] not in TARGETS:
                continue
            if row["prediction_success"] != "True":
                continue
            try:
                value = float(row["predicted_dft_dielectric_scalar"])
            except Exception:
                continue
            if math.isfinite(value):
                values[(row["guidance_factor"], row["target"])].append(value)
    return values


def build_plot(values: dict[tuple[str, str], list[float]]) -> None:
    fig, axes = plt.subplots(1, len(GUIDANCES), figsize=(20, 4.8), sharey=True)

    if len(GUIDANCES) == 1:
        axes = [axes]

    for ax, guidance in zip(axes, GUIDANCES):
        datasets = []
        positions = []
        colors = []
        tick_labels = []
        for idx, target in enumerate(TARGETS, start=1):
            dataset = values.get((guidance, target), [])
            if not dataset:
                continue
            datasets.append(dataset)
            positions.append(idx)
            colors.append(TARGET_COLORS[target])
            tick_labels.append(f"t={target}")

        if datasets:
            parts = ax.violinplot(
                datasets,
                positions=positions,
                widths=0.75,
                showmeans=False,
                showextrema=False,
                showmedians=True,
            )
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_edgecolor("black")
                body.set_alpha(0.72)
                body.set_linewidth(0.8)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.0)

        for idx, target in enumerate(TARGETS, start=1):
            ax.axhline(float(target), color=TARGET_COLORS[target], linestyle="--", linewidth=0.8, alpha=0.35)

        ax.set_title(f"g={guidance}")
        ax.set_xticks(range(1, len(TARGETS) + 1), [f"t={target}" for target in TARGETS], rotation=0)
        ax.set_yscale("symlog", linthresh=10.0)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Predicted dielectric scalar")
    fig.suptitle("fixed_film violin plots by guidance factor", y=1.02)

    legend_handles = [
        Patch(facecolor=TARGET_COLORS[target], edgecolor="black", label=f"target={target}")
        for target in TARGETS
    ]
    fig.legend(handles=legend_handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    values = load_values()
    build_plot(values)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
