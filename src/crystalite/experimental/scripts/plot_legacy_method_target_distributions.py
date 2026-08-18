"""Plot legacy Crystalite target distributions in the final MatterGen layout.

The requested panels use g=1/2/3 for additive, concat-MLP, and
additive+FiLM, and g=1/2/3/5 for residual-delta.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = ROOT / "crystalite" / "outputs" / "final report output"
OUT_ROOT = REPORT_ROOT / "figures_target_distribution_mattergen_layout"
BASELINE_CSV = (
    ROOT
    / "AnisoNet"
    / "generation_tests"
    / "Best Checkpoint generation"
    / "guidance_sweep_n128"
    / "dataset_dielectric_scalar_distribution.csv"
)

METHOD_GUIDANCES = {
    "additive": (1, 2, 3),
    "concat_mlp": (1, 2, 3),
    "residual_delta": (1, 2, 3, 5),
    "additive_plus_film": (1, 2, 3),
}
CONDITIONS = (2, 4, 8, 12)
PRED_COL = "predicted_dft_dielectric_scalar"
PHYSICAL_DIELECTRIC_MIN = 1.0

# Use the MatterGen target-curve colors.
TARGET_COLORS = {2: "C1", 4: "C2", 8: "C0", 12: "C4"}
BASELINE_FILL = "#cfcfcf"
BASELINE_ALPHA = 0.45
TITLE_FONTSIZE = 15.0
LABEL_FONTSIZE = 15.0
TICK_FONTSIZE = 13.5
LEGEND_FONTSIZE = 14.0
TARGET_MARKER_FONTSIZE = 11.5
YTICKS = (0.0, 0.2, 0.4, 0.6)


def simple_kde(values: pd.Series, grid: np.ndarray) -> np.ndarray:
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if len(values_array) == 0:
        return np.zeros_like(grid)
    std = np.nanstd(values_array)
    bandwidth = 1.06 * std * (len(values_array) ** (-1 / 5)) if len(values_array) > 1 else 1.0
    bandwidth = max(float(bandwidth), 0.3)
    diff = (grid[:, None] - values_array[None, :]) / bandwidth
    return np.exp(-0.5 * diff**2).sum(axis=1) / (
        len(values_array) * bandwidth * np.sqrt(2 * np.pi)
    )


def extend_curve_to_zero(grid: np.ndarray, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate(([0.0], grid)), np.concatenate(([0.0], density))


def load_method(method: str) -> pd.DataFrame:
    path = REPORT_ROOT / f"prediction_complete_per_sample_{method}.csv"
    frame = pd.read_csv(path)
    frame["guidance_factor"] = pd.to_numeric(frame["guidance_factor"], errors="coerce")
    frame["target_numeric"] = pd.to_numeric(frame["target_numeric"], errors="coerce")
    frame[PRED_COL] = pd.to_numeric(frame[PRED_COL], errors="coerce")
    success = frame["prediction_success"].fillna(False).astype(bool)
    return frame[success & np.isfinite(frame[PRED_COL]) & (frame[PRED_COL] >= PHYSICAL_DIELECTRIC_MIN)].copy()


def legend_handles() -> list[object]:
    handles: list[object] = [
        Patch(facecolor=BASELINE_FILL, edgecolor="none", alpha=BASELINE_ALPHA, label="baseline")
    ]
    handles.extend(
        plt.Line2D([0], [0], color=TARGET_COLORS[target], linewidth=2.2, label=f"target {target}")
        for target in CONDITIONS
    )
    return handles


def plot_method(method: str, baseline_values: pd.Series) -> Path:
    frame = load_method(method)
    guidance_factors = METHOD_GUIDANCES[method]
    x_max = frame[PRED_COL].quantile(0.99) + 2.0
    grid = np.linspace(PHYSICAL_DIELECTRIC_MIN, x_max, 300)
    baseline_density = simple_kde(baseline_values, grid)

    densities: dict[tuple[int, int], np.ndarray] = {}
    y_max = float(np.nanmax(baseline_density)) if len(baseline_density) else 0.0
    for guidance in guidance_factors:
        for target in CONDITIONS:
            values = frame.loc[
                (frame["guidance_factor"] == guidance)
                & (frame["target_numeric"] == target),
                PRED_COL,
            ]
            density = simple_kde(values, grid)
            densities[(guidance, target)] = density
            if len(values):
                y_max = max(y_max, float(np.nanmax(density)))
    y_max *= 1.15

    fig_width = 14.2 if len(guidance_factors) == 4 else 10.8
    fig, axes = plt.subplots(1, len(guidance_factors), figsize=(fig_width, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    baseline_x, baseline_y = extend_curve_to_zero(grid, baseline_density)
    for index, (axis, guidance) in enumerate(zip(axes, guidance_factors)):
        axis.fill_between(
            baseline_x, baseline_y, color=BASELINE_FILL, alpha=BASELINE_ALPHA,
            linewidth=0, edgecolor="none", zorder=0.2,
        )
        for target in CONDITIONS:
            axis.axvline(target, linestyle="--", linewidth=1.0, alpha=0.55,
                        color=TARGET_COLORS[target], zorder=1.0)
            axis.text(target, y_max * 0.90, str(target), ha="center", va="top",
                      fontsize=TARGET_MARKER_FONTSIZE, color=TARGET_COLORS[target], zorder=3.0)
            values = frame.loc[
                (frame["guidance_factor"] == guidance)
                & (frame["target_numeric"] == target),
                PRED_COL,
            ]
            if len(values):
                curve_x, curve_y = extend_curve_to_zero(grid, densities[(guidance, target)])
                axis.plot(curve_x, curve_y, linewidth=2.2, color=TARGET_COLORS[target], zorder=2.5)
        axis.set_title(f"Guidance factor = {guidance}", fontsize=TITLE_FONTSIZE, pad=8)
        axis.set_xlim(0.0, x_max)
        axis.set_ylim(0.0, y_max)
        axis.set_yticks(YTICKS)
        axis.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
        axis.set_ylabel("Density" if index == 0 else "", fontsize=LABEL_FONTSIZE)

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.07)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01),
               frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=1.6,
               handletextpad=0.6, columnspacing=1.3)
    fig.subplots_adjust(left=0.08 if len(guidance_factors) == 3 else 0.06, right=0.995,
                        bottom=0.22, top=0.76, wspace=0.22)

    method_dir = OUT_ROOT / method
    method_dir.mkdir(parents=True, exist_ok=True)
    output = method_dir / "generated_distribution_by_condition_1x4.png"
    fig.savefig(output, dpi=350, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    baseline = pd.read_csv(BASELINE_CSV)["dft_dielectric_scalar"]
    for method in METHOD_GUIDANCES:
        print(plot_method(method, baseline))


if __name__ == "__main__":
    main()
