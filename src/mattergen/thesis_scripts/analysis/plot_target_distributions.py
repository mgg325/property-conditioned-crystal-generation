from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


RELEASE_ROOT = Path(__file__).resolve().parents[4]
MASTER_CSV = RELEASE_ROOT / "data/processed/mattergen/adapter_evaluation_master_indexfixed.csv"
DATASET_DIST_CSV = RELEASE_ROOT / "data/processed/mattergen/dataset_dielectric_scalar_distribution.csv"
OUTPUT_DIR = RELEASE_ROOT / "outputs/mattergen_figures"
OUT_1X4 = OUTPUT_DIR / "generated_distribution_by_condition_1x4.png"
OUT_2X2 = OUTPUT_DIR / "generated_distribution_by_condition_2x2.png"

PHYSICAL_DIELECTRIC_MIN = 1.0
CONDITIONS = [2, 4, 8, 12]
GUIDANCE_FACTORS = [0, 1, 2, 5]

# Match the existing target curve colors from the original plot sequence.
TARGET_COLORS = {
    2: "C1",
    4: "C2",
    8: "C0",
    12: "C4",
}

BASELINE_FILL = "#cfcfcf"
BASELINE_ALPHA = 0.45

TITLE_FONTSIZE = 15.0
LABEL_FONTSIZE = 15.0
TICK_FONTSIZE = 13.5
LEGEND_FONTSIZE = 14.0
TARGET_MARKER_FONTSIZE = 11.5
YTICKS = [0.0, 0.2, 0.4, 0.6]


def simple_kde(values, grid=None, bandwidth=None):
    values = pd.Series(values).dropna().to_numpy()
    if len(values) == 0:
        return np.array([]), np.array([])

    if grid is None:
        lo = max(PHYSICAL_DIELECTRIC_MIN, np.nanpercentile(values, 1) - 2)
        hi = np.nanpercentile(values, 99) + 2
        grid = np.linspace(lo, hi, 300)

    if grid is not None:
        grid = np.asarray(grid)
        grid = grid[grid >= PHYSICAL_DIELECTRIC_MIN]

    if bandwidth is None:
        std = np.nanstd(values)
        n = len(values)
        bandwidth = 1.06 * std * (n ** (-1 / 5)) if std > 0 and n > 1 else 1.0
        bandwidth = max(bandwidth, 0.3)

    diff = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density /= (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def extend_curve_to_zero(grid, density):
    """Presentation-only extension so the visible curve starts at x=0."""
    if len(grid) == 0:
        return np.array([0.0]), np.array([0.0])
    if grid[0] <= 0:
        return grid, density
    grid_ext = np.concatenate(([0.0], grid))
    dens_ext = np.concatenate(([0.0], density))
    return grid_ext, dens_ext


def load_inputs():
    df = pd.read_csv(MASTER_CSV)
    df["guidance_factor"] = pd.to_numeric(df["guidance_factor"])
    df["condition"] = pd.to_numeric(df["condition"])
    df["predicted_dielectric_scalar"] = pd.to_numeric(df["predicted_dielectric_scalar"])

    dataset_df = pd.read_csv(DATASET_DIST_CSV)
    if "dft_dielectric_scalar" in dataset_df.columns:
        dataset_values = dataset_df["dft_dielectric_scalar"].dropna()
    else:
        numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric column found in dataset distribution CSV.")
        dataset_values = dataset_df[numeric_cols[0]].dropna()
    return df, dataset_values


def compute_density_cache(df, dataset_values):
    x_min = 0.0
    x_max = df["predicted_dielectric_scalar"].quantile(0.99) + 2
    grid = np.linspace(PHYSICAL_DIELECTRIC_MIN, x_max, 300)

    grid_ds, dens_ds = simple_kde(dataset_values, grid=grid)
    dataset_density_max = np.nanmax(dens_ds) if len(dens_ds) else 0.0

    density_cache = {}
    generated_density_max = 0.0
    for g in GUIDANCE_FACTORS:
        sub_g = df[df["guidance_factor"] == g]
        for cond in CONDITIONS:
            vals = sub_g.loc[sub_g["condition"] == cond, "predicted_dielectric_scalar"]
            grid_c, dens_c = simple_kde(vals, grid=grid)
            density_cache[(g, cond)] = (grid_c, dens_c)
            if len(dens_c) > 0:
                generated_density_max = max(generated_density_max, np.nanmax(dens_c))

    y_max = max(dataset_density_max, generated_density_max) * 1.15
    return grid_ds, dens_ds, density_cache, x_min, x_max, y_max


def legend_handles():
    handles = [Patch(facecolor=BASELINE_FILL, edgecolor="none", alpha=BASELINE_ALPHA, label="baseline")]
    for cond in CONDITIONS:
        line = plt.Line2D([0], [0], color=TARGET_COLORS[cond], linewidth=2.2, label=f"target {cond}")
        handles.append(line)
    return handles


def style_axis(ax, show_ylabel):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks(YTICKS)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")


def draw_panel(ax, guidance_factor, show_ylabel):
    grid_ds_plot, dens_ds_plot = extend_curve_to_zero(GRID_DS, DENS_DS)
    ax.fill_between(
        grid_ds_plot,
        dens_ds_plot,
        color=BASELINE_FILL,
        alpha=BASELINE_ALPHA,
        linewidth=0,
        edgecolor="none",
        zorder=0.2,
    )

    for target in CONDITIONS:
        ax.axvline(target, linestyle="--", linewidth=1.0, alpha=0.55, color=TARGET_COLORS[target], zorder=1.0)
        ax.text(
            target,
            Y_MAX * 0.90,
            str(target),
            ha="center",
            va="top",
            fontsize=TARGET_MARKER_FONTSIZE,
            color=TARGET_COLORS[target],
            zorder=3.0,
        )

    for cond in CONDITIONS:
        grid_c, dens_c = DENSITY_CACHE[(guidance_factor, cond)]
        grid_c, dens_c = extend_curve_to_zero(grid_c, dens_c)
        ax.plot(
            grid_c,
            dens_c,
            linewidth=2.2,
            color=TARGET_COLORS[cond],
            zorder=2.5,
        )

    ax.set_title(f"Guidance factor = {guidance_factor}", fontsize=TITLE_FONTSIZE, pad=8)
    style_axis(ax, show_ylabel=show_ylabel)


def save_1x4():
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9), sharey=True)
    for i, (ax, g) in enumerate(zip(axes, GUIDANCE_FACTORS)):
        draw_panel(ax, g, show_ylabel=(i == 0))

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.07)
    handles = legend_handles()
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.76, wspace=0.22)
    fig.savefig(OUT_1X4, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.95), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (ax, g) in enumerate(zip(axes, GUIDANCE_FACTORS)):
        draw_panel(ax, g, show_ylabel=(i in (0, 2)))

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.025)
    handles = legend_handles()
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.82, hspace=0.40, wspace=0.20)
    fig.savefig(OUT_2X2, dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DF, DATASET_VALUES = load_inputs()
    GRID_DS, DENS_DS, DENSITY_CACHE, X_MIN, X_MAX, Y_MAX = compute_density_cache(DF, DATASET_VALUES)
    save_1x4()
    save_2x2()
