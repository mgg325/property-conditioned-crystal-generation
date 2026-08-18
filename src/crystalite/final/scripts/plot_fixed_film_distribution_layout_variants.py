from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT_ROOT = ROOT / "outputs" / "final report output"
ANISONET_SPLIT_ROOT = Path("external/anisonet/splits")

CSV_PATH = FINAL_REPORT_ROOT / "prediction_complete_per_sample_fixed_film.csv"
G0_CSV_PATH = ROOT / "outputs" / "results" / "crystalite_g0" / "anisonet_evaluation" / "per_sample_evaluation.csv"
OUT_DIR = FINAL_REPORT_ROOT / "figures_prediction_complete" / "reference_style_panels"
OUT_1X4 = OUT_DIR / "fixed_film_generated_distribution_by_condition_1x4.png"
OUT_2X2 = OUT_DIR / "fixed_film_generated_distribution_by_condition_2x2.png"
OUT_0125_1X4 = OUT_DIR / "fixed_film_generated_distribution_g0125_1x4.png"
OUT_01235_3X2 = OUT_DIR / "fixed_film_generated_distribution_g01235_3x2.png"

TARGETS = [2, 4, 8, 12]
GUIDANCE_FACTORS = [1, 2, 3, 5]
GUIDANCE_FACTORS_WITH_G0 = [0, 1, 2, 3, 5]
PRED_COL = "predicted_dft_dielectric_scalar"
PHYSICAL_DIELECTRIC_MIN = 1.0

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
YTICKS = [0.0, 0.2, 0.4, 0.6, 0.8]


def simple_kde(values, grid=None, bandwidth=None):
    values = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.array([]), np.array([])

    if grid is None:
        lo = max(PHYSICAL_DIELECTRIC_MIN, np.nanpercentile(values, 1) - 2)
        hi = np.nanpercentile(values, 99) + 2
        grid = np.linspace(lo, hi, 300)

    grid = np.asarray(grid, dtype=float)
    grid = grid[grid >= PHYSICAL_DIELECTRIC_MIN]

    if bandwidth is None:
        std = np.nanstd(values)
        n = len(values)
        bandwidth = 1.06 * std * (n ** (-1 / 5)) if std > 0 and n > 1 else 1.0
        bandwidth = max(bandwidth, 0.3)

    diff = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)
    return grid, density


def extend_curve_to_zero(grid, density):
    if len(grid) == 0:
        return np.array([0.0]), np.array([0.0])
    if grid[0] <= 0:
        return grid, density
    return np.concatenate(([0.0], grid)), np.concatenate(([0.0], density))


def load_dataset_values():
    frames = []
    for split in ("train.csv", "val.csv", "test.csv"):
        path = ANISONET_SPLIT_ROOT / split
        if path.exists():
            frames.append(pd.read_csv(path, usecols=["dft_dielectric_scalar"]))
    if not frames:
        raise FileNotFoundError(f"No dataset CSVs found under {ANISONET_SPLIT_ROOT}")
    values = pd.concat(frames, ignore_index=True)["dft_dielectric_scalar"]
    values = pd.to_numeric(values, errors="coerce")
    return values[np.isfinite(values)]


def load_frame():
    frame = pd.read_csv(CSV_PATH)
    # Add the later target-labelled g=0 evaluator output.
    if G0_CSV_PATH.exists():
        frame = pd.concat([frame, pd.read_csv(G0_CSV_PATH)], ignore_index=True, sort=False)
    if "prediction_success" in frame.columns:
        pred_ok = frame["prediction_success"].fillna(False).astype(bool)
    else:
        pred_ok = pd.Series(True, index=frame.index)

    frame["guidance_factor"] = pd.to_numeric(frame["guidance_factor"], errors="coerce")
    frame["target_numeric"] = pd.to_numeric(frame["target"], errors="coerce")
    frame[PRED_COL] = pd.to_numeric(frame[PRED_COL], errors="coerce")
    frame = frame[
        pred_ok
        & frame["guidance_factor"].notna()
        & frame[PRED_COL].notna()
        & np.isfinite(frame[PRED_COL])
        & frame["guidance_factor"].isin(GUIDANCE_FACTORS_WITH_G0)
        & frame["target_numeric"].isin(TARGETS)
    ].copy()
    return frame


def compute_density_cache(frame, dataset_values):
    x_min = 0.0
    x_max = float(np.nanpercentile(frame[PRED_COL], 99)) + 2.0
    grid = np.linspace(PHYSICAL_DIELECTRIC_MIN, x_max, 300)

    grid_ds, dens_ds = simple_kde(dataset_values, grid=grid)
    dataset_density_max = float(np.nanmax(dens_ds)) if len(dens_ds) else 0.0

    density_cache = {}
    generated_density_max = 0.0
    for guidance in GUIDANCE_FACTORS_WITH_G0:
        sub_g = frame[np.isclose(frame["guidance_factor"], float(guidance), equal_nan=False)]
        for target in TARGETS:
            vals = sub_g.loc[
                np.isclose(sub_g["target_numeric"], float(target), equal_nan=False),
                PRED_COL,
            ]
            grid_c, dens_c = simple_kde(vals, grid=grid)
            density_cache[(guidance, target)] = (grid_c, dens_c)
            if len(dens_c) > 0:
                generated_density_max = max(generated_density_max, float(np.nanmax(dens_c)))

    y_max = max(dataset_density_max, generated_density_max) * 1.15
    return grid_ds, dens_ds, density_cache, x_min, x_max, y_max


def legend_handles():
    handles = [Patch(facecolor=BASELINE_FILL, edgecolor="none", alpha=BASELINE_ALPHA, label="baseline")]
    for target in TARGETS:
        handles.append(
            plt.Line2D([0], [0], color=TARGET_COLORS[target], linewidth=2.2, label=f"target {target}")
        )
    return handles


def style_axis(ax, show_ylabel):
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.set_yticks([tick for tick in YTICKS if tick <= Y_MAX + 1e-9])
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

    for target in TARGETS:
        ax.axvline(
            target,
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            color=TARGET_COLORS[target],
            zorder=1.0,
        )
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

    for target in TARGETS:
        grid_c, dens_c = DENSITY_CACHE[(guidance_factor, target)]
        grid_c, dens_c = extend_curve_to_zero(grid_c, dens_c)
        ax.plot(
            grid_c,
            dens_c,
            linewidth=2.2,
            color=TARGET_COLORS[target],
            zorder=2.5,
        )

    ax.set_title(f"Guidance factor = {guidance_factor}", fontsize=TITLE_FONTSIZE, pad=10)
    style_axis(ax, show_ylabel=show_ylabel)


def save_1x4(guidance_factors=GUIDANCE_FACTORS, output_path=OUT_1X4):
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9), sharey=True)
    for i, (ax, g) in enumerate(zip(axes, guidance_factors)):
        draw_panel(ax, g, show_ylabel=(i == 0))

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.055)
    fig.legend(
        handles=legend_handles(),
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
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.95), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (ax, g) in enumerate(zip(axes, GUIDANCE_FACTORS)):
        draw_panel(ax, g, show_ylabel=(i in (0, 2)))

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.02)
    fig.legend(
        handles=legend_handles(),
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


def save_3x2():
    """Five guidance panels in reading order, with the sixth 3x2 cell blank."""
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.1), sharex=True, sharey=True)
    flat_axes = axes.flatten()
    for i, (ax, g) in enumerate(zip(flat_axes, GUIDANCE_FACTORS_WITH_G0)):
        draw_panel(ax, g, show_ylabel=(i % 3 == 0))
    flat_axes[-1].axis("off")

    fig.supxlabel("AnisoNet-predicted dielectric scalar", fontsize=LABEL_FONTSIZE, y=0.025)
    fig.legend(
        handles=legend_handles(), loc="upper center", ncol=5,
        bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=LEGEND_FONTSIZE,
        handlelength=1.6, handletextpad=0.6, columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.14, top=0.83, hspace=0.38, wspace=0.20)
    fig.savefig(OUT_01235_3X2, dpi=350, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME = load_frame()
    DATASET_VALUES = load_dataset_values()
    GRID_DS, DENS_DS, DENSITY_CACHE, X_MIN, X_MAX, Y_MAX = compute_density_cache(FRAME, DATASET_VALUES)
    save_1x4()
    save_2x2()
    save_1x4([0, 1, 2, 5], OUT_0125_1X4)
    save_3x2()
