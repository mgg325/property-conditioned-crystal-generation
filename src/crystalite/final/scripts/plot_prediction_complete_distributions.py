from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
FINAL_REPORT_ROOT = ROOT / "outputs" / "final report output"
FIGURE_ROOT = FINAL_REPORT_ROOT / "figures_prediction_complete"
MANIFEST_JSON = FINAL_REPORT_ROOT / "prediction_complete_plot_manifest.json"
ANISONET_SPLIT_ROOT = Path("external/anisonet/splits")
PRED_COL = "predicted_dft_dielectric_scalar"
TARGETS = (2, 4, 8, 12)

TARGET_COLORS = {
    2: "#d1495b",
    4: "#5e548e",
    8: "#2a9d8f",
    12: "#f4a261",
}


@dataclass(frozen=True)
class EvalSource:
    method: str
    label: str
    per_sample_csv: Path


EVAL_SOURCES = (
    EvalSource(
        method="official_baseline",
        label="official_baseline",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation_official_baseline" / "per_sample_evaluation.csv",
    ),
    EvalSource(
        method="additive",
        label="additive",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation_additive" / "per_sample_evaluation.csv",
    ),
    EvalSource(
        method="concat_mlp",
        label="concat_mlp",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation_concat_mlp" / "per_sample_evaluation.csv",
    ),
    EvalSource(
        method="residual_delta",
        label="residual_delta",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation_residual_delta" / "per_sample_evaluation.csv",
    ),
    EvalSource(
        method="additive_plus_film",
        label="additive_plus_film",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation_additive_plus_film" / "per_sample_evaluation.csv",
    ),
    EvalSource(
        method="fixed_film",
        label="fixed_film",
        per_sample_csv=FINAL_REPORT_ROOT / "evaluation" / "per_sample_evaluation.csv",
    ),
)


def load_anisonet_distribution() -> pd.Series:
    frames: list[pd.DataFrame] = []
    for split in ("train.csv", "val.csv", "test.csv"):
        split_path = ANISONET_SPLIT_ROOT / split
        if split_path.exists():
            frames.append(pd.read_csv(split_path, usecols=["dft_dielectric_scalar"]))
    if not frames:
        raise FileNotFoundError(
            f"No AnisoNet split CSVs found under {ANISONET_SPLIT_ROOT}"
        )
    values = pd.concat(frames, ignore_index=True)["dft_dielectric_scalar"]
    values = pd.to_numeric(values, errors="coerce")
    return values[np.isfinite(values) & (values >= 0.0)]


def load_source_frame(source: EvalSource) -> pd.DataFrame:
    if not source.per_sample_csv.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source.per_sample_csv)
    if frame.empty:
        return frame
    if "prediction_success" in frame.columns:
        pred_ok = frame["prediction_success"].fillna(False).astype(bool)
    else:
        pred_ok = pd.Series(False, index=frame.index)
    frame[PRED_COL] = pd.to_numeric(frame[PRED_COL], errors="coerce")
    frame["guidance_factor"] = pd.to_numeric(frame["guidance_factor"], errors="coerce")
    frame["target_numeric"] = pd.to_numeric(frame["target"], errors="coerce")
    frame = frame[pred_ok & frame[PRED_COL].notna() & np.isfinite(frame[PRED_COL])]
    frame = frame[frame[PRED_COL] >= 0.0].copy()
    frame["source_method"] = source.method
    frame["source_label"] = source.label
    return frame


def kde_line(
    ax: plt.Axes,
    values: np.ndarray,
    label: str,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 0.95,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0.0]
    if values.size < 2:
        return
    sns.kdeplot(
        x=values,
        ax=ax,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        fill=False,
        warn_singular=False,
        clip=(0.0, None),
    )


def _guidance_values_for_method(frame: pd.DataFrame) -> list[int]:
    values = sorted(
        {
            int(value)
            for value in pd.to_numeric(frame["guidance_factor"], errors="coerce").dropna().tolist()
            if float(value) > 0.0
        }
    )
    return values


def plot_method_guidance_figure(
    method: str,
    frame: pd.DataFrame,
    dataset_values: pd.Series,
    baseline_values: np.ndarray,
    guidance_factor: int,
) -> Path | None:
    method_dir = FIGURE_ROOT / method
    method_dir.mkdir(parents=True, exist_ok=True)

    uncond_rows = frame[frame["group"].eq(f"{method}__uncond")]
    cond_rows = frame[
        np.isclose(frame["guidance_factor"], float(guidance_factor), equal_nan=False)
        & frame["target_numeric"].isin(TARGETS)
    ]
    if cond_rows.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    kde_line(
        ax,
        dataset_values.to_numpy(),
        "AnisoNet dataset",
        color="#111111",
        linewidth=2.8,
    )
    kde_line(
        ax,
        baseline_values,
        "Official baseline",
        color="#777777",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
    )
    kde_line(
        ax,
        uncond_rows[PRED_COL].to_numpy(),
        f"{method} uncond",
        color="#444444",
        linestyle=":",
        linewidth=2.0,
        alpha=0.9,
    )

    plotted_targets: list[int] = []
    for target in TARGETS:
        rows = cond_rows[np.isclose(cond_rows["target_numeric"], float(target), equal_nan=False)]
        values = rows[PRED_COL].to_numpy()
        if np.isfinite(values).sum() < 2:
            continue
        kde_line(
            ax,
            values,
            f"target {target}",
            color=TARGET_COLORS[target],
            linewidth=2.2,
        )
        plotted_targets.append(target)

    if not plotted_targets:
        plt.close(fig)
        return None

    for target in plotted_targets:
        ax.axvline(
            target,
            color=TARGET_COLORS[target],
            linestyle="--",
            linewidth=1.2,
            alpha=0.35,
        )

    ax.set_title(f"{method}: predicted dielectric distributions at g={guidance_factor}")
    ax.set_xlabel("Predicted dielectric scalar")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 25)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    out_path = method_dir / f"{method}_prediction_density_g{guidance_factor}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_prediction_complete_distribution_report() -> dict[str, Any]:
    sns.set_theme(style="whitegrid")
    dataset_values = load_anisonet_distribution()

    baseline_frame = load_source_frame(EVAL_SOURCES[0])
    baseline_values = baseline_frame[PRED_COL].to_numpy()

    manifest_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    missing_sources: list[str] = []

    for source in EVAL_SOURCES[1:]:
        frame = load_source_frame(source)
        if frame.empty:
            missing_sources.append(source.method)
            continue

        guidance_values = _guidance_values_for_method(frame)
        manifest_entry: dict[str, Any] = {
            "method": source.method,
            "label": source.label,
            "per_sample_csv": str(source.per_sample_csv),
            "n_rows": int(len(frame)),
            "n_groups": int(frame["group"].nunique()),
            "guidance_factors_plotted": guidance_values,
            "figure_paths": [],
        }

        combined_csv = FINAL_REPORT_ROOT / f"prediction_complete_per_sample_{source.method}.csv"
        frame.to_csv(combined_csv, index=False)
        manifest_entry["combined_per_sample_csv"] = str(combined_csv)

        for guidance_factor in guidance_values:
            figure_path = plot_method_guidance_figure(
                source.method,
                frame,
                dataset_values,
                baseline_values,
                guidance_factor,
            )
            if figure_path is not None:
                figure_paths.append(str(figure_path))
                manifest_entry["figure_paths"].append(str(figure_path))

        manifest_rows.append(manifest_entry)

    payload = {
        "dataset_split_root": str(ANISONET_SPLIT_ROOT),
        "sources_used": manifest_rows,
        "missing_sources": missing_sources,
        "targets_plotted": list(TARGETS),
        "target_colors": {str(k): v for k, v in TARGET_COLORS.items()},
        "figure_paths": figure_paths,
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    result = build_prediction_complete_distribution_report()
    print(json.dumps(result, indent=2))
