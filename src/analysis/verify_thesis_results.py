#!/usr/bin/env python3
"""Verify summary metrics from released final-evaluation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def add_row(
    rows: list[dict],
    output: str,
    condition: str,
    metric: str,
    reported: object,
    recomputed: object,
    decimal_places: int | None = None,
) -> None:
    if pd.isna(reported) or pd.isna(recomputed):
        match = "NOT_CHECKED"
    elif decimal_places is not None:
        match = round(float(reported), decimal_places) == round(float(recomputed), decimal_places)
    else:
        match = bool(abs(float(reported) - float(recomputed)) < 1e-6)
    rows.append(
        {
            "thesis_output": output,
            "condition": condition,
            "metric": metric,
            "reported_value": reported,
            "recomputed_value": recomputed,
            "match": match,
            "notes": "",
        }
    )


def verify_mattergen(root: Path, rows: list[dict]) -> None:
    samples = pd.read_csv(root / "data/final_evaluations/mattergen/mattergen_final_samples.csv")
    summary = pd.read_csv(root / "data/final_evaluations/mattergen/source/adapter_group_summary_indexfixed.csv")
    for reference in summary.itertuples(index=False):
        subset = samples[(samples.guidance == reference.guidance_factor) & (samples.target == reference.condition)]
        condition = f"g{reference.guidance_factor}_target{reference.condition}"
        add_row(rows, "MatterGen guidance sweep", condition, "mean_prediction", reference.mean_predicted_dielectric_scalar, subset.dielectric_prediction.mean())
        add_row(rows, "MatterGen guidance sweep", condition, "median_prediction", reference.median_predicted_dielectric_scalar, subset.dielectric_prediction.median())
        add_row(rows, "MatterGen guidance sweep", condition, "MAE", reference.mae_to_target, (subset.dielectric_prediction - subset.target).abs().mean())
        add_row(rows, "MatterGen guidance sweep", condition, "stable_fraction", reference.frac_stable, subset.stable.mean())
        add_row(rows, "MatterGen guidance sweep", condition, "SUN_fraction", reference.frac_novel_unique_stable, subset.sun.mean())


def verify_crystalite(root: Path, rows: list[dict]) -> None:
    samples = pd.read_csv(root / "data/final_evaluations/crystalite/crystalite_final_samples.csv")
    summary = pd.read_csv(root / "data/final_evaluations/crystalite/source/consolidated_group_summary.csv")
    unconditional = pd.read_csv(
        root / "data/final_evaluations/crystalite/source/final_film_thermo_per_sample.csv"
    )
    unconditional_predictions = unconditional.loc[
        (unconditional.guidance_factor == 0) & unconditional.prediction_success,
        "predicted_dft_dielectric_scalar",
    ]
    for target, reported_mae in {2: 4.82, 4: 3.50, 8: 4.24, 12: 6.54}.items():
        add_row(
            rows,
            "Crystalite FiLM guidance sweep",
            f"g0_target{target}",
            "MAE",
            reported_mae,
            (unconditional_predictions - target).abs().mean(),
            decimal_places=2,
        )
    for reference in summary.itertuples(index=False):
        if reference.guidance_factor == 0:
            continue
        group = str(reference.group)
        subset = samples[(samples.generated_cif.str.contains(group, na=False)) & samples.anisone_prediction_available]
        if subset.empty:
            continue
        add_row(rows, "Crystalite FiLM guidance sweep", group, "mean_prediction", reference.mean_predicted_dielectric_scalar, subset.dielectric_prediction.mean())
        add_row(rows, "Crystalite FiLM guidance sweep", group, "median_prediction", reference.median_predicted_dielectric_scalar, subset.dielectric_prediction.median())
        add_row(rows, "Crystalite FiLM guidance sweep", group, "MAE", reference.mae, (subset.dielectric_prediction - pd.to_numeric(subset.target, errors="coerce")).abs().mean())


def verify_reward(root: Path, rows: list[dict]) -> None:
    samples = pd.read_csv(root / "data/final_evaluations/reward_guided/reward_guided_target12_samples.csv")
    for summary_path in (root / "data/final_evaluations/reward_guided").glob("*/*/loop_summary.json"):
        payload = json.loads(summary_path.read_text())
        branch = summary_path.parents[1].name
        iteration = summary_path.parent.name
        subset = samples[(samples.branch == branch) & (samples.iteration.astype(str) == iteration) & samples.anisone_prediction_available]
        if subset.empty:
            continue
        reported = payload.get("reward_summary", {})
        condition = f"{branch}_{iteration}"
        add_row(rows, "Reward-guided MatterGen", condition, "mean_prediction", reported.get("mean_predicted_dielectric_scalar"), subset.dielectric_prediction.mean())
        add_row(rows, "Reward-guided MatterGen", condition, "mean_reward", reported.get("mean_reward_final"), subset.reward.mean())
        add_row(rows, "Reward-guided MatterGen", condition, "within_pm2", reported.get("within_2.0"), ((subset.dielectric_prediction - 12).abs() <= 2).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    verify_mattergen(args.release_root, rows)
    verify_crystalite(args.release_root, rows)
    verify_reward(args.release_root, rows)
    output = args.release_root / "data/verification/thesis_result_verification.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
