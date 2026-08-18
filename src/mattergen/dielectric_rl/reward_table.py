from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mattergen.dielectric_rl.rewards import RewardConfig, build_reward_row


def archive_member_uri(zip_path: Path, member_name: str) -> str:
    return f"zip://{zip_path}!{member_name}"


def _normalize_guidance_token(value: int | float | str) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric).replace(".", "p")


def _build_source_folder(root: Path, guidance_factor: int | float | str, condition: int | str) -> Path:
    guidance_token = _normalize_guidance_token(guidance_factor)
    return root / f"guidance{guidance_token}_cond{condition}_n128"


def build_best_checkpoint_guidance_sweep_reward_table(
    source_root: Path,
    combined_evaluation_csv: Path,
    guidance_factor: int | float,
    conditions: list[int],
    reward_config: RewardConfig,
) -> pd.DataFrame:
    source_root = Path(source_root)
    df = pd.read_csv(combined_evaluation_csv)
    df = df[df["guidance_factor"] == guidance_factor].copy()
    df = df[df["condition"].isin(conditions)].copy()
    df = df.sort_values(["condition", "structure_index"]).reset_index(drop=True)
    df["local_structure_index"] = df.groupby(["guidance_factor", "condition"]).cumcount()

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        condition = int(record["condition"])
        local_structure_index = int(record["local_structure_index"])
        source_folder = _build_source_folder(
            root=source_root,
            guidance_factor=guidance_factor,
            condition=condition,
        )
        zip_path = source_folder / "generated_crystals_cif.zip"
        cif_member = f"gen_{local_structure_index}.cif"
        row = {
            "structure_id": (
                f"gf{_normalize_guidance_token(guidance_factor)}_"
                f"cond{condition}_idx{local_structure_index:03d}"
            ),
            "cif_path": archive_member_uri(zip_path=zip_path, member_name=cif_member),
            "relaxed_cif_path": None,
            "source_folder": str(source_folder),
            "source_kind": "best_checkpoint_guidance_sweep",
            "source_condition": condition,
            "guidance_factor": float(guidance_factor),
            "structure_index": int(record["structure_index"]),
            "local_structure_index": local_structure_index,
            "formula": record.get("formula"),
            "chemical_system": record.get("chemical_system"),
            "predicted_dielectric_scalar": record.get("predicted_dielectric_scalar"),
            "energy_above_hull_per_atom": record.get("energy_above_hull_per_atom"),
            "stable": record.get("stable"),
            "rmsd_from_relaxation": record.get("rmsd_from_relaxation"),
            "is_unique": record.get("is_unique"),
            "is_novel": record.get("is_novel"),
            "is_explored": record.get("is_explored"),
            "composition_valid": record.get("composition_valid"),
            "structure_valid": record.get("structure_valid"),
            "structure_comp_valid": record.get("structure_comp_valid"),
            "abs_error_to_source_condition": record.get("abs_error_to_target"),
        }
        rows.append(build_reward_row(row, reward_config))

    return pd.DataFrame(rows).sort_values(
        ["reward_final", "reward_raw", "abs_error"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_released_final_evaluation_reward_table(
    sample_ledger: Path,
    source_evaluation_csv: Path,
    guidance_factor: int | float,
    conditions: list[int],
    reward_config: RewardConfig,
) -> pd.DataFrame:
    """Build a reward table from release-relative CIF and evaluation records."""
    samples = pd.read_csv(sample_ledger)
    evaluation = pd.read_csv(source_evaluation_csv)
    samples = samples[
        (samples["guidance"] == guidance_factor) & samples["target"].isin(conditions)
    ].copy()
    evaluation = evaluation[
        (evaluation["guidance_factor"] == guidance_factor)
        & evaluation["condition"].isin(conditions)
    ].copy()
    samples["local_structure_index"] = samples["sample_id"].str.removeprefix("gen_").astype(int)
    merged = evaluation.merge(
        samples,
        left_on=["guidance_factor", "condition", "local_structure_index"],
        right_on=["guidance", "target", "local_structure_index"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_sample"),
    )
    if merged["generated_cif"].isna().any():
        missing = merged.loc[merged["generated_cif"].isna(), "local_structure_index"].tolist()
        raise ValueError(f"Missing released CIF records for local indices: {missing}")

    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        row = {
            "structure_id": record["sample_id"],
            "cif_path": record["generated_cif"],
            "relaxed_cif_path": record["relaxed_cif"],
            "source_folder": str(Path(record["generated_cif"]).parent),
            "source_kind": "released_final_evaluation",
            "source_condition": int(record["condition"]),
            "guidance_factor": float(record["guidance_factor"]),
            "structure_index": int(record["structure_index"]),
            "local_structure_index": int(record["local_structure_index"]),
            "formula": record["formula"],
            "chemical_system": record["chemical_system"],
            "predicted_dielectric_scalar": record["predicted_dielectric_scalar"],
            "energy_above_hull_per_atom": record["energy_above_hull_per_atom"],
            "stable": record["stable"],
            "rmsd_from_relaxation": record["rmsd_from_relaxation"],
            "is_unique": record["is_unique"],
            "is_novel": record["is_novel"],
            "is_explored": record["is_explored"],
            "composition_valid": record["composition_valid"],
            "structure_valid": record["structure_valid"],
            "structure_comp_valid": record["structure_comp_valid"],
            "relaxation_success": record["relaxation_success"],
            "prediction_success": record["anisone_prediction_available"],
            "abs_error_to_source_condition": record["abs_error_to_target"],
        }
        rows.append(build_reward_row(row, reward_config))

    return pd.DataFrame(rows).sort_values(
        ["reward_final", "reward_raw", "abs_error"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
