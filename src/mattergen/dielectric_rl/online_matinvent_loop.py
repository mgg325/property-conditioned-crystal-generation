from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

from mattergen.dielectric_rl.dataset import build_crystal_dataset_from_records
from mattergen.dielectric_rl.replay_buffer import (
    DiversityFilterConfig,
    LongTermMemory,
    ReplayBuffer,
    ReplayBufferConfig,
)
from mattergen.dielectric_rl.rewards import RewardConfig, build_reward_row, passes_prefilter
from mattergen.dielectric_rl.training import offline_reward_weighted_finetune
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity
from mattergen.evaluation.reference.presets import ReferenceMP2020Correction, ReferenceTRI2024Correction
from mattergen.evaluation.utils.relaxation import relax_structures


MATTERGEN_ROOT = Path("external/mattergen")
MATTERGEN_PYTHON = MATTERGEN_ROOT / ".venv/bin/python"
ANISONET_PYTHON = Path("external/anisonet/.venv/bin/python")
DEFAULT_ANISONET_SCRIPT = (
    MATTERGEN_ROOT
    / "outputs/dielectric_rl/direct_rl_target12_online/anisonet_predict_relaxed_generic.py"
)


@dataclass(frozen=True)
class OnlineLoopConfig:
    run_root: Path
    start_policy_dir: Path
    anisonet_script: Path
    target: float = 2.0
    sigma: float = 2.0
    iterations: int = 1
    sample_batch_size: int = 32
    sample_num_batches: int = 2
    guidance_factor: float = 1.0
    eval_size: int = 16
    topk_ratio: float = 0.5
    replay_buffer_size: int = 100
    replay_sample_size: int = 10
    replay_reward_cutoff: float = 0.1
    replay_recent_window: int = 0
    replay_recent_fraction: float = 0.0
    diversity_enabled: bool = True
    diversity_tol: int = 3
    diversity_buff: int = 6
    training_epochs: int = 3
    training_timesteps: int = 1000
    training_accum_steps: int = 50
    learning_rate: float = 1.0e-5
    kl_weight: float = 1000.0
    anchor_source: str = "generated_pool"
    anchor_max_records: int = 0
    checkpoint_every_epoch: bool = True
    num_workers: int = 0
    device: str = "cuda"


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=cwd, env=env, check=True)


def _reference_file_is_usable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(64)
    except Exception:
        return False
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False
    return True


def _load_reference_and_scheme_if_available() -> tuple[ReferenceMP2020Correction | ReferenceTRI2024Correction | None, Any | None, str]:
    mp2020_path = MATTERGEN_ROOT / "data-release/alex-mp/reference_MP2020correction.gz"
    tri2024_path = MATTERGEN_ROOT / "data-release/alex-mp/reference_TRI2024correction.gz"

    if _reference_file_is_usable(mp2020_path):
        print("Using MP2020 reference dataset for novel/unique/stable filtering.", flush=True)
        return ReferenceMP2020Correction(), MaterialsProject2020Compatibility(), "MP2020"

    if _reference_file_is_usable(tri2024_path):
        print(
            "WARNING: reference_MP2020correction.gz is unavailable or only a Git LFS pointer. "
            "Falling back to local TRI2024 reference dataset for novel/unique/stable filtering.",
            flush=True,
        )
        return ReferenceTRI2024Correction(), TRI110Compatibility2024(), "TRI2024"

    print(
        "WARNING: neither MP2020 nor TRI2024 reference dataset is locally usable. "
        "Falling back to validity-only prefilter for this run.",
        flush=True,
    )
    return None, None, "NONE"


def _policy_paths(model_dir: Path) -> tuple[Path, Path]:
    return model_dir / "checkpoints/last.ckpt", model_dir / "config.yaml"


def _export_policy_dir(source_policy_dir: Path, source_checkpoint_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = destination / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_policy_dir / "config.yaml", destination / "config.yaml")
    shutil.copy2(source_checkpoint_path, checkpoints_dir / "last.ckpt")
    return destination


def _generate_structures(policy_dir: Path, iteration_dir: Path, cfg: OnlineLoopConfig) -> Path:
    generated_dir = iteration_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    if (generated_dir / "generated_crystals.extxyz").exists():
        return generated_dir
    env = dict(os.environ)
    env["PYTHONPATH"] = str(MATTERGEN_ROOT)
    cmd = [
        MATTERGEN_PYTHON,
        MATTERGEN_ROOT / "mattergen/scripts/generate.py",
        generated_dir,
        f"--model_path={policy_dir}",
        f"--batch_size={cfg.sample_batch_size}",
        f"--num_batches={cfg.sample_num_batches}",
        f"--properties_to_condition_on={{dft_dielectric_scalar:{cfg.target}}}",
        f"--diffusion_guidance_factor={cfg.guidance_factor}",
        "--record_trajectories=False",
    ]
    _run(cmd, cwd=MATTERGEN_ROOT, env=env)
    return generated_dir


def _extract_input_cifs(generated_dir: Path, iteration_dir: Path) -> pd.DataFrame:
    input_dir = iteration_dir / "postprocess/input_cifs"
    input_dir.mkdir(parents=True, exist_ok=True)
    marker = input_dir / ".extracted"
    zip_path = generated_dir / "generated_crystals_cif.zip"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(input_dir)
        marker.write_text("done\n")

    rows: list[dict[str, Any]] = []
    for cif_path in sorted(input_dir.glob("gen_*.cif")):
        record: dict[str, Any] = {
            "structure_id": cif_path.stem,
            "cif_path": str(cif_path),
            "input_cif_path": str(cif_path),
            "raw_valid": False,
            "formula": None,
            "chemical_system": None,
            "num_sites": None,
            "raw_valid_reason": "",
        }
        try:
            structure = Structure.from_file(cif_path)
            valid = (
                structure_validity(structure)
                and is_smact_valid(structure)
                and max(structure.lattice.abc) < 25.0
            )
            record["raw_valid"] = bool(valid)
            record["formula"] = structure.composition.formula
            record["chemical_system"] = structure.composition.chemical_system
            record["num_sites"] = len(structure)
            if not valid:
                record["raw_valid_reason"] = "invalid_raw_structure"
        except Exception as exc:
            record["raw_valid_reason"] = f"parse_failed:{type(exc).__name__}"
        rows.append(record)
    df = pd.DataFrame(rows)
    metadata_dir = iteration_dir / "postprocess/metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_dir / "basic_validity.csv", index=False)
    return df


def _prefilter_ehull(
    records: list[dict[str, Any]],
    reference: ReferenceMP2020Correction | ReferenceTRI2024Correction | None,
) -> list[dict[str, Any]]:
    if reference is None:
        return [record for record in records if record.get("raw_valid", False)]
    ref_set = set(reference.entries_by_chemsys.keys())
    filtered: list[dict[str, Any]] = []
    for record in records:
        if not record.get("raw_valid", False):
            continue
        try:
            structure = Structure.from_file(record["input_cif_path"])
        except Exception:
            continue
        element_set = {str(el) for el in structure.composition.elements}
        if any(element not in ref_set for element in element_set):
            continue
        missing_energy = False
        for chemsys in element_set:
            entries = reference.entries_by_chemsys.get(chemsys, [])
            if entries and all(np.isnan(entry.energy) for entry in entries):
                missing_energy = True
                break
        if not missing_energy:
            filtered.append(record)
    return filtered


def _run_relaxation(filtered_records: list[dict[str, Any]], iteration_dir: Path, cfg: OnlineLoopConfig) -> pd.DataFrame:
    relaxed_dir = iteration_dir / "postprocess/relaxed"
    relaxed_cif_dir = relaxed_dir / "relaxed_cifs"
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    relaxed_cif_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = relaxed_dir / "relax_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    structures = [Structure.from_file(record["input_cif_path"]) for record in filtered_records]
    relaxed_structures, energies = relax_structures(
        structures,
        device=cfg.device,
        output_path=str(relaxed_dir / "relaxed_structures.extxyz"),
    )
    np.save(relaxed_dir / "relaxed_energies.npy", energies)

    rows: list[dict[str, Any]] = []
    for record, relaxed_structure, energy in zip(filtered_records, relaxed_structures, energies):
        relaxed_cif_path = relaxed_cif_dir / f"{record['structure_id']}_relaxed.cif"
        relaxed_structure.to(filename=str(relaxed_cif_path), fmt="cif")
        rows.append(
            {
                **record,
                "relaxed_cif_path": str(relaxed_cif_path),
                "relaxation_success": True,
                "total_energy": float(energy),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def _compute_sun_prefilter(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    structures = [Structure.from_file(path) for path in manifest["relaxed_cif_path"]]
    validity_mask = [
        bool(structure_validity(structure)) and bool(is_smact_valid(structure))
        for structure in structures
    ]
    enriched = manifest.copy()
    enriched["composition_valid"] = validity_mask
    enriched["structure_valid"] = validity_mask
    enriched["structure_comp_valid"] = validity_mask
    reference, energy_correction_scheme, reference_name = _load_reference_and_scheme_if_available()
    enriched["reference_name"] = reference_name
    if reference is None:
        enriched["is_unique"] = None
        enriched["is_novel"] = None
        enriched["stable"] = None
        return enriched

    original_structures = [Structure.from_file(path) for path in manifest["input_cif_path"]]
    energies = manifest["total_energy"].astype(float).tolist()
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=structures,
        energies=energies,
        original_structures=original_structures,
        reference=reference,
        energy_correction_scheme=energy_correction_scheme,
    )
    enriched["is_unique"] = evaluator.is_unique.astype(bool)
    enriched["is_novel"] = evaluator.is_novel.astype(bool)
    enriched["stable"] = evaluator.is_stable.astype(bool)
    return enriched


def _select_eval_candidates(
    manifest_with_metrics: pd.DataFrame,
    iteration_dir: Path,
    reward_cfg: RewardConfig,
    eval_size: int,
) -> pd.DataFrame:
    ordered_records = manifest_with_metrics.sort_values("structure_id").to_dict(orient="records")
    filtered_records = [record for record in ordered_records if passes_prefilter(record, reward_cfg)]
    selected = pd.DataFrame(filtered_records[: int(eval_size)])
    selected_dir = iteration_dir / "postprocess/selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(selected_dir / "prefiltered_eval_candidates.csv", index=False)
    return selected


def _run_prediction(selected_manifest: pd.DataFrame, iteration_dir: Path, cfg: OnlineLoopConfig) -> pd.DataFrame:
    prediction_dir = iteration_dir / "postprocess/predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    staged_dir = prediction_dir / "input_relaxed_cifs"
    staged_dir.mkdir(parents=True, exist_ok=True)
    for relaxed_cif_path in selected_manifest["relaxed_cif_path"]:
        src = Path(relaxed_cif_path)
        dst = staged_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
    output_csv = prediction_dir / "anisonet_relaxed_predictions.csv"
    summary_json = prediction_dir / "anisonet_relaxed_predictions_summary.json"
    if not output_csv.exists():
        cmd = [
            ANISONET_PYTHON,
            cfg.anisonet_script,
            "--input_dir",
            staged_dir,
            "--output_csv",
            output_csv,
            "--summary_json",
            summary_json,
        ]
        _run(cmd, cwd=cfg.run_root)
    return pd.read_csv(output_csv)


def _build_reward_table(
    selected_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    iteration_dir: Path,
    reward_cfg: RewardConfig,
) -> pd.DataFrame:
    merged = selected_manifest.merge(
        predictions[
            [
                "structure_id",
                "predicted_dielectric_scalar",
                "predicted_tensor",
                "volume",
                "prediction_success",
            ]
        ],
        on="structure_id",
        how="left",
    )
    merged["prediction_success"] = merged["prediction_success"].fillna(False)
    rows = [build_reward_row(record, reward_cfg) for record in merged.to_dict(orient="records")]
    reward_table = pd.DataFrame(rows).reset_index(drop=True)
    rewards_dir = iteration_dir / "postprocess/rewards"
    rewards_dir.mkdir(parents=True, exist_ok=True)
    reward_table.to_csv(rewards_dir / "reward_table.csv", index=False)
    return reward_table


def _save_reward_summary(reward_table: pd.DataFrame, rewards_dir: Path) -> dict[str, Any]:
    rewards = reward_table["reward_final"].fillna(0.0)
    dielectric = reward_table["predicted_dielectric_scalar"].dropna()
    summary = {
        "n_samples": int(len(reward_table)),
        "mean_predicted_dielectric_scalar": float(dielectric.mean()) if not dielectric.empty else None,
        "mean_abs_error": float(reward_table["abs_error"].dropna().mean()) if "abs_error" in reward_table else None,
        "mean_reward_final": float(rewards.mean()),
        "within_1.0": int((reward_table["abs_error"] <= 1.0).sum()),
        "within_2.0": int((reward_table["abs_error"] <= 2.0).sum()),
        "reward_gt_0p1": int((rewards > 0.1).sum()),
        "reward_gt_0p5": int((rewards > 0.5).sum()),
    }
    (rewards_dir / "reward_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _next_iteration_index(run_root: Path) -> int:
    existing = []
    for path in run_root.glob("loop_*"):
        if not path.is_dir():
            continue
        try:
            existing.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    if not existing:
        return 0
    return max(existing) + 1


def _load_existing_replay(run_root: Path, replay: ReplayBuffer) -> None:
    replay_path = run_root / "replay_buffer.csv"
    if not replay_path.exists():
        return
    replay_df = pd.read_csv(replay_path)
    if replay_df.empty:
        return
    if "record" in replay_df.columns:
        def _parse_record(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            stripped = value.strip()
            if not stripped.startswith("{"):
                return value
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                # Support replay rows containing Python literals with NaN tokens.
                return eval(
                    stripped,
                    {"__builtins__": {}},
                    {
                        "nan": float("nan"),
                        "True": True,
                        "False": False,
                        "None": None,
                    },
                )

        replay_df["record"] = replay_df["record"].apply(_parse_record)
    replay.buffer = replay_df


def _load_existing_ltm(run_root: Path, ltm: LongTermMemory) -> None:
    ltm_path = run_root / "long_term_memory.csv"
    if not ltm_path.exists():
        return
    ltm_df = pd.read_csv(ltm_path)
    if ltm_df.empty:
        return
    ltm.memory = ltm_df


def _build_training_cfg(cfg: OnlineLoopConfig, policy_dir: Path) -> Any:
    checkpoint_path, config_path = _policy_paths(policy_dir)
    return OmegaConf.create(
        {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(config_path),
            "freeze_strategy": "adapter_only",
            "trainable_parameter_keywords": [
                "property_embeddings",
                "adapt",
                "mixin",
                "adapter",
                "ctrl",
                "condition",
            ],
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.training_epochs,
            "timesteps": cfg.training_timesteps,
            "accum_steps": cfg.training_accum_steps,
            "kl_weight": cfg.kl_weight,
            "anchor_source": cfg.anchor_source,
            "anchor_max_records": cfg.anchor_max_records,
            "checkpoint_every_epoch": cfg.checkpoint_every_epoch,
            "num_workers": cfg.num_workers,
        }
    )


def _select_anchor_records(
    manifest_with_metrics: pd.DataFrame,
    replay_records: list[dict[str, Any]],
    loop_cfg: OnlineLoopConfig,
) -> list[dict[str, Any]]:
    anchor_source = str(getattr(loop_cfg, "anchor_source", "generated_pool"))
    if anchor_source == "generated_pool":
        anchor_records = manifest_with_metrics.sort_values("structure_id").to_dict(orient="records")
    elif anchor_source == "replay":
        anchor_records = list(replay_records)
    else:
        raise ValueError(f"Unsupported anchor_source: {anchor_source}")

    anchor_max_records = int(getattr(loop_cfg, "anchor_max_records", 0))
    if anchor_max_records > 0:
        anchor_records = anchor_records[:anchor_max_records]
    return anchor_records


def run_iteration(
    loop_cfg: OnlineLoopConfig,
    current_policy_dir: Path,
    replay: ReplayBuffer,
    ltm: LongTermMemory,
    reward_cfg: RewardConfig,
    diversity_cfg: DiversityFilterConfig,
    iteration_idx: int,
) -> Path:
    iteration_dir = loop_cfg.run_root / f"loop_{iteration_idx:04d}"
    iteration_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = _generate_structures(current_policy_dir, iteration_dir, loop_cfg)
    raw_validity = _extract_input_cifs(generated_dir, iteration_dir)
    reference, _, _ = _load_reference_and_scheme_if_available()
    filtered_records = _prefilter_ehull(raw_validity.to_dict(orient="records"), reference)
    if not filtered_records:
        raise RuntimeError(f"No valid records survived pre_filter_ehull in loop {iteration_idx}.")

    manifest = _run_relaxation(filtered_records, iteration_dir, loop_cfg)
    manifest_with_metrics = _compute_sun_prefilter(manifest)
    selected_manifest = _select_eval_candidates(
        manifest_with_metrics=manifest_with_metrics,
        iteration_dir=iteration_dir,
        reward_cfg=reward_cfg,
        eval_size=loop_cfg.eval_size,
    )
    if selected_manifest.empty:
        raise RuntimeError(f"No samples survived SUN-style prefilter in loop {iteration_idx}.")

    predictions = _run_prediction(selected_manifest, iteration_dir, loop_cfg)
    reward_table = _build_reward_table(
        selected_manifest=selected_manifest,
        predictions=predictions,
        iteration_dir=iteration_dir,
        reward_cfg=reward_cfg,
    )
    rewards_dir = iteration_dir / "postprocess/rewards"
    summary = _save_reward_summary(reward_table, rewards_dir)

    raw_rewards = reward_table["reward_final"].to_numpy(dtype=np.float32)
    records = reward_table.to_dict(orient="records")
    ltm.extend(records, raw_rewards, step=iteration_idx)
    adjusted_rewards, penalty_idx, tol_n, buff_n = ltm.div_filter(records, raw_rewards, diversity_cfg)
    reward_table = reward_table.copy()
    reward_table["reward_final_diversity"] = adjusted_rewards
    reward_table["reward_final"] = adjusted_rewards
    reward_table["diversity_penalty"] = False
    if penalty_idx:
        reward_table.loc[penalty_idx, "diversity_penalty"] = True
    reward_table.to_csv(rewards_dir / "reward_table.csv", index=False)

    topk_count = max(1, int(loop_cfg.eval_size * loop_cfg.topk_ratio))
    reward_table_sorted = reward_table.sort_values("reward_final", ascending=False).reset_index(drop=True)
    topk_table = reward_table_sorted.head(topk_count).copy()
    topk_table["selection_rank"] = np.arange(1, len(topk_table) + 1)
    topk_table.to_csv(rewards_dir / "topk_records.csv", index=False)

    topk_records = topk_table.to_dict(orient="records")
    for record in topk_records:
        record["source_loop"] = int(iteration_idx)
    replay_records, _ = replay.sample_mixed(current_loop=int(iteration_idx))
    finetune_records = topk_records + replay_records
    replay.extend(topk_records)

    replay_df = pd.DataFrame(
        {
            "structure_id": [record.get("structure_id") for record in finetune_records],
            "reward_final": [float(record.get("reward_final", 0.0)) for record in finetune_records],
            "source": ["topk"] * len(topk_records) + ["replay"] * len(replay_records),
        }
    )
    replay_df.to_csv(iteration_dir / "finetune_records.csv", index=False)

    dataset_result = build_crystal_dataset_from_records(
        records=finetune_records,
        target_dielectric_scalar=loop_cfg.target,
        use_relaxed_if_available=True,
    )
    anchor_records = _select_anchor_records(
        manifest_with_metrics=manifest_with_metrics,
        replay_records=replay_records,
        loop_cfg=loop_cfg,
    )
    anchor_dataset_result = build_crystal_dataset_from_records(
        records=anchor_records,
        target_dielectric_scalar=loop_cfg.target,
        use_relaxed_if_available=True,
    )
    training_cfg = _build_training_cfg(loop_cfg, current_policy_dir)
    rl_update_dir = iteration_dir / "rl_update"
    checkpoints_dir = rl_update_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    training_result = offline_reward_weighted_finetune(
        dataset=dataset_result.dataset,
        sample_weights=np.asarray(
            [float(record.get("reward_final", 0.0)) for record in finetune_records], dtype=np.float32
        ),
        training_cfg=training_cfg,
        output_checkpoint_dir=checkpoints_dir,
        predicted_dielectric_scalar=np.asarray(
            [float(record.get("predicted_dielectric_scalar", np.nan)) for record in finetune_records],
            dtype=np.float32,
        ),
        target_dielectric_scalar=loop_cfg.target,
        anchor_dataset=anchor_dataset_result.dataset,
    )

    policy_for_generation = iteration_dir / "policy_for_generation"
    new_policy_dir = _export_policy_dir(
        source_policy_dir=current_policy_dir,
        source_checkpoint_path=checkpoints_dir / "final.ckpt",
        destination=policy_for_generation,
    )

    loop_summary = {
        "iteration": iteration_idx,
        "current_policy_dir": str(current_policy_dir),
        "generated_dir": str(generated_dir),
        "n_generated": int(loop_cfg.sample_batch_size * loop_cfg.sample_num_batches),
        "n_prefiltered_eval": int(len(selected_manifest)),
        "n_topk": int(len(topk_table)),
        "n_replay_sampled": int(len(replay_records)),
        "n_finetune_records": int(len(finetune_records)),
        "n_anchor_records": int(len(anchor_records)),
        "diversity_tol_hits": int(tol_n),
        "diversity_buff_hits": int(buff_n),
        "reward_summary": summary,
        "training_result": training_result,
        "next_policy_dir": str(new_policy_dir),
    }
    (iteration_dir / "loop_summary.json").write_text(json.dumps(loop_summary, indent=2))

    ltm_df = ltm.memory.copy()
    ltm_df.to_csv(loop_cfg.run_root / "long_term_memory.csv", index=False)
    if len(replay.buffer) > 0:
        replay.buffer.to_csv(loop_cfg.run_root / "replay_buffer.csv", index=False)
    return new_policy_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MatInvent-style online RL loop for MatterGen.")
    parser.add_argument(
        "--run_root",
        default="outputs/dielectric_rl/direct_rl_target12_online/matinvent_style_target12",
    )
    parser.add_argument(
        "--start_policy_dir",
        default="outputs/dielectric_rl/direct_rl_target2_online/bootstrap/initial_policy_fixed_scaler",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--target", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--sample_batch_size", type=int, default=32)
    parser.add_argument("--sample_num_batches", type=int, default=2)
    parser.add_argument("--eval_size", type=int, default=16)
    parser.add_argument("--topk_ratio", type=float, default=0.5)
    parser.add_argument("--replay_recent_window", type=int, default=0)
    parser.add_argument("--replay_recent_fraction", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--training_epochs", type=int, default=3)
    parser.add_argument("--training_timesteps", type=int, default=1000)
    parser.add_argument("--training_accum_steps", type=int, default=50)
    parser.add_argument("--kl_weight", type=float, default=1000.0)
    parser.add_argument("--anchor_source", default="generated_pool")
    parser.add_argument("--anchor_max_records", type=int, default=0)
    parser.add_argument("--guidance_factor", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anisonet_script", default=str(DEFAULT_ANISONET_SCRIPT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OnlineLoopConfig(
        run_root=Path(args.run_root),
        start_policy_dir=Path(args.start_policy_dir),
        anisonet_script=Path(args.anisonet_script),
        target=float(args.target),
        sigma=float(args.sigma),
        iterations=int(args.iterations),
        sample_batch_size=int(args.sample_batch_size),
        sample_num_batches=int(args.sample_num_batches),
        eval_size=int(args.eval_size),
        topk_ratio=float(args.topk_ratio),
        replay_recent_window=int(args.replay_recent_window),
        replay_recent_fraction=float(args.replay_recent_fraction),
        learning_rate=float(args.learning_rate),
        training_epochs=int(args.training_epochs),
        training_timesteps=int(args.training_timesteps),
        training_accum_steps=int(args.training_accum_steps),
        kl_weight=float(args.kl_weight),
        anchor_source=str(args.anchor_source),
        anchor_max_records=int(args.anchor_max_records),
        guidance_factor=float(args.guidance_factor),
        device=str(args.device),
    )
    cfg.run_root.mkdir(parents=True, exist_ok=True)
    reward_cfg = RewardConfig(
        target_dielectric_scalar=cfg.target,
        sigma=cfg.sigma,
        gate_on_missing_prediction=True,
        gate_on_composition_valid=True,
        gate_on_structure_valid=True,
        gate_on_structure_comp_valid=True,
        gate_on_stable=False,
        prefilter_on_composition_valid=True,
        prefilter_on_structure_valid=True,
        prefilter_on_structure_comp_valid=True,
        prefilter_on_stable=True,
        prefilter_on_unique=True,
        prefilter_on_novel=True,
    )
    replay = ReplayBuffer(
        ReplayBufferConfig(
            enabled=True,
            buffer_size=cfg.replay_buffer_size,
            sample_size=cfg.replay_sample_size,
            reward_cutoff=cfg.replay_reward_cutoff,
            dedup_method="composition",
            topk_ratio=cfg.topk_ratio,
            eval_size=cfg.eval_size,
            recent_window=cfg.replay_recent_window,
            recent_fraction=cfg.replay_recent_fraction,
        )
    )
    ltm = LongTermMemory()
    _load_existing_replay(cfg.run_root, replay)
    _load_existing_ltm(cfg.run_root, ltm)
    current_policy_dir = cfg.start_policy_dir
    start_iteration_idx = _next_iteration_index(cfg.run_root)
    run_metadata = {
        "start_policy_dir": str(current_policy_dir),
        "start_iteration_idx": start_iteration_idx,
        "target": cfg.target,
        "sigma": cfg.sigma,
        "iterations": cfg.iterations,
        "sample_batch_size": cfg.sample_batch_size,
        "sample_num_batches": cfg.sample_num_batches,
        "eval_size": cfg.eval_size,
        "topk_ratio": cfg.topk_ratio,
        "replay_recent_window": cfg.replay_recent_window,
        "replay_recent_fraction": cfg.replay_recent_fraction,
        "learning_rate": cfg.learning_rate,
        "training_epochs": cfg.training_epochs,
        "training_timesteps": cfg.training_timesteps,
        "training_accum_steps": cfg.training_accum_steps,
        "kl_weight": cfg.kl_weight,
        "anchor_source": cfg.anchor_source,
        "anchor_max_records": cfg.anchor_max_records,
    }
    (cfg.run_root / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    diversity_cfg = DiversityFilterConfig(
        enabled=cfg.diversity_enabled,
        tol=cfg.diversity_tol,
        buff=cfg.diversity_buff,
        method="composition",
    )
    for local_iteration_idx in range(cfg.iterations):
        iteration_idx = start_iteration_idx + local_iteration_idx
        current_policy_dir = run_iteration(
            loop_cfg=cfg,
            current_policy_dir=current_policy_dir,
            replay=replay,
            ltm=ltm,
            reward_cfg=reward_cfg,
            diversity_cfg=diversity_cfg,
            iteration_idx=iteration_idx,
        )
        print(json.dumps({"completed_iteration": iteration_idx, "next_policy_dir": str(current_policy_dir)}, indent=2))


if __name__ == "__main__":
    main()
