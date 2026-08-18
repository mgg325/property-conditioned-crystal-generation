from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.dielectric_rl.dataset import build_crystal_dataset_from_records
from mattergen.dielectric_rl.replay_buffer import ReplayBufferConfig, initialize_replay_buffer
from mattergen.dielectric_rl.rewards import RewardConfig, passes_prefilter
from mattergen.dielectric_rl.reward_table import (
    build_best_checkpoint_guidance_sweep_reward_table,
    build_released_final_evaluation_reward_table,
)
from mattergen.dielectric_rl.training import offline_reward_weighted_finetune


def _resolve_project_relative_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return MODELS_PROJECT_ROOT.parent / path


def _make_output_layout(run_name: str, output_root: str | Path) -> dict[str, Path]:
    run_dir = _resolve_project_relative_path(output_root) / run_name
    layout = {
        "run_dir": run_dir,
        "generated_cifs": run_dir / "generated_cifs",
        "relaxed": run_dir / "relaxed",
        "rewards": run_dir / "rewards",
        "selected": run_dir / "selected",
        "checkpoints": run_dir / "checkpoints",
        "metadata": run_dir / "metadata",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _build_reward_config(cfg: DictConfig) -> RewardConfig:
    return RewardConfig(
        target_dielectric_scalar=float(cfg.reward.target_dielectric_scalar),
        sigma=float(cfg.reward.sigma),
        gate_on_missing_prediction=bool(cfg.reward.gate_on_missing_prediction),
        gate_on_composition_valid=bool(cfg.reward.gate_on_composition_valid),
        gate_on_structure_valid=bool(cfg.reward.gate_on_structure_valid),
        gate_on_structure_comp_valid=bool(cfg.reward.gate_on_structure_comp_valid),
        gate_on_stable=bool(cfg.reward.gate_on_stable),
        prefilter_on_composition_valid=bool(cfg.reward.prefilter_on_composition_valid),
        prefilter_on_structure_valid=bool(cfg.reward.prefilter_on_structure_valid),
        prefilter_on_structure_comp_valid=bool(cfg.reward.prefilter_on_structure_comp_valid),
        prefilter_on_stable=bool(cfg.reward.prefilter_on_stable),
        prefilter_on_unique=bool(cfg.reward.prefilter_on_unique),
        prefilter_on_novel=bool(cfg.reward.prefilter_on_novel),
    )


def build_reward_table(cfg: DictConfig) -> pd.DataFrame:
    if cfg.source.kind == "released_final_evaluation":
        return build_released_final_evaluation_reward_table(
            sample_ledger=Path(cfg.source.sample_ledger),
            source_evaluation_csv=Path(cfg.source.source_evaluation_csv),
            guidance_factor=float(cfg.source.guidance_factor),
            conditions=[int(x) for x in cfg.source.conditions],
            reward_config=_build_reward_config(cfg),
        )
    if cfg.source.kind != "best_checkpoint_guidance_sweep":
        raise ValueError(f"Unsupported source.kind: {cfg.source.kind}")

    return build_best_checkpoint_guidance_sweep_reward_table(
        source_root=Path(cfg.source.root),
        combined_evaluation_csv=Path(cfg.source.combined_evaluation_csv),
        guidance_factor=float(cfg.source.guidance_factor),
        conditions=[int(x) for x in cfg.source.conditions],
        reward_config=_build_reward_config(cfg),
    )


def _prefilter_reward_table(reward_table: pd.DataFrame, reward_config: RewardConfig) -> pd.DataFrame:
    if reward_table.empty:
        return reward_table.copy()
    records = reward_table.to_dict(orient="records")
    filtered_records = [record for record in records if passes_prefilter(record, reward_config)]
    return pd.DataFrame(filtered_records)


def prepare_reward_table_and_buffer(cfg: DictConfig) -> dict[str, Any]:
    output_layout = _make_output_layout(run_name=str(cfg.run_name), output_root=cfg.output_root)
    reward_config = _build_reward_config(cfg)
    reward_table = build_reward_table(cfg)
    reward_table = _prefilter_reward_table(reward_table, reward_config)
    reward_table_path = output_layout["rewards"] / "reward_table.csv"
    reward_table.to_csv(reward_table_path, index=False)

    replay_buffer = initialize_replay_buffer(
        reward_table=reward_table,
        config=ReplayBufferConfig(
            enabled=bool(cfg.replay_buffer.enabled),
            buffer_size=int(cfg.replay_buffer.buffer_size),
            sample_size=int(cfg.replay_buffer.sample_size),
            reward_cutoff=float(cfg.replay_buffer.reward_cutoff),
            dedup_method=str(cfg.replay_buffer.dedup_method),
            topk_ratio=float(cfg.training.topk_ratio),
            eval_size=int(cfg.training.eval_size),
        ),
    )
    replay_buffer_path = output_layout["selected"] / "replay_buffer_init.csv"
    replay_buffer.to_csv(replay_buffer_path, index=False)

    summary = {
        "reward_table_path": str(reward_table_path),
        "replay_buffer_path": str(replay_buffer_path),
        "n_reward_rows": int(len(reward_table)),
        "n_selected_rows": int(len(replay_buffer)),
        "target_dielectric_scalar": float(cfg.reward.target_dielectric_scalar),
        "guidance_factor": float(cfg.source.guidance_factor),
        "source_conditions": [int(x) for x in cfg.source.conditions],
    }
    with open(output_layout["metadata"] / "prepare_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "reward_table": reward_table,
        "replay_buffer": replay_buffer,
        "summary": summary,
        "output_layout": output_layout,
    }


def train_from_replay_buffer(cfg: DictConfig) -> dict[str, Any]:
    prepared = prepare_reward_table_and_buffer(cfg)
    replay_buffer = prepared["replay_buffer"]
    if replay_buffer.empty:
        raise ValueError("Replay buffer is empty after reward gating and top-k selection.")

    dataset_result = build_crystal_dataset_from_records(
        records=replay_buffer.to_dict(orient="records"),
        target_dielectric_scalar=float(cfg.reward.target_dielectric_scalar),
        use_relaxed_if_available=True,
    )
    training_result = offline_reward_weighted_finetune(
        dataset=dataset_result.dataset,
        sample_weights=replay_buffer["reward_final"].to_numpy(),
        training_cfg=cfg.training,
        output_checkpoint_dir=prepared["output_layout"]["checkpoints"],
        predicted_dielectric_scalar=replay_buffer["predicted_dielectric_scalar"].to_numpy(),
        target_dielectric_scalar=float(cfg.reward.target_dielectric_scalar),
    )
    result = {
        **prepared["summary"],
        "n_dataset_rows": int(len(dataset_result.dataset)),
        "failed_structure_ids": dataset_result.failed_structure_ids,
        "history": training_result["history"],
        "trainable_parameters": training_result["trainable_parameters"],
        "topk_stats": training_result["topk_stats"],
    }
    with open(prepared["output_layout"]["metadata"] / "train_summary.json", "w") as handle:
        json.dump(result, handle, indent=2)
    return result
