from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition


def _safe_reduced_formula(formula: Any) -> str | None:
    if formula is None or (isinstance(formula, float) and pd.isna(formula)):
        return None
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def _safe_element_combination(record: dict[str, Any]) -> tuple[str, ...] | None:
    chemical_system = record.get("chemical_system")
    if chemical_system is None or (isinstance(chemical_system, float) and pd.isna(chemical_system)):
        return None
    if isinstance(chemical_system, str):
        parts = [part.strip() for part in chemical_system.split("-") if part.strip()]
        if parts:
            return tuple(sorted(parts))
    return None


@dataclass(frozen=True)
class ReplayBufferConfig:
    enabled: bool = True
    buffer_size: int = 100
    sample_size: int = 10
    reward_cutoff: float = 0.0
    dedup_method: str = "composition"
    topk_ratio: float = 0.5
    eval_size: int = 16
    recent_window: int = 0
    recent_fraction: float = 0.0


@dataclass(frozen=True)
class DiversityFilterConfig:
    enabled: bool = True
    tol: int = 3
    buff: int = 6
    method: str = "composition"


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.buffer = pd.DataFrame(
            columns=["record", "comp", "ele_comb", "reward", "source_loop"]
        )

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = df.sort_values("reward", ascending=False)
        if self.config.dedup_method == "composition":
            return sorted_df.drop_duplicates(subset=["comp"])
        if self.config.dedup_method == "element_comb":
            return sorted_df.drop_duplicates(subset=["ele_comb"])
        raise ValueError(f"Unsupported replay dedup method: {self.config.dedup_method}")

    def extend(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        df_new = pd.DataFrame(
            {
                "record": records,
                "comp": [_safe_reduced_formula(record.get("formula")) for record in records],
                "ele_comb": [_safe_element_combination(record) for record in records],
                "reward": [float(record.get("reward_final", 0.0)) for record in records],
                "source_loop": [
                    int(record.get("source_loop", record.get("loop_idx", -1))) for record in records
                ],
            }
        )
        df_all = pd.concat([self.buffer, df_new], ignore_index=True)
        unique_df = self._deduplicate(df_all)
        sorted_df = unique_df.sort_values("reward", ascending=False)
        filtered_df = sorted_df[sorted_df["reward"] > float(self.config.reward_cutoff)]
        self.buffer = filtered_df.head(int(self.config.buffer_size)).reset_index(drop=True)

    def sample(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        sample_size = min(len(self.buffer), int(self.config.sample_size))
        if sample_size <= 0:
            return [], np.asarray([], dtype=np.float32)
        sampled = self.buffer.sample(sample_size)
        records = sampled["record"].tolist()
        rewards = sampled["reward"].to_numpy(dtype=np.float32)
        return records, rewards

    def sample_mixed(
        self,
        *,
        current_loop: int,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        sample_size = min(len(self.buffer), int(self.config.sample_size))
        if sample_size <= 0:
            return [], np.asarray([], dtype=np.float32)

        recent_window = int(self.config.recent_window)
        recent_fraction = float(self.config.recent_fraction)
        if recent_window <= 0 or recent_fraction <= 0.0 or "source_loop" not in self.buffer.columns:
            return self.sample()

        recent_cutoff = int(current_loop) - recent_window
        recent_pool = self.buffer[self.buffer["source_loop"].fillna(-1).astype(int) > recent_cutoff]
        global_pool = self.buffer
        recent_target = int(round(sample_size * recent_fraction))
        recent_target = max(0, min(sample_size, recent_target))
        recent_take = min(len(recent_pool), recent_target)

        sampled_parts: list[pd.DataFrame] = []
        if recent_take > 0:
            sampled_recent = recent_pool.sample(recent_take)
            sampled_parts.append(sampled_recent)
            global_pool = global_pool.drop(index=sampled_recent.index)

        remaining = sample_size - recent_take
        if remaining > 0 and len(global_pool) > 0:
            sampled_global = global_pool.sample(min(remaining, len(global_pool)))
            sampled_parts.append(sampled_global)

        if not sampled_parts:
            return [], np.asarray([], dtype=np.float32)

        sampled = pd.concat(sampled_parts, ignore_index=False)
        if len(sampled) < sample_size:
            deficit = sample_size - len(sampled)
            refill_pool = self.buffer.drop(index=sampled.index, errors="ignore")
            if deficit > 0 and len(refill_pool) > 0:
                refill = refill_pool.sample(min(deficit, len(refill_pool)))
                sampled = pd.concat([sampled, refill], ignore_index=False)

        sampled = sampled.head(sample_size)
        records = sampled["record"].tolist()
        rewards = sampled["reward"].to_numpy(dtype=np.float32)
        return records, rewards

    def purge(self, records: list[dict[str, Any]]) -> None:
        comps_to_remove = {_safe_reduced_formula(record.get("formula")) for record in records}
        comps_to_remove.discard(None)
        if not comps_to_remove:
            return
        self.buffer = self.buffer[~self.buffer["comp"].isin(comps_to_remove)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.buffer)


class LongTermMemory:
    def __init__(self) -> None:
        self.memory = pd.DataFrame(columns=["comp", "ele_comb", "reward", "rl_step"])

    def extend(self, records: list[dict[str, Any]], rewards: np.ndarray, step: int) -> None:
        if not records:
            return
        df = pd.DataFrame(
            {
                "comp": [_safe_reduced_formula(record.get("formula")) for record in records],
                "ele_comb": [_safe_element_combination(record) for record in records],
                "reward": rewards.astype(float),
                "rl_step": [step] * len(records),
            }
        )
        self.memory = pd.concat([self.memory, df], ignore_index=True)

    def div_filter(
        self,
        records: list[dict[str, Any]],
        rewards: np.ndarray,
        config: DiversityFilterConfig,
    ) -> tuple[np.ndarray, list[int], int, int]:
        if not config.enabled or len(records) == 0:
            return rewards.astype(np.float32), [], 0, 0

        if config.method == "composition":
            key_values = [_safe_reduced_formula(record.get("formula")) for record in records]
            key = "comp"
        elif config.method == "element_comb":
            key_values = [_safe_element_combination(record) for record in records]
            key = "ele_comb"
        else:
            raise ValueError(f"Unsupported diversity filter method: {config.method}")

        adjusted = []
        penalty_idx: list[int] = []
        tol_n = 0
        buff_n = 0
        value_counts = self.memory[key].value_counts() if len(self.memory) > 0 else pd.Series(dtype=int)
        for index, value in enumerate(key_values):
            occ = int(value_counts.get(value, 0))
            if occ <= int(config.tol):
                adjusted.append(float(rewards[index]))
            elif int(config.tol) < occ < int(config.buff):
                adjusted.append(float(rewards[index]) * (int(config.buff) - occ) / (int(config.buff) - int(config.tol)))
                tol_n += 1
            else:
                adjusted.append(0.0)
                penalty_idx.append(index)
                buff_n += 1
        return np.asarray(adjusted, dtype=np.float32), penalty_idx, tol_n, buff_n

    def get_baseline(self, step: int, prev: int = 3) -> float:
        recent = self.memory[self.memory["rl_step"] > step - prev]
        if recent.empty:
            return 0.0
        return float(recent["reward"].mean())


def select_topk_records(
    reward_table: pd.DataFrame,
    topk_ratio: float,
    eval_size: int,
) -> pd.DataFrame:
    if reward_table.empty:
        return reward_table.copy()
    if not (0.0 < float(topk_ratio) <= 1.0):
        raise ValueError("topk_ratio must be in (0, 1].")
    topk = max(1, int(int(eval_size) * float(topk_ratio)))
    sorted_df = reward_table.sort_values("reward_final", ascending=False).reset_index(drop=True)
    selected = sorted_df.head(topk).copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def initialize_replay_buffer(
    reward_table: pd.DataFrame,
    config: ReplayBufferConfig,
) -> pd.DataFrame:
    filtered = reward_table.copy()
    filtered = filtered[filtered["reward_final"].fillna(0.0) > float(config.reward_cutoff)].copy()
    filtered = filtered.sort_values(
        ["reward_final", "reward_raw", "abs_error"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return select_topk_records(
        reward_table=filtered,
        topk_ratio=float(config.topk_ratio),
        eval_size=int(config.eval_size),
    )
