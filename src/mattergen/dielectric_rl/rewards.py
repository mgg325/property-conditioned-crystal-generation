from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        if lowered in {"", "nan", "none", "null"}:
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


@dataclass(frozen=True)
class RewardConfig:
    target_dielectric_scalar: float
    sigma: float
    gate_on_missing_prediction: bool = True
    gate_on_composition_valid: bool = True
    gate_on_structure_valid: bool = True
    gate_on_structure_comp_valid: bool = True
    gate_on_stable: bool = False
    prefilter_on_composition_valid: bool = True
    prefilter_on_structure_valid: bool = True
    prefilter_on_structure_comp_valid: bool = True
    prefilter_on_stable: bool = True
    prefilter_on_unique: bool = True
    prefilter_on_novel: bool = True

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("reward sigma must be positive")


def gaussian_target_reward(
    predicted_dielectric_scalar: float,
    target_dielectric_scalar: float,
    sigma: float,
) -> float:
    squared_error = (predicted_dielectric_scalar - target_dielectric_scalar) ** 2
    return math.exp(-(squared_error) / (2.0 * sigma**2))


def evaluate_reward_gate(record: dict[str, Any], config: RewardConfig) -> tuple[bool, str]:
    reasons: list[str] = []

    predicted = _as_float(record.get("predicted_dielectric_scalar"))
    if config.gate_on_missing_prediction and predicted is None:
        reasons.append("missing_prediction")

    if config.gate_on_composition_valid:
        composition_valid = _as_bool(record.get("composition_valid"))
        if composition_valid is False:
            reasons.append("composition_invalid")

    if config.gate_on_structure_valid:
        structure_valid = _as_bool(record.get("structure_valid"))
        if structure_valid is False:
            reasons.append("structure_invalid")

    if config.gate_on_structure_comp_valid:
        structure_comp_valid = _as_bool(record.get("structure_comp_valid"))
        if structure_comp_valid is False:
            reasons.append("structure_comp_invalid")

    if config.gate_on_stable:
        stable = _as_bool(record.get("stable"))
        if stable is False:
            reasons.append("unstable")

    relaxation_success = _as_bool(record.get("relaxation_success"))
    if relaxation_success is False:
        reasons.append("relaxation_failed")

    prediction_success = _as_bool(record.get("prediction_success"))
    if prediction_success is False:
        reasons.append("prediction_failed")

    return (len(reasons) == 0, ";".join(reasons))


def passes_prefilter(record: dict[str, Any], config: RewardConfig) -> bool:
    if config.prefilter_on_composition_valid:
        composition_valid = _as_bool(record.get("composition_valid"))
        if composition_valid is False:
            return False

    if config.prefilter_on_structure_valid:
        structure_valid = _as_bool(record.get("structure_valid"))
        if structure_valid is False:
            return False

    if config.prefilter_on_structure_comp_valid:
        structure_comp_valid = _as_bool(record.get("structure_comp_valid"))
        if structure_comp_valid is False:
            return False

    if config.prefilter_on_stable:
        stable = _as_bool(record.get("stable"))
        if stable is False:
            return False

    if config.prefilter_on_unique:
        is_unique = _as_bool(record.get("is_unique"))
        if is_unique is False:
            return False

    if config.prefilter_on_novel:
        is_novel = _as_bool(record.get("is_novel"))
        if is_novel is False:
            return False

    return True


def build_reward_row(record: dict[str, Any], config: RewardConfig) -> dict[str, Any]:
    predicted = _as_float(record.get("predicted_dielectric_scalar"))
    abs_error = (
        abs(predicted - config.target_dielectric_scalar) if predicted is not None else None
    )
    reward_raw = (
        gaussian_target_reward(
            predicted_dielectric_scalar=predicted,
            target_dielectric_scalar=config.target_dielectric_scalar,
            sigma=config.sigma,
        )
        if predicted is not None
        else 0.0
    )
    passed_gate, gate_reason = evaluate_reward_gate(record, config)
    reward_final = reward_raw if passed_gate else 0.0

    return {
        **record,
        "target_dielectric_scalar": config.target_dielectric_scalar,
        "predicted_dielectric_scalar": predicted,
        "abs_error": abs_error,
        "reward_raw": reward_raw,
        "reward_final": reward_final,
        "gate_reason": gate_reason,
    }
