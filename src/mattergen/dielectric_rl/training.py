from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.utils.globals import get_device
from mattergen.diffusion.corruption.multi_corruption import apply
from mattergen.diffusion.lightning_module import DiffusionLightningModule
from mattergen.diffusion.training.field_loss import aggregate_per_sample

from mattergen.dielectric_rl.dataset import RewardWeightedDataset


def load_lightning_module_from_paths(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device | None = None,
) -> DiffusionLightningModule:
    checkpoint_path = str(checkpoint_path)
    config = OmegaConf.load(str(config_path))
    if "lightning_module" not in config:
        raise ValueError(f"Config at {config_path} does not contain a lightning_module section.")
    device = device or get_device()
    model, incompatible = DiffusionLightningModule.load_from_checkpoint_and_config(
        checkpoint_path=checkpoint_path,
        config=config.lightning_module,
        map_location=device,
        strict=True,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint/config mismatch while loading RL fine-tuning model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.to(device)


def freeze_for_strategy(
    lightning_module: DiffusionLightningModule,
    freeze_strategy: str,
    trainable_parameter_keywords: list[str],
) -> list[str]:
    for _, parameter in lightning_module.named_parameters():
        parameter.requires_grad_(False)

    selected: list[str] = []
    if freeze_strategy == "all_trainable":
        for name, parameter in lightning_module.named_parameters():
            parameter.requires_grad_(True)
            selected.append(name)
        return selected

    if freeze_strategy != "adapter_only":
        raise ValueError(f"Unsupported freeze_strategy: {freeze_strategy}")

    for name, parameter in lightning_module.named_parameters():
        if any(keyword in name for keyword in trainable_parameter_keywords):
            parameter.requires_grad_(True)
            selected.append(name)

    if not selected:
        raise ValueError(
            "No trainable parameters matched the configured keywords for adapter_only training."
        )
    return selected


def _compute_per_sample_diffusion_loss(
    lightning_module: DiffusionLightningModule,
    clean_batch,
    noisy_batch,
    score_model_output,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diffusion_module = lightning_module.diffusion_module
    loss_fn = diffusion_module.loss_fn
    if not hasattr(loss_fn, "loss_fns") or not hasattr(loss_fn, "loss_weights"):
        raise NotImplementedError(
            "MatInvent-style fine-tuning currently expects a SummedFieldLoss-like loss implementation."
        )

    batch_idx = {field_name: clean_batch.get_batch_idx(field_name) for field_name in loss_fn.loss_fns}
    node_is_unmasked = {field_name: None for field_name in loss_fn.loss_fns}
    per_field = apply(
        fns=loss_fn.loss_fns,
        corruption=diffusion_module.corruption.corruptions,
        x=clean_batch,
        noisy_x=noisy_batch,
        score_model_output=score_model_output,
        batch_idx=batch_idx,
        broadcast=dict(
            t=timesteps,
            batch_size=clean_batch.get_batch_size(),
            batch=clean_batch,
        ),
        node_is_unmasked=node_is_unmasked,
    )
    total_per_sample = torch.stack(
        [loss_fn.loss_weights[field_name] * per_field[field_name] for field_name in per_field],
        dim=0,
    ).sum(0)
    mean_metrics = {f"loss_{field_name}": values.mean() for field_name, values in per_field.items()}
    return total_per_sample, mean_metrics


def _is_adapter_penalty_param(name: str) -> bool:
    return (
        "dft_dielectric_scalar" in name
        or "cond_adapt" in name
        or "cond_mixin" in name
    )


def snapshot_reference_params(
    policy_module: DiffusionLightningModule,
) -> tuple[dict[str, torch.Tensor], int]:
    reference_params: dict[str, torch.Tensor] = {}
    total_trainable_numel = 0
    for name, param in policy_module.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_adapter_penalty_param(name):
            continue
        reference_params[name] = param.detach().clone()
        total_trainable_numel += int(param.numel())
    if not reference_params:
        raise ValueError("No trainable adapter parameters were captured for parameter-space KL.")
    return reference_params, total_trainable_numel


def compute_param_l2_penalty(
    current_model: DiffusionLightningModule,
    reference_params: dict[str, torch.Tensor],
    total_trainable_numel: int,
) -> torch.Tensor:
    penalty = None
    for name, param in current_model.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_adapter_penalty_param(name):
            continue
        ref_param = reference_params[name].to(device=param.device, dtype=param.dtype)
        term = (param - ref_param).square().sum()
        penalty = term if penalty is None else penalty + term
    if penalty is None:
        device = get_device()
        penalty = torch.zeros((), device=device)
    if total_trainable_numel <= 0:
        raise ValueError("total_trainable_numel must be positive for mean-square drift.")
    return penalty / float(total_trainable_numel)


def _build_time_schedule(diffusion_module, timesteps: int, device: torch.device) -> torch.Tensor:
    max_t = diffusion_module.corruption.T
    return torch.linspace(max_t, 1.0 / float(timesteps), timesteps, device=device)


def _add_noise_at_timestep(policy_module: DiffusionLightningModule, batch, timestep_index: int, timesteps: int):
    diffusion_module = policy_module.diffusion_module
    clean_batch = diffusion_module.pre_corruption_fn(batch)
    device = diffusion_module._get_device(clean_batch)
    time_list = _build_time_schedule(diffusion_module, timesteps=timesteps, device=device)
    t = torch.full(
        (clean_batch.get_batch_size(),),
        time_list[int(timestep_index)],
        device=device,
    )
    noisy_batch = diffusion_module.corruption.sample_marginal(clean_batch, t)
    return clean_batch, noisy_batch, t


def _collect_topk_stats(
    rewards: np.ndarray,
    predicted_dielectric_scalar: np.ndarray | None,
    target_dielectric_scalar: float | None,
) -> dict[str, float]:
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    stats = {
        "topk_count": float(len(reward_tensor)),
        "topk_reward_min": float(reward_tensor.min().item()),
        "topk_reward_mean": float(reward_tensor.mean().item()),
        "topk_reward_max": float(reward_tensor.max().item()),
    }
    if predicted_dielectric_scalar is not None and len(predicted_dielectric_scalar) > 0:
        predicted_array = np.asarray(predicted_dielectric_scalar, dtype=np.float32)
        stats.update(
            {
                "topk_dielectric_min": float(np.min(predicted_array)),
                "topk_dielectric_mean": float(np.mean(predicted_array)),
                "topk_dielectric_max": float(np.max(predicted_array)),
            }
        )
        if target_dielectric_scalar is not None:
            stats["topk_within_target_1"] = float(
                np.mean(np.abs(predicted_array - float(target_dielectric_scalar)) < 1.0)
            )
    return stats


def save_finetuned_checkpoint(
    source_checkpoint_path: str | Path,
    policy_module: DiffusionLightningModule,
    output_checkpoint_path: str | Path,
) -> None:
    source_checkpoint = torch.load(str(source_checkpoint_path), map_location="cpu")
    source_checkpoint["state_dict"] = deepcopy(policy_module.state_dict())
    torch.save(source_checkpoint, str(output_checkpoint_path))


def _matinvent_style_epoch(
    policy_module: DiffusionLightningModule,
    reference_params: dict[str, torch.Tensor],
    total_trainable_numel: int,
    optimizer: torch.optim.Optimizer,
    reward_batch,
    anchor_batch,
    timesteps: int,
    accum_steps: int,
    kl_weight: float,
    epoch_index: int,
    optimizer_step_start: int,
) -> tuple[dict[str, float], list[dict[str, float]], int]:
    optimizer.zero_grad(set_to_none=True)
    reward_tensor = reward_batch.sample_weight.float()

    total_loss_scalar = 0.0
    total_loss_diff = 0.0
    total_loss_kl = 0.0
    total_loss_diff_raw = 0.0
    total_loss_kl_raw = 0.0
    field_metric_sums: dict[str, float] = {}
    kl_step_trace_rows: list[dict[str, float]] = []
    optimizer_step_idx = int(optimizer_step_start)
    last_timestep_index = -1
    last_rewarded_diffusion_loss = 0.0
    last_loss_diffusion_raw = 0.0
    last_drift_penalty_pre_step = 0.0

    def _record_optimizer_step_trace() -> None:
        nonlocal optimizer_step_idx
        with torch.no_grad():
            drift_penalty_post_step = float(
                compute_param_l2_penalty(
                    current_model=policy_module,
                    reference_params=reference_params,
                    total_trainable_numel=total_trainable_numel,
                ).detach().cpu()
            )
        kl_step_trace_rows.append(
            {
                "epoch": int(epoch_index),
                "optimizer_step": int(optimizer_step_idx),
                "timestep_index_end": int(last_timestep_index),
                "rewarded_diffusion_loss": float(last_rewarded_diffusion_loss),
                "loss_diffusion_raw": float(last_loss_diffusion_raw),
                "drift_penalty_pre_step": float(last_drift_penalty_pre_step),
                "drift_penalty_post_step": float(drift_penalty_post_step),
                "weighted_kl_over_rewarded_diffusion_pre_step": float(kl_weight)
                * float(last_drift_penalty_pre_step)
                / max(float(last_rewarded_diffusion_loss), 1e-12),
                "weighted_kl_over_rewarded_diffusion_post_step": float(kl_weight)
                * float(drift_penalty_post_step)
                / max(float(last_rewarded_diffusion_loss), 1e-12),
            }
        )
        optimizer_step_idx += 1

    for timestep_index in range(int(timesteps)):
        clean_batch, noisy_batch, current_t = _add_noise_at_timestep(
            policy_module=policy_module,
            batch=reward_batch,
            timestep_index=timestep_index,
            timesteps=timesteps,
        )
        policy_output = policy_module.diffusion_module.model(noisy_batch, current_t)
        sample_loss, field_metrics = _compute_per_sample_diffusion_loss(
            lightning_module=policy_module,
            clean_batch=clean_batch,
            noisy_batch=noisy_batch,
            score_model_output=policy_output,
            timesteps=current_t,
        )
        loss_diff_vec = reward_tensor * sample_loss
        loss_diff_scalar = loss_diff_vec.mean()

        # Use an L2 penalty against the frozen reference parameters.
        _ = anchor_batch
        loss_kl_scalar = compute_param_l2_penalty(
            current_model=policy_module,
            reference_params=reference_params,
            total_trainable_numel=total_trainable_numel,
        )

        step_loss = (loss_diff_scalar + loss_kl_scalar * kl_weight) / float(accum_steps)
        step_loss.backward()
        last_timestep_index = int(timestep_index)
        last_rewarded_diffusion_loss = float(loss_diff_scalar.detach().cpu())
        last_loss_diffusion_raw = float(sample_loss.mean().detach().cpu())
        last_drift_penalty_pre_step = float(loss_kl_scalar.detach().cpu())

        if (timestep_index + 1) % int(accum_steps) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _record_optimizer_step_trace()

        total_loss_scalar += float(step_loss.detach().cpu()) * float(accum_steps)
        total_loss_diff += float(loss_diff_scalar.detach().cpu())
        total_loss_kl += float(loss_kl_scalar.detach().cpu())
        total_loss_diff_raw += float(sample_loss.mean().detach().cpu())
        total_loss_kl_raw += float(loss_kl_scalar.detach().cpu())
        for key, value in field_metrics.items():
            field_metric_sums[key] = field_metric_sums.get(key, 0.0) + float(value.detach().cpu())

    if int(timesteps) % int(accum_steps) != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _record_optimizer_step_trace()

    metrics = {
        "loss_total": total_loss_scalar / float(timesteps),
        "loss_diffusion": total_loss_diff / float(timesteps),
        "loss_reference_kl": total_loss_kl / float(timesteps),
        "loss_diffusion_raw": total_loss_diff_raw / float(timesteps),
        "loss_kl_raw": total_loss_kl_raw / float(timesteps),
    }
    for key, value in field_metric_sums.items():
        metrics[key] = value / float(timesteps)
    return metrics, kl_step_trace_rows, optimizer_step_idx


def offline_reward_weighted_finetune(
    dataset: CrystalDataset,
    sample_weights,
    training_cfg,
    output_checkpoint_dir: Path,
    predicted_dielectric_scalar: np.ndarray | None = None,
    target_dielectric_scalar: float | None = None,
    anchor_dataset: CrystalDataset | None = None,
) -> dict[str, Any]:
    device = get_device()
    policy_module = load_lightning_module_from_paths(
        checkpoint_path=training_cfg.checkpoint_path,
        config_path=training_cfg.config_path,
        device=device,
    )
    selected_params = freeze_for_strategy(
        lightning_module=policy_module,
        freeze_strategy=training_cfg.freeze_strategy,
        trainable_parameter_keywords=list(training_cfg.trainable_parameter_keywords),
    )
    # Freeze the reference parameters for this fine-tuning run.
    reference_params, total_trainable_numel = snapshot_reference_params(policy_module)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy_module.parameters() if parameter.requires_grad],
        lr=float(training_cfg.learning_rate),
    )

    sample_weights_array = np.asarray(sample_weights, dtype=np.float32)
    weighted_dataset = RewardWeightedDataset(
        base_dataset=dataset,
        sample_weights=sample_weights_array,
    )
    data_loader = DataLoader(
        weighted_dataset,
        batch_size=len(weighted_dataset),
        shuffle=False,
        num_workers=int(getattr(training_cfg, "num_workers", 0)),
        collate_fn=collate,
    )
    anchor_data_loader = None
    if anchor_dataset is not None and len(anchor_dataset) > 0:
        anchor_data_loader = DataLoader(
            anchor_dataset,
            batch_size=len(anchor_dataset),
            shuffle=False,
            num_workers=int(getattr(training_cfg, "num_workers", 0)),
            collate_fn=collate,
        )

    topk_stats = _collect_topk_stats(
        rewards=sample_weights_array,
        predicted_dielectric_scalar=predicted_dielectric_scalar,
        target_dielectric_scalar=target_dielectric_scalar,
    )

    history: list[dict[str, float]] = []
    kl_step_trace_rows: list[dict[str, float]] = []
    optimizer_step_idx = 0
    policy_module.train()
    for epoch in range(int(training_cfg.epochs)):
        epoch_metrics: list[dict[str, float]] = []
        reward_batches = list(data_loader)
        anchor_batches = list(anchor_data_loader) if anchor_data_loader is not None else [None]
        if len(reward_batches) != 1:
            raise ValueError("MatInvent-style finetune currently expects a single reward batch per epoch.")
        if anchor_data_loader is not None and len(anchor_batches) != 1:
            raise ValueError("Anchor-batch KL currently expects a single anchor batch per epoch.")

        for reward_batch in reward_batches:
            reward_batch = reward_batch.to(device)
            anchor_batch = anchor_batches[0]
            if anchor_batch is not None:
                anchor_batch = anchor_batch.to(device)
            metrics, epoch_step_trace_rows, optimizer_step_idx = _matinvent_style_epoch(
                policy_module=policy_module,
                reference_params=reference_params,
                total_trainable_numel=total_trainable_numel,
                optimizer=optimizer,
                reward_batch=reward_batch,
                anchor_batch=anchor_batch,
                timesteps=int(training_cfg.timesteps),
                accum_steps=int(training_cfg.accum_steps),
                kl_weight=float(training_cfg.kl_weight),
                epoch_index=epoch,
                optimizer_step_start=optimizer_step_idx,
            )
            epoch_metrics.append(metrics)
            kl_step_trace_rows.extend(epoch_step_trace_rows)

        if not epoch_metrics:
            continue

        mean_metrics = {
            key: float(sum(metric[key] for metric in epoch_metrics) / len(epoch_metrics))
            for key in epoch_metrics[0]
        }
        mean_metrics["epoch"] = epoch
        history.append(mean_metrics)
        if bool(getattr(training_cfg, "checkpoint_every_epoch", True)):
            save_finetuned_checkpoint(
                source_checkpoint_path=training_cfg.checkpoint_path,
                policy_module=policy_module,
                output_checkpoint_path=output_checkpoint_dir / f"epoch_{epoch:03d}.ckpt",
            )

    save_finetuned_checkpoint(
        source_checkpoint_path=training_cfg.checkpoint_path,
        policy_module=policy_module,
        output_checkpoint_path=output_checkpoint_dir / "final.ckpt",
    )
    kl_step_trace_path = output_checkpoint_dir.parent / "kl_step_trace.csv"
    with kl_step_trace_path.open("w", newline="") as handle:
        fieldnames = [
            "epoch",
            "optimizer_step",
            "timestep_index_end",
            "rewarded_diffusion_loss",
            "loss_diffusion_raw",
            "drift_penalty_pre_step",
            "drift_penalty_post_step",
            "weighted_kl_over_rewarded_diffusion_pre_step",
            "weighted_kl_over_rewarded_diffusion_post_step",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kl_step_trace_rows)
    return {
        "trainable_parameters": selected_params,
        "history": history,
        "topk_stats": topk_stats,
        "anchor_batch_count": int(len(anchor_dataset)) if anchor_dataset is not None else 0,
        "total_trainable_numel": int(total_trainable_numel),
        "kl_step_trace_path": str(kl_step_trace_path),
        "optimizer_step_count": int(len(kl_step_trace_rows)),
    }


def inspect_single_step_loss_scale(
    dataset: CrystalDataset,
    sample_weights,
    training_cfg,
    anchor_dataset: CrystalDataset | None = None,
    timestep_index: int = 0,
) -> dict[str, float]:
    device = get_device()
    policy_module = load_lightning_module_from_paths(
        checkpoint_path=training_cfg.checkpoint_path,
        config_path=training_cfg.config_path,
        device=device,
    )
    freeze_for_strategy(
        lightning_module=policy_module,
        freeze_strategy=training_cfg.freeze_strategy,
        trainable_parameter_keywords=list(training_cfg.trainable_parameter_keywords),
    )
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in policy_module.parameters()
            if parameter.requires_grad
        ],
        lr=float(getattr(training_cfg, "learning_rate", 1.0e-5)),
    )
    reference_params, total_trainable_numel = snapshot_reference_params(policy_module)

    sample_weights_array = np.asarray(sample_weights, dtype=np.float32)
    weighted_dataset = RewardWeightedDataset(
        base_dataset=dataset,
        sample_weights=sample_weights_array,
    )
    reward_batch = next(
        iter(
            DataLoader(
                weighted_dataset,
                batch_size=len(weighted_dataset),
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )
        )
    ).to(device)

    anchor_batch = None
    if anchor_dataset is not None and len(anchor_dataset) > 0:
        anchor_batch = next(
            iter(
                DataLoader(
                    anchor_dataset,
                    batch_size=len(anchor_dataset),
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collate,
                )
            )
        ).to(device)

    reward_tensor = reward_batch.sample_weight.float()
    clean_batch, noisy_batch, current_t = _add_noise_at_timestep(
        policy_module=policy_module,
        batch=reward_batch,
        timestep_index=int(timestep_index),
        timesteps=int(training_cfg.timesteps),
    )
    policy_output = policy_module.diffusion_module.model(noisy_batch, current_t)
    sample_loss, _ = _compute_per_sample_diffusion_loss(
        lightning_module=policy_module,
        clean_batch=clean_batch,
        noisy_batch=noisy_batch,
        score_model_output=policy_output,
        timesteps=current_t,
    )
    loss_diffusion_raw = float(sample_loss.mean().detach().cpu())
    reward_weighted_diffusion_loss = float((reward_tensor * sample_loss).mean().detach().cpu())
    _ = anchor_batch
    loss_kl_raw_before_step = float(
        compute_param_l2_penalty(
            current_model=policy_module,
            reference_params=reference_params,
            total_trainable_numel=total_trainable_numel,
        ).detach().cpu()
    )

    step_loss = reward_weighted_diffusion_loss + float(training_cfg.kl_weight) * loss_kl_raw_before_step
    optimizer.zero_grad(set_to_none=True)
    (
        (reward_tensor * sample_loss).mean()
        + float(training_cfg.kl_weight)
        * compute_param_l2_penalty(
            current_model=policy_module,
            reference_params=reference_params,
            total_trainable_numel=total_trainable_numel,
        )
    ).backward()
    optimizer.step()

    loss_kl_raw_after_step = float(
        compute_param_l2_penalty(
            current_model=policy_module,
            reference_params=reference_params,
            total_trainable_numel=total_trainable_numel,
        ).detach().cpu()
    )
    ratio_before = loss_kl_raw_before_step / max(loss_diffusion_raw, 1e-12)
    weighted_ratio_before = float(training_cfg.kl_weight) * loss_kl_raw_before_step / max(loss_diffusion_raw, 1e-12)
    ratio_after = loss_kl_raw_after_step / max(loss_diffusion_raw, 1e-12)
    weighted_ratio_after = float(training_cfg.kl_weight) * loss_kl_raw_after_step / max(loss_diffusion_raw, 1e-12)
    return {
        "loss_diffusion_raw": loss_diffusion_raw,
        "reward_weighted_diffusion_loss": reward_weighted_diffusion_loss,
        "loss_kl_raw_before_step": loss_kl_raw_before_step,
        "loss_kl_raw_after_step": loss_kl_raw_after_step,
        "loss_kl_over_diffusion_raw_before_step": ratio_before,
        "weighted_kl_over_diffusion_raw_before_step": weighted_ratio_before,
        "loss_kl_over_diffusion_raw_after_step": ratio_after,
        "weighted_kl_over_diffusion_raw_after_step": weighted_ratio_after,
    }
