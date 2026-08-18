from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

# Ensure repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.dataset import (
    compute_allowed_elements,
    compute_dataset_element_distribution,
    dataset_to_structures,
    ensure_dataset_splits,
    sample_input_stats,
)
from src.utils.ema import EMA

from src.data.mp20_tokens import (
    MP20Tokens,
    collate_mp20_tokens,
    NMAX as DEFAULT_NMAX,
    VZ,
)

from src.eval.stability import _compute_thermo_metrics
from src.eval.wasserstein import _compute_wasserstein_metrics
from src.models.type_encoding import build_type_encoding
from src.models.lattice_repr import (
    lattice_latent_to_y1,
    y1_to_lattice_latent,
)

from src.utils.sample_stats import collect_structure_stats
from src.utils.stability_logger import StabilityLogger, _ThermoConfig
from src.utils.wandb_utils import init_wandb, log_images, log_metrics
from src.utils.constants import DATASET_NMAX_DEFAULTS, _DIAGNOSTIC_SECTION_KEYS
from src.utils.seeding import seed_everything, seed_dataloader_worker
from src.utils.checkpoint import (
    BestCkptState,
    _build_best_candidate,
    build_val_fallback_candidate,
    maybe_update_best_ckpt,
    resolve_post_training_eval_ckpt,
    select_primary_candidate_from_sampling,
    BEST_CKPT_SELECTOR_CHOICES,
)

from src.crystalite.sampler import (
    clamp_lattice_latent as _clamp_lattice_latent,
    edm_sampler,
    wrap_frac,
)
from src.crystalite.edm_utils import (
    sample_sigma,
    denoise_edm,
    compute_edm_loss,
    compute_edm_loss_per_sample,
)
from src.crystalite import CrystaliteModel, mod1
from src.eval.sample_runtime import (
    SamplingContext,
    SamplingRequest,
    generate_sampling_batch,
    evaluate_dng_sampling_batch,
    evaluate_csp_sampling_batch,
    save_sampling_artifacts,
    maybe_use_ema,
    _build_sampling_runs,
)
from scripts.diagnose_fused_diff_sweep import compute_fused_diff_sweep

_build_best_candidate = _build_best_candidate
_BestCkptState = BestCkptState
_build_val_fallback_candidate = build_val_fallback_candidate
_maybe_update_best_ckpt = maybe_update_best_ckpt
_select_primary_candidate_from_sampling = select_primary_candidate_from_sampling
_resolve_post_training_eval_ckpt = resolve_post_training_eval_ckpt
_compute_allowed_elements = compute_allowed_elements
_dataset_to_structures = dataset_to_structures


def _load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def load_finetune_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    allowed_missing_prefixes: tuple[str, ...] = (
        "property_embed.",
        "property_film.",
        "property_fuse.",
        "property_delta.",
        "property_alpha",
    ),
) -> dict[str, Any]:
    """Initialize a model from an older checkpoint without resuming optimizer state."""
    ckpt = _load_checkpoint_file(checkpoint_path)
    state = ckpt.get("model_state_dict", None)
    if state is None:
        raise KeyError(f"{checkpoint_path} does not contain model_state_dict.")

    property_norm_before = None
    property_norm_from_checkpoint = None
    property_embed = getattr(model, "property_embed", None)
    property_film_norm_before = None
    property_film_norm_from_checkpoint = None
    property_film = getattr(model, "property_film", None)
    if property_embed is not None:
        property_norm_before = {
            "property_transform": getattr(property_embed, "property_transform", "identity"),
            "property_mean": float(property_embed.property_mean.detach().cpu().item()),
            "property_std": float(property_embed.property_std.detach().cpu().item()),
        }
        ckpt_mean = state.get("property_embed.property_mean", None)
        ckpt_std = state.get("property_embed.property_std", None)
        if ckpt_mean is not None and ckpt_std is not None:
            property_norm_from_checkpoint = {
                "property_mean": float(ckpt_mean.detach().cpu().item()),
                "property_std": float(ckpt_std.detach().cpu().item()),
            }
    if property_film is not None:
        property_film_norm_before = {
            "property_transform": getattr(property_film, "property_transform", "identity"),
            "property_mean": float(property_film.property_mean.detach().cpu().item()),
            "property_std": float(property_film.property_std.detach().cpu().item()),
        }
        ckpt_mean = state.get("property_film.property_mean", None)
        ckpt_std = state.get("property_film.property_std", None)
        if ckpt_mean is not None and ckpt_std is not None:
            property_film_norm_from_checkpoint = {
                "property_mean": float(ckpt_mean.detach().cpu().item()),
                "property_std": float(ckpt_std.detach().cpu().item()),
            }

    incompatible = model.load_state_dict(state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    property_norm_after_load = None
    property_norm_restored = None
    property_film_norm_after_load = None
    property_film_norm_restored = None
    if property_embed is not None and property_norm_before is not None:
        property_norm_after_load = {
            "property_transform": getattr(property_embed, "property_transform", "identity"),
            "property_mean": float(property_embed.property_mean.detach().cpu().item()),
            "property_std": float(property_embed.property_std.detach().cpu().item()),
        }
        property_embed.set_normalization(
            property_norm_before["property_mean"],
            property_norm_before["property_std"],
        )
        property_norm_restored = {
            "property_transform": getattr(property_embed, "property_transform", "identity"),
            "property_mean": float(property_embed.property_mean.detach().cpu().item()),
            "property_std": float(property_embed.property_std.detach().cpu().item()),
        }
    if property_film is not None and property_film_norm_before is not None:
        property_film_norm_after_load = {
            "property_transform": getattr(property_film, "property_transform", "identity"),
            "property_mean": float(property_film.property_mean.detach().cpu().item()),
            "property_std": float(property_film.property_std.detach().cpu().item()),
        }
        property_film.set_normalization(
            property_film_norm_before["property_mean"],
            property_film_norm_before["property_std"],
        )
        property_film_norm_restored = {
            "property_transform": getattr(property_film, "property_transform", "identity"),
            "property_mean": float(property_film.property_mean.detach().cpu().item()),
            "property_std": float(property_film.property_std.detach().cpu().item()),
        }

    def is_allowed_missing(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allowed_missing_prefixes)

    disallowed_missing = [key for key in missing_keys if not is_allowed_missing(key)]
    if disallowed_missing:
        raise RuntimeError(
            "Fine-tune checkpoint is missing non-property model keys: "
            f"{disallowed_missing}. Refusing to continue."
        )
    if unexpected_keys:
        raise RuntimeError(
            "Fine-tune checkpoint has unexpected model keys: "
            f"{unexpected_keys}. Refusing to continue."
        )

    model_keys = set(model.state_dict().keys())
    loaded_keys = sorted(key for key in state.keys() if key in model_keys)
    report = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "loaded_key_count": len(loaded_keys),
        "missing_key_count": len(missing_keys),
        "missing_keys": missing_keys,
        "unexpected_key_count": len(unexpected_keys),
        "unexpected_keys": unexpected_keys,
        "missing_keys_only_property_conditioning": True,
        "checkpoint_step": ckpt.get("step", None),
        "property_normalization_before_load": property_norm_before,
        "property_normalization_from_checkpoint": property_norm_from_checkpoint,
        "property_normalization_after_load": property_norm_after_load,
        "property_normalization_restored": property_norm_restored,
        "property_film_normalization_before_load": property_film_norm_before,
        "property_film_normalization_from_checkpoint": property_film_norm_from_checkpoint,
        "property_film_normalization_after_load": property_film_norm_after_load,
        "property_film_normalization_restored": property_film_norm_restored,
    }
    print(
        "[finetune] Loaded model initialization checkpoint "
        f"{report['checkpoint_path']}"
    )
    print(
        "[finetune] loaded_keys="
        f"{report['loaded_key_count']} missing_keys={report['missing_key_count']} "
        f"unexpected_keys={report['unexpected_key_count']}"
    )
    print(f"[finetune] missing_keys={missing_keys}")
    print(f"[finetune] unexpected_keys={unexpected_keys}")
    if missing_keys:
        print(
            "[finetune] Missing keys are limited to property-conditioning "
            "parameters; optimizer/scheduler/EMA/step are not restored."
        )
    else:
        print("[finetune] All model keys were present; optimizer/scheduler/EMA/step are not restored.")
    if property_norm_before is not None:
        print(
            "[finetune] property normalization "
            f"checkpoint={property_norm_from_checkpoint} "
            f"after_load={property_norm_after_load} "
            f"restored={property_norm_restored}"
        )
    if property_film_norm_before is not None:
        print(
            "[finetune] property FiLM normalization "
            f"checkpoint={property_film_norm_from_checkpoint} "
            f"after_load={property_film_norm_after_load} "
            f"restored={property_film_norm_restored}"
        )
    return report


def build_optimizer_with_property_lr(
    model: torch.nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    property_embed_lr_multiplier: float = 1.0,
    property_delta_lr_multiplier: float = 1.0,
) -> torch.optim.Optimizer:
    """Build AdamW with separate LR groups for property_embed and property_delta."""
    embed_names: list[str] = []
    embed_params: list[torch.nn.Parameter] = []
    delta_names: list[str] = []
    delta_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "property_embed" in name or "property_film" in name:
            embed_names.append(name)
            embed_params.append(param)
        elif "property_delta" in name:
            delta_names.append(name)
            delta_params.append(param)
        else:
            base_params.append(param)

    base_param_count = sum(p.numel() for p in base_params)
    embed_param_count = sum(p.numel() for p in embed_params)
    delta_param_count = sum(p.numel() for p in delta_params)
    embed_multiplier = float(property_embed_lr_multiplier)
    delta_multiplier = float(property_delta_lr_multiplier)
    embed_lr = float(base_lr) * embed_multiplier
    delta_lr = float(base_lr) * delta_multiplier
    embed_preview = ", ".join(embed_names[:8]) if embed_names else "(none)"
    delta_preview = ", ".join(delta_names[:8]) if delta_names else "(none)"
    print(
        "[optim] property LR groups: "
        f"base_lr={float(base_lr):.6g} "
        f"embed_lr={embed_lr:.6g} "
        f"delta_lr={delta_lr:.6g} "
        f"property_embed_lr_multiplier={embed_multiplier:.6g} "
        f"property_delta_lr_multiplier={delta_multiplier:.6g}"
    )
    print(
        "[optim] parameter tensors: "
        f"base={len(base_params)} embed={len(embed_params)} delta={len(delta_params)}"
    )
    print(
        "[optim] parameter counts: "
        f"base={base_param_count} embed={embed_param_count} delta={delta_param_count}"
    )
    print(f"[optim] property_embed parameter preview: {embed_preview}")
    print(f"[optim] property_delta parameter preview: {delta_preview}")

    if (embed_multiplier != 1.0 and not embed_params) or (
        delta_multiplier != 1.0 and not delta_params
    ):
        print(
            "[optim] property LR multiplier was set, but no trainable "
            "parameters were found for one or more requested property groups."
        )
    param_groups = [{"params": base_params, "lr": base_lr}]
    if embed_params:
        param_groups.append({"params": embed_params, "lr": embed_lr})
    if delta_params:
        param_groups.append({"params": delta_params, "lr": delta_lr})

    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


def _is_rank0() -> bool:
    rank = os.environ.get("RANK")
    if rank is None:
        rank = os.environ.get("LOCAL_RANK")
    return rank in (None, "", "0")


def _finite_median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.median(finite))


def _run_lattice_sanity_check(
    *,
    step: int,
    model: torch.nn.Module,
    sampling_ctx: SamplingContext,
    val_ds: Any,
    count: int = 32,
) -> dict[str, float | None]:
    from src.data.mp20_tokens import tokens_to_structure
    from src.eval_crystalite_samples import _minimum_distance

    diag_args = copy.copy(sampling_ctx.args)
    diag_args.sample_vis_count = 0
    diag_args.csp_precise_topk_list = []
    diag_args.csp_precise_topk_samples = 0
    diag_ctx = replace(
        sampling_ctx,
        args=diag_args,
        model=model,
        ema=None,
        wandb_enabled=False,
    )
    request = SamplingRequest(
        tag="diagnostic",
        step=step,
        base_seed=int(diag_args.sample_seed) + int(step),
        use_ema=False,
        metrics_count=int(count),
        csp_source_ds=val_ds if bool(diag_args.csp) else None,
        csp_source_label="val",
    )
    batch = generate_sampling_batch(request, diag_ctx)
    min_distances: list[float] = []
    volume_per_atom: list[float] = []
    densities: list[float] = []
    for item in batch.sample_items[:count]:
        try:
            structure = tokens_to_structure(item)
            num_atoms = int(len(structure))
            if num_atoms <= 0:
                continue
            min_distances.append(float(_minimum_distance(structure)))
            volume_per_atom.append(float(structure.volume) / float(num_atoms))
            densities.append(float(structure.density))
        except Exception:
            continue
    return {
        "min_distance_median": _finite_median(min_distances),
        "volume_per_atom_median": _finite_median(volume_per_atom),
        "density_median": _finite_median(densities),
    }


def _run_training_diagnostics(
    *,
    step: int,
    model: torch.nn.Module,
    ema: EMA | None,
    sampling_ctx: SamplingContext,
    val_ds: Any,
    diagnostics_path: Path,
) -> None:
    start = time.monotonic()
    was_training = model.training
    try:
        model.eval()
        with maybe_use_ema(model, ema, ema is not None):
            fused_diff_sweep = compute_fused_diff_sweep(model)
            sanity = _run_lattice_sanity_check(
                step=step,
                model=model,
                sampling_ctx=sampling_ctx,
                val_ds=val_ds,
                count=32,
            )
        payload = {
            "step": int(step),
            "fused_diff_sweep": [
                {
                    "sigma": row["sigma"],
                    "ratio_pct": row["ratio_pct"],
                    "delta_norm": row["delta_norm"],
                }
                for row in fused_diff_sweep
            ],
            **sanity,
        }
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diagnostics_path, "a") as f:
            f.write(json.dumps(payload, allow_nan=False) + "\n")
    except Exception as exc:
        print(f"[diagnostic] warning: diagnostic eval failed at step={step}: {exc}")
    finally:
        elapsed = time.monotonic() - start
        if elapsed > 30.0:
            print(
                f"[diagnostic] warning: diagnostic eval at step={step} took "
                f"{elapsed:.1f}s; continuing training."
            )
        if was_training:
            model.train()


def _build_count_distribution(dataset, nmax: int) -> torch.Tensor:
    counts = torch.zeros(nmax + 1, dtype=torch.float64)
    for i in range(len(dataset)):
        n = int(dataset[i]["num_atoms"])
        if 1 <= n <= nmax:
            counts[n] += 1
    if counts[1:].sum() == 0:
        raise ValueError("No valid num_atoms found for count distribution.")
    probs = counts[1:] / counts[1:].sum()
    return probs

def main() -> None:
    from src.training.config import build_parser, validate_args
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    nmax = args.nmax

    seed_everything(args.seed, deterministic=args.deterministic)
    print(f"[seed] seed={args.seed} deterministic={bool(args.deterministic)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metrics_data_root = (
        str(args.metrics_data_root)
        if args.metrics_data_root is not None
        else str(args.data_root)
    )
    metrics_dataset_name = (
        str(args.metrics_dataset_name)
        if args.metrics_dataset_name is not None
        else str(args.dataset_name)
    )
    args.metrics_data_root = metrics_data_root
    args.metrics_dataset_name = metrics_dataset_name
    property_name = str(args.property_name).strip()
    prop_list = [property_name] if property_name else None
    use_property_conditioning = bool(property_name)

    has_split = ensure_dataset_splits(args.data_root, args.dataset_name)

    ds = MP20Tokens(
        root=args.data_root,
        augment_translate=True,
        split="train" if has_split else "all",
        nmax=nmax,
        prop_list=prop_list,
    )
    val_ds = MP20Tokens(
        root=args.data_root,
        augment_translate=False,
        split="val" if has_split else "all",
        nmax=nmax,
        prop_list=prop_list,
    )
    ref_ds = MP20Tokens(
        root=args.data_root,
        augment_translate=False,
        split="train" if has_split else "all",
        nmax=nmax,
        prop_list=prop_list,
    )
    train_element_dist = compute_dataset_element_distribution(ds)
    train_allowed_mask = compute_allowed_elements(ds)

    print(
        f"Dataset split sizes ({args.dataset_name}, nmax={nmax}):",
        f"train={len(ds)}",
        f"val={len(val_ds)}",
    )
    same_metrics_source = (
        metrics_dataset_name == str(args.dataset_name)
        and Path(metrics_data_root).resolve() == Path(args.data_root).resolve()
    )
    if not args.csp:
        if same_metrics_source:
            print(
                "[eval] Metric references will reuse the training dataset "
                f"({metrics_dataset_name} @ {metrics_data_root})."
            )
        else:
            print(
                "[eval] Metric references will use a separate dataset "
                f"({metrics_dataset_name} @ {metrics_data_root})."
            )

    train_loader_gen = torch.Generator()
    train_loader_gen.manual_seed(int(args.seed))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mp20_tokens,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        generator=train_loader_gen,
        worker_init_fn=seed_dataloader_worker,
    )

    def _infinite_loader(dl):
        while True:
            yield from dl

    steps_per_epoch = max(1, len(loader))
    data_iter = _infinite_loader(loader)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mp20_tokens,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    type_encoding = build_type_encoding(args.type_encoding, vz=VZ)

    model = CrystaliteModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        vz=VZ,
        type_dim=type_encoding.type_dim,
        n_freqs=args.coord_n_freqs,
        coord_embed_mode=args.coord_embed_mode,
        coord_head_mode=args.coord_head_mode,
        coord_rff_dim=args.coord_rff_dim,
        coord_rff_sigma=args.coord_rff_sigma,
        lattice_embed_mode=args.lattice_embed_mode,
        lattice_rff_dim=args.lattice_rff_dim,
        lattice_rff_sigma=args.lattice_rff_sigma,
        lattice_repr=args.lattice_repr,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_distance_bias=args.use_distance_bias,
        use_edge_bias=args.use_edge_bias,
        edge_bias_n_freqs=args.edge_bias_n_freqs,
        edge_bias_hidden_dim=args.edge_bias_hidden_dim,
        edge_bias_n_rbf=args.edge_bias_n_rbf,
        edge_bias_rbf_max=args.edge_bias_rbf_max,
        pbc_radius=args.pbc_radius,
        dist_slope_init=args.dist_slope_init,
        use_noise_gate=args.use_noise_gate,
        gem_per_layer=args.gem_per_layer,
        use_property_conditioning=use_property_conditioning,
        property_encoder_type=args.property_encoder_type,
        property_fusion_mode=args.property_fusion_mode,
        property_num_clusters=args.property_num_clusters,
        property_cluster_min=args.property_cluster_min,
        property_cluster_max=args.property_cluster_max,
        property_mean=args.property_mean,
        property_std=args.property_std,
        property_transform=args.property_transform,
        use_property_film=args.use_property_film,
        film_start_layer=args.film_start_layer,
        film_num_layers=args.film_num_layers,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params}")
    if use_property_conditioning:
        stat_label = "log " if args.property_transform == "log" else ""
        print(
            "[cfg] property conditioning enabled: "
            f"name={property_name} transform={args.property_transform} "
            f"encoder={args.property_encoder_type} "
            f"fusion={args.property_fusion_mode} "
            f"{stat_label}mean={float(args.property_mean):.17g} "
            f"{stat_label}std={float(args.property_std):.17g} "
            f"p_uncond={float(args.p_uncond):.3f} "
            f"conditioned_loss_multiplier={float(args.conditioned_loss_multiplier):.3f}"
        )
        if args.property_encoder_type == "soft_cluster":
            print(
                "[cfg] soft-cluster property encoder: "
                f"num_clusters={int(args.property_num_clusters)} "
                f"cluster_min={float(args.property_cluster_min):.6g} "
                f"cluster_max={float(args.property_cluster_max):.6g}"
            )
        if args.use_property_film:
            film_start = model.film_start_layer
            film_end = model.film_start_layer + model.film_num_layers - 1
            print(
                "[cfg] property FiLM enabled: "
                f"layers={film_start}-{film_end} "
                f"num_layers={model.film_num_layers}"
            )
    if str(args.finetune_from_checkpoint).strip():
        load_finetune_checkpoint(model, args.finetune_from_checkpoint)
    optimizer = build_optimizer_with_property_lr(
        model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        property_embed_lr_multiplier=args.property_embed_lr_multiplier,
        property_delta_lr_multiplier=args.property_delta_lr_multiplier,
    )
    warmup_steps = max(0, args.lr_warmup_steps)
    max_steps = max(1, args.max_steps)
    if warmup_steps >= max_steps:
        warmup_steps = max_steps - 1

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if step >= max_steps:
            return 0.0
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema = None
    if args.ema_decay > 0.0:
        ema = EMA(model, decay=args.ema_decay)
    bf16_dtype = torch.bfloat16 if args.bf16 else None

    run = init_wandb(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        enabled=(not args.no_wandb),
    )
    log_metrics(
        {
            "dataset_splits/train": len(ds),
            "dataset_splits/val": len(val_ds),
            "dataset_splits/num_params": num_params,
            "dataset/name": args.dataset_name,
            "dataset/nmax": nmax,
            "dataset/metrics_reference_name": metrics_dataset_name,
            "dataset/metrics_reference_same_as_train": float(same_metrics_source),
        },
        step=0,
        enabled=(not args.no_wandb),
    )

    weights = {
        "A": 0.0 if args.csp else float(args.loss_weights[0]),
        "F": float(args.loss_weights[1]),
        "Y": float(args.loss_weights[2]),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "diagnostics.jsonl"
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build a checkpoint dict (includes EMA when available) so saves stay consistent.
    def build_ckpt(step_value: int):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_args": vars(args),
            "step": step_value,
            "type_encoding": type_encoding.name,
            "type_dim": type_encoding.type_dim,
        }
        if ema is not None:
            ckpt["ema_state_dict"] = ema.state_dict()
            ckpt["ema_decay"] = args.ema_decay
        return ckpt

    ase_view = None

    enable_dng_metrics = args.sample_count > 0
    novelty_ref_structs = None
    ref_stats = None
    ref_structs = []
    metrics_ref_ds = None
    metrics_val_ds = None
    if not args.csp:
        if same_metrics_source:
            metrics_ref_ds = ref_ds
            metrics_val_ds = val_ds if len(val_ds) > 0 else None
        else:
            metrics_has_split = ensure_dataset_splits(
                metrics_data_root, metrics_dataset_name
            )
            metrics_ref_ds = MP20Tokens(
                root=metrics_data_root,
                augment_translate=False,
                split="train" if metrics_has_split else "all",
                nmax=nmax,
            )
            if metrics_has_split:
                maybe_metrics_val = MP20Tokens(
                    root=metrics_data_root,
                    augment_translate=False,
                    split="val",
                    nmax=nmax,
                )
                if len(maybe_metrics_val) > 0:
                    metrics_val_ds = maybe_metrics_val

        if metrics_ref_ds is None or len(metrics_ref_ds) == 0:
            raise RuntimeError(
                "Metric reference train split is empty under " f"{metrics_data_root}."
            )

        if enable_dng_metrics:
            novelty_ref_structs = dataset_to_structures(metrics_ref_ds)

        if (
            args.sample_frequency > 0
            and metrics_ref_ds is not None
            and len(metrics_ref_ds) > 0
        ):
            ref_items = (
                metrics_ref_ds.items
                if hasattr(metrics_ref_ds, "items")
                else [metrics_ref_ds[i] for i in range(len(metrics_ref_ds))]
            )
            ref_stats = collect_structure_stats(ref_items)

        ref_struct_source = (
            metrics_val_ds
            if (metrics_val_ds is not None and len(metrics_val_ds) > 0)
            else metrics_ref_ds
        )
        if ref_struct_source is not None and len(ref_struct_source) > 0:
            ref_structs = dataset_to_structures(ref_struct_source)
            ref_path = getattr(ref_struct_source, "raw_csv", None)
            ref_split = getattr(ref_struct_source, "split", "unknown")
            if ref_path:
                print(
                    f"[eval] Reference structures will use split='{ref_split}' at {ref_path}"
                )
            else:
                print(
                    f"[eval] Reference structures will use split='{ref_split}' (path unavailable)"
                )
        else:
            print(
                "[eval] No reference structures available; reference-based metrics will be skipped."
            )
    else:
        print("[eval] CSP mode: skipping de novo evaluator/reference setup.")

    thermo_logger = None
    if args.thermo_stability_check:
        if args.thermo_ppd_mp is None or not args.thermo_ppd_mp.exists():
            raise FileNotFoundError(
                "Thermo stability requires --thermo_ppd_mp pointing to a valid PPD pickle."
            )
        thermo_cfg = _ThermoConfig(
            batch_size=max(1, int(args.thermo_stability_batch)),
            relax_steps=int(args.thermo_relax_steps),
            ppd_path=str(args.thermo_ppd_mp),
            device=str(args.thermo_stability_device),
            ehull_method=str(args.thermo_ehull_method),
            mlip=str(args.thermo_mlip),
            nequip_compile_path=str(args.nequip_compile_path),
            nequip_relax_mode=str(args.nequip_relax_mode),
            nequip_optimizer=str(args.nequip_optimizer),
            nequip_cell_filter=str(args.nequip_cell_filter),
            nequip_fmax=float(args.nequip_fmax),
            nequip_max_force_abort=float(args.nequip_max_force_abort),
        )
        thermo_logger = StabilityLogger(gamma_cfg=None, thermo_cfg=thermo_cfg)
    thermo_reference_cache: dict[tuple[str, int, str], dict[str, float]] = {}

    best_ckpt_state = BestCkptState()

    count_probs = None
    if args.atom_count_strategy == "empirical":
        count_probs = _build_count_distribution(ds, nmax=nmax)

    # Optional: report input feature stats on a random subset.
    if args.stat_samples > 0:
        stats = sample_input_stats(
            ds, sample_size=args.stat_samples, type_encoding=type_encoding
        )
        if stats:
            # Print to stdout for quick inspection.
            print("Input stats (sampled):")
            for k, v in stats.items():
                print(f"  {k}: {v.detach().cpu().numpy()}")
            # Flatten and log to wandb if enabled.
            log_payload = {}
            for k, v in stats.items():
                if v.dim() == 1:
                    # Log the full vector as one entry (avoid per-element spam).
                    log_payload[f"data_stats/{k}"] = v.detach().cpu().numpy().tolist()
                else:
                    log_payload[f"data_stats/{k}"] = float(v)
            log_metrics(log_payload, step=0, enabled=(not args.no_wandb))

    sampling_ctx = SamplingContext(
        args=args,
        model=model,
        ema=ema,
        device=device,
        nmax=nmax,
        type_encoding=type_encoding,
        count_probs=count_probs,
        train_allowed_mask=train_allowed_mask,
        train_element_dist=train_element_dist,
        ref_stats=ref_stats,
        ref_structs=ref_structs,
        enable_evaluator_metrics=(not args.csp) and enable_dng_metrics,
        novelty_ref_structs=novelty_ref_structs,
        thermo_logger=thermo_logger,
        thermo_reference_cache=thermo_reference_cache,
        sample_dir=sample_dir,
        ase_view=ase_view,
        wandb_enabled=(not args.no_wandb),
    )

    model.train()
    train_window = {
        "loss_total": 0.0,
        "loss_a": 0.0,
        "loss_f": 0.0,
        "loss_y": 0.0,
        "steps": 0,
        "uncond": 0,
        "samples": 0,
        "cfg_contrast_loss": 0.0,
        "cfg_contrast_active": 0.0,
        "cfg_cond_loss": 0.0,
        "cfg_uncond_loss": 0.0,
        "cfg_steps": 0,
    }
    progress = tqdm(range(1, args.max_steps + 1), desc="train", dynamic_ncols=True)
    for step in progress:
        batch = next(data_iter)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        sigma = sample_sigma(
            bsz=batch["A0"].shape[0],
            device=device,
            P_mean=args.edm_P_mean,
            P_std=args.edm_P_std,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        pad_mask = batch["pad_mask"].bool()
        real_mask = ~pad_mask

        type_clean = type_encoding.encode_from_A0(batch["A0"], pad_mask)
        frac_clean = batch["F1"]
        frac_clean_c = frac_clean - 0.5  # centered coordinates for Euclidean EDM
        lat_clean = y1_to_lattice_latent(batch["Y1"], args.lattice_repr)

        g_type = torch.randn_like(type_clean)
        g_frac = torch.randn_like(frac_clean_c)
        g_lat = torch.randn_like(lat_clean)

        if args.csp:
            type_noisy = type_clean
        else:
            type_noisy = type_clean + sigma[:, None, None] * g_type
        frac_noisy = (
            frac_clean_c + sigma[:, None, None] * g_frac
        )  # unwrapped Euclidean diffusion
        frac_noisy = torch.where(
            real_mask[..., None], frac_noisy, torch.zeros_like(frac_noisy)
        )
        lat_noisy = lat_clean + sigma[:, None] * g_lat
        property_values = None
        force_uncond_mask = None
        sample_loss_weights = None
        if use_property_conditioning:
            if property_name not in batch:
                raise KeyError(
                    f"Missing property '{property_name}' in batch. "
                    "Check MP20Tokens(prop_list=...) and collate_mp20_tokens."
                )
            property_values = batch[property_name].to(device=device, dtype=torch.float32)
            force_uncond_mask = (
                torch.rand(property_values.shape[0], device=device) < float(args.p_uncond)
            )
            if float(args.conditioned_loss_multiplier) != 1.0:
                sample_loss_weights = torch.where(
                    force_uncond_mask,
                    torch.ones_like(property_values),
                    torch.full_like(property_values, float(args.conditioned_loss_multiplier)),
                )

        cfg_contrast_weight = float(args.cfg_contrast_weight)
        clean_targets = {"type": type_clean, "frac_c": frac_clean_c, "lat": lat_clean}
        if (
            cfg_contrast_weight > 0.0
            and use_property_conditioning
            and property_values is not None
            and force_uncond_mask is not None
        ):
            force_cond_mask = torch.zeros_like(force_uncond_mask)
            force_null_mask = torch.ones_like(force_uncond_mask)
            denoised_cond = denoise_edm(
                model=model,
                type_noisy=type_noisy,
                frac_noisy=frac_noisy,
                lat_noisy=lat_noisy,
                pad_mask=pad_mask,
                sigma=sigma,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                autocast_dtype=bf16_dtype,
                skip_type_scaling=args.csp,
                property_values=property_values,
                force_uncond_mask=force_cond_mask,
            )
            denoised_null = denoise_edm(
                model=model,
                type_noisy=type_noisy,
                frac_noisy=frac_noisy,
                lat_noisy=lat_noisy,
                pad_mask=pad_mask,
                sigma=sigma,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                autocast_dtype=bf16_dtype,
                skip_type_scaling=args.csp,
                property_values=property_values,
                force_uncond_mask=force_null_mask,
            )
            mix_mask_atom = force_uncond_mask[:, None, None]
            mix_mask_lat = force_uncond_mask[:, None]
            denoised = {
                "type": torch.where(
                    mix_mask_atom, denoised_null["type"], denoised_cond["type"]
                ),
                "frac": torch.where(
                    mix_mask_atom, denoised_null["frac"], denoised_cond["frac"]
                ),
                "lat": torch.where(
                    mix_mask_lat, denoised_null["lat"], denoised_cond["lat"]
                ),
                "raw": denoised_cond["raw"],
            }
            losses = compute_edm_loss(
                denoised=denoised,
                clean=clean_targets,
                frac_noisy=frac_noisy,
                sigma=sigma,
                pad_mask=pad_mask,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                loss_weights=weights,
                coord_loss_mode=args.coord_loss_mode,
                lattice_repr=args.lattice_repr,
                sample_weights=sample_loss_weights,
            )
            cond_per_sample = compute_edm_loss_per_sample(
                denoised=denoised_cond,
                clean=clean_targets,
                frac_noisy=frac_noisy,
                sigma=sigma,
                pad_mask=pad_mask,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                loss_weights=weights,
                coord_loss_mode=args.coord_loss_mode,
                lattice_repr=args.lattice_repr,
            )
            null_per_sample = compute_edm_loss_per_sample(
                denoised=denoised_null,
                clean=clean_targets,
                frac_noisy=frac_noisy,
                sigma=sigma,
                pad_mask=pad_mask,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                loss_weights=weights,
                coord_loss_mode=args.coord_loss_mode,
                lattice_repr=args.lattice_repr,
            )
            contrast_values = torch.relu(
                float(args.cfg_contrast_margin)
                + cond_per_sample["loss_total"]
                - null_per_sample["loss_total"]
            )
            cfg_contrast_loss = contrast_values.mean()
            base_total = losses["loss_total"]
            losses["loss_total"] = base_total + cfg_contrast_weight * cfg_contrast_loss
            losses["loss_base_total"] = base_total.detach()
            losses["loss_cfg_contrast"] = cfg_contrast_loss
            losses["loss_cfg_cond"] = cond_per_sample["loss_total"].mean().detach()
            losses["loss_cfg_uncond"] = null_per_sample["loss_total"].mean().detach()
            losses["loss_cfg_active"] = (contrast_values > 0).float().mean().detach()
        else:
            denoised = denoise_edm(
                model=model,
                type_noisy=type_noisy,
                frac_noisy=frac_noisy,
                lat_noisy=lat_noisy,
                pad_mask=pad_mask,
                sigma=sigma,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                autocast_dtype=bf16_dtype,
                skip_type_scaling=args.csp,
                property_values=property_values,
                force_uncond_mask=force_uncond_mask,
            )

            losses = compute_edm_loss(
                denoised=denoised,
                clean=clean_targets,
                frac_noisy=frac_noisy,
                sigma=sigma,
                pad_mask=pad_mask,
                sigma_data_type=args.sigma_data_type,
                sigma_data_coord=args.sigma_data_coord,
                sigma_data_lat=args.sigma_data_lattice,
                loss_weights=weights,
                coord_loss_mode=args.coord_loss_mode,
                lattice_repr=args.lattice_repr,
                sample_weights=sample_loss_weights,
            )
        if not torch.isfinite(losses["loss_total"]):
            raise RuntimeError(f"Non-finite loss_total at step {step}.")
        train_window["loss_total"] += float(losses["loss_total"].detach().item())
        train_window["loss_a"] += float(losses["loss_a"].detach().item())
        train_window["loss_f"] += float(losses["loss_f"].detach().item())
        train_window["loss_y"] += float(losses["loss_y"].detach().item())
        train_window["steps"] += 1
        if force_uncond_mask is not None:
            train_window["uncond"] += int(force_uncond_mask.sum().detach().item())
            train_window["samples"] += int(force_uncond_mask.numel())
        if "loss_cfg_contrast" in losses:
            train_window["cfg_contrast_loss"] += float(
                losses["loss_cfg_contrast"].detach().item()
            )
            train_window["cfg_contrast_active"] += float(
                losses["loss_cfg_active"].detach().item()
            )
            train_window["cfg_cond_loss"] += float(
                losses["loss_cfg_cond"].detach().item()
            )
            train_window["cfg_uncond_loss"] += float(
                losses["loss_cfg_uncond"].detach().item()
            )
            train_window["cfg_steps"] += 1
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        if step % args.log_every == 0:
            metrics = {
                "loss/total": float(losses["loss_total"].item()),
                "loss/type": float(losses["loss_a"].item() * weights["A"]),
                "loss/coord": float(losses["loss_f"].item() * weights["F"]),
                "loss/lattice": float(losses["loss_y"].item() * weights["Y"]),
                "lr": optimizer.param_groups[0]["lr"],
                "sigma/mean": float(sigma.mean().item()),
                "sigma/std": float(sigma.std().item()),
            }
            if "loss_cfg_contrast" in losses:
                metrics.update(
                    {
                        "loss/base_total": float(losses["loss_base_total"].item()),
                        "cfg_contrast/loss": float(
                            losses["loss_cfg_contrast"].item()
                        ),
                        "cfg_contrast/active_fraction": float(
                            losses["loss_cfg_active"].item()
                        ),
                        "cfg_contrast/cond_loss": float(losses["loss_cfg_cond"].item()),
                        "cfg_contrast/uncond_loss": float(
                            losses["loss_cfg_uncond"].item()
                        ),
                    }
                )
            if real_mask.any():
                type_pred = type_encoding.decode_logits_to_A0(
                    type_logits=denoised["type"], pad_mask=pad_mask
                )
                target_zero = batch["A0"]
                correct = (type_pred == target_zero) & real_mask
                metrics["stats/type_acc"] = float(
                    correct.float().sum().item() / real_mask.sum().item()
                )
                frac_delta = wrap_frac(denoised["frac"] - frac_clean_c)
                mask_exp = real_mask[..., None].float()
                metrics["stats/coord_l2"] = float(
                    ((frac_delta**2) * mask_exp).sum().item()
                    / mask_exp.sum().clamp_min(1.0).item()
                )
            else:
                metrics["stats/type_acc"] = 0.0
                metrics["stats/coord_l2"] = 0.0
            metrics["stats/lattice_l2"] = float(
                ((denoised["lat"] - lat_clean) ** 2).mean().item()
            )
            log_metrics(metrics, step=step, enabled=(not args.no_wandb))
            progress.set_postfix(loss=f"{metrics['loss/total']:.4f}")

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            ckpt_path = (
                ckpt_dir / "step_latest.pt"
                if args.ckpt_latest_only
                else ckpt_dir / f"step_{step:07d}.pt"
            )
            torch.save(build_ckpt(step), ckpt_path)

        if (
            args.diagnostic_eval_every > 0
            and step % args.diagnostic_eval_every == 0
            and _is_rank0()
        ):
            _run_training_diagnostics(
                step=step,
                model=model,
                ema=ema,
                sampling_ctx=sampling_ctx,
                val_ds=val_ds,
                diagnostics_path=diagnostics_path,
            )

        if args.val_every > 0 and step % args.val_every == 0 and has_split:
            model.eval()
            val_losses = {
                "loss_total": 0.0,
                "loss_a": 0.0,
                "loss_f": 0.0,
                "loss_y": 0.0,
            }
            num = 0
            with torch.no_grad():
                for v_idx, v_batch in enumerate(val_loader):
                    if v_idx >= args.val_batches:
                        break
                    v_batch = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in v_batch.items()
                    }
                    sigma_v = sample_sigma(
                        bsz=v_batch["A0"].shape[0],
                        device=device,
                        P_mean=args.edm_P_mean,
                        P_std=args.edm_P_std,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                    )
                    pad_v = v_batch["pad_mask"].bool()
                    real_v = ~pad_v
                    type_clean_v = type_encoding.encode_from_A0(v_batch["A0"], pad_v)
                    frac_clean_v = v_batch["F1"]
                    frac_clean_v_c = frac_clean_v - 0.5
                    lat_clean_v = y1_to_lattice_latent(v_batch["Y1"], args.lattice_repr)
                    if args.csp:
                        type_noisy_v = type_clean_v
                    else:
                        type_noisy_v = type_clean_v + sigma_v[
                            :, None, None
                        ] * torch.randn_like(type_clean_v)
                    frac_noisy_v = frac_clean_v_c + sigma_v[
                        :, None, None
                    ] * torch.randn_like(frac_clean_v_c)
                    frac_noisy_v = torch.where(
                        real_v[..., None], frac_noisy_v, torch.zeros_like(frac_noisy_v)
                    )
                    lat_noisy_v = lat_clean_v + sigma_v[:, None] * torch.randn_like(
                        lat_clean_v
                    )
                    property_values_v = None
                    force_uncond_mask_v = None
                    if use_property_conditioning:
                        if property_name not in v_batch:
                            raise KeyError(
                                f"Missing property '{property_name}' in validation batch."
                            )
                        property_values_v = v_batch[property_name].to(
                            device=device, dtype=torch.float32
                        )
                        force_uncond_mask_v = torch.zeros(
                            property_values_v.shape[0],
                            device=device,
                            dtype=torch.bool,
                        )

                    denoised_v = denoise_edm(
                        model=model,
                        type_noisy=type_noisy_v,
                        frac_noisy=frac_noisy_v,
                        lat_noisy=lat_noisy_v,
                        pad_mask=pad_v,
                        sigma=sigma_v,
                        sigma_data_type=args.sigma_data_type,
                        sigma_data_coord=args.sigma_data_coord,
                        sigma_data_lat=args.sigma_data_lattice,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        autocast_dtype=bf16_dtype,
                        skip_type_scaling=args.csp,
                        property_values=property_values_v,
                        force_uncond_mask=force_uncond_mask_v,
                    )
                    v_losses = compute_edm_loss(
                        denoised=denoised_v,
                        clean={
                            "type": type_clean_v,
                            "frac_c": frac_clean_v_c,
                            "lat": lat_clean_v,
                        },
                        frac_noisy=frac_noisy_v,
                        sigma=sigma_v,
                        pad_mask=pad_v,
                        sigma_data_type=args.sigma_data_type,
                        sigma_data_coord=args.sigma_data_coord,
                        sigma_data_lat=args.sigma_data_lattice,
                        loss_weights=weights,
                        coord_loss_mode=args.coord_loss_mode,
                        lattice_repr=args.lattice_repr,
                    )
                    for k in val_losses:
                        val_losses[k] += float(v_losses[k].item())
                    num += 1
            if num > 0:
                for k in val_losses:
                    val_losses[k] /= num
            if train_window["steps"] > 0:
                train_steps = max(1, int(train_window["steps"]))
                train_loss_total = train_window["loss_total"] / train_steps
                train_loss_a = train_window["loss_a"] / train_steps
                train_loss_f = train_window["loss_f"] / train_steps
                train_loss_y = train_window["loss_y"] / train_steps
                if train_window["samples"] > 0:
                    uncond_count = int(train_window["uncond"])
                    sample_count = int(train_window["samples"])
                    uncond_rate = uncond_count / sample_count
                else:
                    uncond_count = 0
                    sample_count = 0
                    uncond_rate = 0.0
                epoch_parts = [
                    f"step={step}",
                    f"epoch={math.ceil(step / steps_per_epoch)}",
                    f"train_loss_total={train_loss_total:.6f}",
                    f"train_loss_type={train_loss_a:.6f}",
                    f"train_loss_coord={train_loss_f:.6f}",
                    f"train_loss_lattice={train_loss_y:.6f}",
                    f"val_loss_total={val_losses['loss_total']:.6f}",
                    f"val_loss_type={val_losses['loss_a']:.6f}",
                    f"val_loss_coord={val_losses['loss_f']:.6f}",
                    f"val_loss_lattice={val_losses['loss_y']:.6f}",
                    f"p_uncond={float(args.p_uncond):.3f}",
                    f"uncond={uncond_count}/{sample_count}",
                    f"uncond_rate={uncond_rate:.4f}",
                ]
                if train_window["cfg_steps"] > 0:
                    cfg_steps = max(1, int(train_window["cfg_steps"]))
                    epoch_parts.extend(
                        [
                            "cfg_contrast_weight="
                            f"{float(args.cfg_contrast_weight):.4f}",
                            "cfg_contrast_margin="
                            f"{float(args.cfg_contrast_margin):.4f}",
                            "cfg_contrast_loss="
                            f"{train_window['cfg_contrast_loss'] / cfg_steps:.6f}",
                            "cfg_contrast_active="
                            f"{train_window['cfg_contrast_active'] / cfg_steps:.4f}",
                            "cfg_cond_loss="
                            f"{train_window['cfg_cond_loss'] / cfg_steps:.6f}",
                            "cfg_uncond_loss="
                            f"{train_window['cfg_uncond_loss'] / cfg_steps:.6f}",
                        ]
                    )
                print("[epoch] " + " ".join(epoch_parts), flush=True)
                for key in train_window:
                    train_window[key] = 0.0
            log_metrics(
                {
                    "val/loss_total": val_losses["loss_total"],
                    "val/loss_type": val_losses["loss_a"],
                    "val/loss_coord": val_losses["loss_f"],
                    "val/loss_lattice": val_losses["loss_y"],
                },
                step=step,
                enabled=(not args.no_wandb),
            )
            progress.set_postfix(val_loss=f"{val_losses['loss_total']:.4f}")

            # Best-checkpoint fallback: track best val loss before sampling metrics are available.
            _fb = build_val_fallback_candidate(
                step=step,
                epoch=math.ceil(step / steps_per_epoch),
                mode="csp" if args.csp else "dng",
                val_loss=val_losses["loss_total"],
            )
            if _fb is not None:
                maybe_update_best_ckpt(
                    state=best_ckpt_state,
                    candidate=_fb,
                    maximize=False,
                    ckpt_dir=ckpt_dir,
                    build_ckpt_fn=build_ckpt,
                    enabled=args.best_ckpt,
                )

            model.train()

        do_sample = args.sample_frequency > 0 and (step % args.sample_frequency == 0)
        if do_sample:
            base_seed = args.sample_seed + step
            was_training = model.training

            # Accumulators for best-checkpoint metric collection.
            _dng_payloads: dict[str, dict[str, float]] = {}
            _csp_payloads: dict[str, list[dict[str, Any]]] = {}

            def _run_one_sampling(
                tag: str,
                use_ema: bool,
                metrics_count: int,
                csp_source_ds=None,
                csp_source_label: str = "val",
            ) -> None:
                request = SamplingRequest(
                    tag=tag,
                    step=step,
                    base_seed=base_seed,
                    use_ema=use_ema,
                    metrics_count=metrics_count,
                    csp_source_ds=csp_source_ds,
                    csp_source_label=csp_source_label,
                )
                with maybe_use_ema(model, ema, use_ema):
                    batch = generate_sampling_batch(request, sampling_ctx)
                if args.csp:
                    outcome = evaluate_csp_sampling_batch(batch, request, sampling_ctx)
                else:
                    outcome = evaluate_dng_sampling_batch(batch, request, sampling_ctx)
                save_sampling_artifacts(batch, request, sampling_ctx)
                if outcome.dng_payload:
                    _dng_payloads.setdefault(tag, {}).update(outcome.dng_payload)
                for payload in outcome.csp_payloads:
                    _csp_payloads.setdefault(tag, []).append(payload)

            runs, ema_missing = _build_sampling_runs(
                do_sample=do_sample,
                sample_mode=args.sample_mode,
                ema_use_for_sampling=args.ema_use_for_sampling,
                ema_available=(ema is not None),
                sample_count=args.sample_count,
            )

            if ema_missing:
                print(
                    "[sample] EMA requested via --sample_mode but unavailable; using regular weights instead."
                )

            for tag, use_ema, mcount in runs:
                if args.csp:
                    _run_one_sampling(tag, use_ema, mcount, csp_source_ds=val_ds, csp_source_label="val")
                else:
                    _run_one_sampling(tag, use_ema, mcount)

            # Best-checkpoint: select primary candidate from sampling metrics.
            if args.best_ckpt and (_dng_payloads or _csp_payloads):
                _primary = select_primary_candidate_from_sampling(
                    is_csp=args.csp,
                    step=step,
                    epoch=math.ceil(step / steps_per_epoch),
                    dng_payloads=_dng_payloads if not args.csp else None,
                    csp_payloads=_csp_payloads if args.csp else None,
                    best_ckpt_selector=args.best_ckpt_selector,
                )
                if _primary is not None:
                    maybe_update_best_ckpt(
                        state=best_ckpt_state,
                        candidate=_primary,
                        maximize=True,
                        ckpt_dir=ckpt_dir,
                        build_ckpt_fn=build_ckpt,
                        enabled=True,
                    )

            if was_training:
                model.train()

    # Reuse the epoch_latest file for the final save to avoid writing twice.
    final_epoch_path = ckpt_dir / "epoch_latest.pt"
    ckpt = build_ckpt(args.max_steps)
    ckpt["epoch"] = math.ceil(args.max_steps / steps_per_epoch)
    torch.save(ckpt, final_epoch_path)

    # Provide a compatibility link at checkpoints/final.pt without duplicating data.
    final_path = ckpt_dir / "final.pt"
    try:
        if final_path.exists() or final_path.is_symlink():
            final_path.unlink()
        os.link(final_epoch_path, final_path)
    except OSError:
        # Fallback to a symlink; if that fails, we still have epoch_latest.pt.
        try:
            if final_path.exists() or final_path.is_symlink():
                final_path.unlink()
            os.symlink(final_epoch_path.name, final_path)
        except OSError:
            pass

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
