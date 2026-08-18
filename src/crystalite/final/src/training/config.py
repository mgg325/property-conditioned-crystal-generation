from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.utils.checkpoint import BEST_CKPT_SELECTOR_CHOICES
from src.utils.constants import DATASET_NMAX_DEFAULTS
from src.data.mp20_tokens import NMAX as DEFAULT_NMAX
from src.crystalite.sampler import resolve_nonnegative_scalar, resolve_aa_rho_pair


def _normalize_topk_list(values: list[int]) -> list[int]:
    if not values:
        return []
    out = sorted({int(v) for v in values})
    if any(v <= 0 for v in out):
        raise ValueError("csp_precise_topk_list values must be positive integers.")
    return out


def _compute_topk_target_count(total_targets: int, requested_targets: int) -> int:
    total_targets = max(0, int(total_targets))
    requested_targets = int(requested_targets)
    if requested_targets <= 0:
        return 0
    return min(total_targets, requested_targets)


class _StoreWithSeenAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        seen = getattr(namespace, "_explicit_args", None)
        if seen is None:
            seen = set()
            setattr(namespace, "_explicit_args", seen)
        seen.add(self.dest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MP20 EDM training with Subatomic transformer."
    )
    # Data
    parser.add_argument("--data_root", type=str, default="data/mp20")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mp20",
        choices=["mp20", "mpts_52", "perov_5", "alex_mp20", "custom"],
        help="Dataset preset used for defaults/logging.",
    )
    parser.add_argument(
        "--metrics_data_root",
        type=str,
        default=None,
        help=(
            "Optional dataset root used only for evaluation references "
            "(novelty/UN/SUN/MSUN/Wasserstein/reference metrics). "
            "Default: use --data_root."
        ),
    )
    parser.add_argument(
        "--metrics_dataset_name",
        type=str,
        default=None,
        choices=["mp20", "mpts_52", "perov_5", "alex_mp20", "custom"],
        help=(
            "Optional dataset preset used only for evaluation references. "
            "Default: use --dataset_name."
        ),
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=None,
        help="Max atoms per structure. Defaults by dataset_name (mp20=20, mpts_52=52, perov_5=5, alex_mp20=20).",
    )
    parser.add_argument(
        "--property_name",
        type=str,
        default="",
        help="Optional scalar property column to condition on, e.g. dft_dielectric_scalar.",
    )
    parser.add_argument(
        "--property_mean",
        type=float,
        default=0.0,
        help="Train-split mean used to normalize --property_name.",
    )
    parser.add_argument(
        "--property_std",
        type=float,
        default=1.0,
        help="Train-split std used to normalize --property_name.",
    )
    parser.add_argument(
        "--property_transform",
        type=str,
        default="identity",
        choices=["identity", "raw", "none", "standardize", "log"],
        help=(
            "Scalar property transform applied before normalization. "
            "'raw', 'none', and 'standardize' are aliases for 'identity'. "
            "For 'log', --property_mean/--property_std must be log-space stats."
        ),
    )
    parser.add_argument(
        "--property_encoder_type",
        type=str,
        default="scalar_mlp",
        choices=["scalar_mlp", "soft_cluster"],
        help="Scalar property encoder. Default scalar_mlp preserves the baseline.",
    )
    parser.add_argument(
        "--property_fusion_mode",
        type=str,
        default="add",
        choices=["add", "gated_add", "concat_mlp", "residual_delta", "none"],
        help=(
            "How the property embedding is fused with the sigma/time AdaLN condition. "
            "'none' = no additive/t_emb fusion (FiLM-only)."
        ),
    )
    parser.add_argument(
        "--property_num_clusters",
        type=int,
        default=32,
        help="Number of soft clusters for --property_encoder_type soft_cluster.",
    )
    parser.add_argument(
        "--property_cluster_min",
        type=float,
        default=-2.5,
        help="Initial minimum normalized scalar cluster center.",
    )
    parser.add_argument(
        "--property_cluster_max",
        type=float,
        default=4.0,
        help="Initial maximum normalized scalar cluster center.",
    )
    parser.add_argument(
        "--use_property_film",
        action="store_true",
        help=(
            "Add scalar-property FiLM modulation after upper Transformer blocks. "
            "Requires scalar property conditioning data at training/sampling time."
        ),
    )
    parser.add_argument(
        "--film_start_layer",
        type=int,
        default=-1,
        help=(
            "First Transformer layer index to receive property FiLM. "
            "Default -1 selects the upper half, n_layers // 2."
        ),
    )
    parser.add_argument(
        "--film_num_layers",
        type=int,
        default=-1,
        help=(
            "Number of suffix Transformer layers with property FiLM. "
            "Default -1 covers film_start_layer through the final layer."
        ),
    )
    parser.add_argument(
        "--p_uncond",
        type=float,
        default=0.2,
        help="Probability of replacing property conditioning with the learned null embedding.",
    )
    parser.add_argument(
        "--conditioned_loss_multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiplier for EDM loss contributions from property-conditioned "
            "training samples. CFG-dropped unconditional samples keep weight 1.0. "
            "Default 1.0 preserves baseline behavior."
        ),
    )
    parser.add_argument(
        "--cfg_contrast_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for an optional paired CFG contrast hinge loss. "
            "When > 0, training evaluates true-conditioned and null-conditioned "
            "denoising on the same noisy inputs and encourages the true-conditioned "
            "branch to beat the null branch by --cfg_contrast_margin. "
            "Default 0 preserves baseline behavior."
        ),
    )
    parser.add_argument(
        "--cfg_contrast_margin",
        type=float,
        default=0.25,
        help="Per-sample denoising-loss margin for --cfg_contrast_weight.",
    )
    parser.add_argument(
        "--sample_property_value",
        type=float,
        default=None,
        help=(
            "Optional scalar target used for periodic in-training sampling when "
            "--property_name is set."
        ),
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        default="",
        help=(
            "Initialize model weights from a checkpoint for a new fine-tuning run. "
            "Does not restore optimizer, scheduler, EMA, or step."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Global training seed for python/numpy/torch and dataloader workers.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic training ops where supported (can be slower).",
    )
    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument(
        "--type_encoding",
        type=str,
        default="atomic_number",
        choices=[
            "atomic_number",
            "subatomic_tokenizer_raw",
            "subatomic_tokenizer_pca",
            "subatomic_tokenizer_pca_8",
            "subatomic_tokenizer_pca_16",
            "subatomic_tokenizer_pca_24",
        ],
        help=(
            "Atom-type encoding mode for EDM: direct atomic-number channels, "
            "subatomic_tokenizer_raw, or the PCA-compressed subatomic tokenizer "
            "family. The shorthand subatomic_tokenizer_pca resolves to "
            "subatomic_tokenizer_pca_24."
        ),
    )
    parser.add_argument(
        "--use_distance_bias",
        action="store_true",
        help="Enable distance-based attention bias in the Transformer trunk.",
    )
    parser.add_argument(
        "--use_edge_bias",
        action="store_true",
        help=(
            "Enable learned edge-conditioned attention bias from minimum-image "
            "fractional displacements + lattice metric features."
        ),
    )
    parser.add_argument(
        "--edge_bias_n_freqs",
        type=int,
        default=8,
        help="Number of sinusoidal frequencies per coordinate axis for edge-bias features.",
    )
    parser.add_argument(
        "--edge_bias_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the edge-bias MLP.",
    )
    parser.add_argument(
        "--edge_bias_n_rbf",
        type=int,
        default=16,
        help="Number of RBF channels used for normalized-distance edge-bias features.",
    )
    parser.add_argument(
        "--edge_bias_rbf_max",
        type=float,
        default=2.0,
        help="Maximum normalized distance center for edge-bias RBF features.",
    )
    parser.add_argument(
        "--pbc_radius",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "Minimum-image translation radius per lattice axis when computing geometry bias "
            "(r=1 uses 27 offsets; r=2 uses 125)."
        ),
    )
    parser.add_argument(
        "--dist_slope_init",
        type=float,
        default=-1.0,
        help="Initial distance-prior slope for GEM (implemented via monotone -softplus).",
    )
    parser.add_argument(
        "--use_noise_gate",
        dest="use_noise_gate",
        action="store_true",
        help="Enable sigma-dependent gating for GEM bias.",
    )
    parser.add_argument(
        "--no_noise_gate",
        dest="use_noise_gate",
        action="store_false",
        help="Disable sigma-dependent gating for GEM bias.",
    )
    parser.set_defaults(use_noise_gate=True)
    parser.add_argument(
        "--gem_per_layer",
        action="store_true",
        help="Use a separate GEM module per transformer block (more expressive, higher compute).",
    )
    parser.add_argument(
        "--coord_n_freqs",
        "--n_freqs",
        dest="coord_n_freqs",
        type=int,
        default=32,
        help="Number of deterministic coordinate Fourier frequencies when --coord_embed_mode=fourier.",
    )
    parser.add_argument(
        "--coord_embed_mode",
        type=str,
        default="fourier",
        choices=["rff", "fourier"],
        help="Coordinate embedding type: random Fourier features (rff) or deterministic Fourier.",
    )
    parser.add_argument(
        "--coord_head_mode",
        type=str,
        default="direct",
        choices=["direct", "relative"],
        help=(
            "Coordinate prediction head: 'direct' uses a linear per-atom head; "
            "'relative' uses attention-weighted minimum-image displacements."
        ),
    )
    parser.add_argument(
        "--coord_rff_dim",
        type=int,
        default=256,
        help="Dimension of random Fourier feature projection (defaults to n_freqs when omitted).",
    )
    parser.add_argument(
        "--coord_rff_sigma",
        type=float,
        default=1.0,
        help="Stddev for RFF projection matrix.",
    )
    parser.add_argument(
        "--lattice_embed_mode",
        type=str,
        default="rff",
        choices=["mlp", "rff"],
        help="Lattice embedding type: legacy MLP or random Fourier features.",
    )
    parser.add_argument(
        "--lattice_repr",
        type=str,
        default="y1",
        choices=["y1", "ltri"],
        help=(
            "Internal lattice latent representation. "
            "'y1' uses [log lengths, cos angles]; "
            "'ltri' uses lower-triangular matrix params with positive diagonals."
        ),
    )
    parser.add_argument(
        "--lattice_rff_dim",
        type=int,
        default=256,
        help="Dimension of random Fourier feature projection for lattice embedding.",
    )
    parser.add_argument(
        "--lattice_rff_sigma",
        type=float,
        default=5.0,
        help="Stddev for lattice RFF projection matrix.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    # Optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--property_lr_multiplier",
        type=float,
        default=None,
        action=_StoreWithSeenAction,
        help=(
            "Deprecated compatibility alias for both "
            "--property_embed_lr_multiplier and --property_delta_lr_multiplier."
        ),
    )
    parser.add_argument(
        "--property_embed_lr_multiplier",
        type=float,
        default=1.0,
        action=_StoreWithSeenAction,
        help="LR multiplier for property_embed.* parameters (cluster centers/widths/encoder MLP).",
    )
    parser.add_argument(
        "--property_delta_lr_multiplier",
        type=float,
        default=1.0,
        action=_StoreWithSeenAction,
        help=(
            "LR multiplier for property_delta.* parameters (residual injection into t_emb). "
            "Keep this close to 1.0 to preserve the zero-init stabilization."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    # Logging / eval
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_batches", type=int, default=10)
    parser.add_argument(
        "--diagnostic_eval_every",
        type=int,
        default=200,
        help="Run fused_diff sweep + EMA lattice sanity check every N steps. Set to 0 to disable.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="subatomic-edm")
    parser.add_argument("--wandb_name", type=str, default="mp20_edm")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/train_crystalite")
    # Sampling
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=50000,
        help="Step interval for sampling/metrics (0 to disable).",
    )
    parser.add_argument("--sample_vis_count", type=int, default=10)
    parser.add_argument(
        "--sample_count",
        type=int,
        default=10000,
        help="Number of samples used for metrics/eval.",
    )
    parser.add_argument("--sample_num_steps", type=int, default=50)
    parser.add_argument("--sample_seed", type=int, default=123)
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="ema",
        choices=["regular", "ema", "both"],
        help="Which weights to use for sampling/precise eval: ema (default), regular, or both.",
    )
    parser.add_argument(
        "--sample_chunk_size",
        type=int,
        default=0,
        help="Optional chunk size for sampling to reduce peak GPU memory. 0 disables chunking.",
    )
    parser.add_argument(
        "--sample_compute_novelty",
        action="store_true",
        help=(
            "Deprecated compatibility flag. ADiT novelty is no longer part of the "
            "runtime metric path."
        ),
    )
    parser.add_argument(
        "--sample_novelty_limit",
        type=int,
        default=0,
        help=(
            "Max structures used for the novelty reference train split "
            "(0 or negative = full reference train set)."
        ),
    )
    parser.add_argument(
        "--thermo_stability_check",
        action="store_true",
        help="Compute MLIP + hull thermo stability metrics during sampling.",
    )
    parser.add_argument(
        "--thermo_ppd_mp",
        type=Path,
        default=Path("data/mp20/hull/2023-02-07-ppd-mp.pkl"),
        help="Patched MP pickle (PatchedPhaseDiagram) for thermo stability.",
    )
    parser.add_argument(
        "--thermo_ehull_method",
        type=str,
        default="uncorrected",
        choices=["uncorrected", "mp2020_like"],
        help=(
            "How to compute e_above_hull for thermo stability. "
            "'uncorrected' uses (E_total/atom - hull_energy_per_atom), "
            "'mp2020_like' applies MP2020 compatibility corrections first."
        ),
    )
    parser.add_argument("--thermo_stability_device", type=str, default="cuda")
    parser.add_argument(
        "--thermo_mlip",
        type=str,
        default="chgnet",
        choices=["chgnet", "nequip"],
        help="MLIP backend for thermo stability checks.",
    )
    parser.add_argument(
        "--nequip_compile_path",
        type=str,
        default="data/mlip/nequip/*.nequip.pt2",
        help=(
            "Path, directory, or glob for compiled NequIP AOT artifacts "
            "(`.nequip.pt2`). Used when --thermo_mlip nequip."
        ),
    )
    parser.add_argument(
        "--nequip_relax_mode",
        type=str,
        default="sequential",
        choices=["sequential", "batch"],
        help=(
            "NequIP relax mode for thermo checks. "
            "'batch' uses the TorchSim batched relaxer."
        ),
    )
    parser.add_argument(
        "--nequip_optimizer",
        type=str,
        default="FIRE",
        choices=["FIRE", "LBFGS", "BFGS", "BFGSLineSearch", "LBFGSLineSearch"],
        help="ASE optimizer for NequIP relaxations.",
    )
    parser.add_argument(
        "--nequip_cell_filter",
        type=str,
        default="none",
        choices=["none", "frechet", "exp"],
        help="Optional ASE cell filter for NequIP relaxations.",
    )
    parser.add_argument(
        "--nequip_fmax",
        type=float,
        default=0.01,
        help="Force convergence criterion (eV/A) for NequIP relaxations.",
    )
    parser.add_argument(
        "--nequip_max_force_abort",
        type=float,
        default=1e6,
        help="Abort NequIP relaxation if max force exceeds this threshold.",
    )
    parser.add_argument("--thermo_stability_batch", type=int, default=32)
    parser.add_argument("--thermo_relax_steps", type=int, default=200)
    parser.add_argument(
        "--thermo_stability_count",
        type=int,
        default=0,
        help="Number of samples for thermo stability (0 uses metrics_count). Ignored when --sun_k > 0.",
    )
    parser.add_argument(
        "--no_thermo_before_steps",
        type=int,
        default=90000,
        help="Skip thermo stability / SUN/MSUN logging until this step.",
    )
    parser.add_argument(
        "--sun_k",
        type=int,
        default=2048,
        help="Number of UN structures to relax for SUN/MSUN metrics (0 disables).",
    )
    parser.add_argument(
        "--csp_precise_topk_list",
        type=int,
        nargs="+",
        default=[],
        help="In CSP mode, evaluate precise runs with best-of-k metrics for each k in this list (e.g. --csp_precise_topk_list 1 20).",
    )
    parser.add_argument(
        "--csp_precise_topk_samples",
        type=int,
        default=0,
        help="In CSP precise mode, evaluate top-k over exactly this many targets (0 disables top-k eval).",
    )
    parser.add_argument(
        "--atom_count_strategy",
        type=str,
        default="empirical",
        choices=["fixed", "empirical"],
    )
    # EMA / ckpt
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument(
        "--ema_use_for_sampling",
        action="store_true",
        help="Deprecated: prefer --sample_mode to control EMA usage.",
    )
    parser.add_argument("--ckpt_every", type=int, default=0)
    parser.add_argument(
        "--ckpt_latest_only",
        action="store_true",
        help="When set, ckpt_every overwrites checkpoints/step_latest.pt instead of writing step_*.pt snapshots.",
    )
    parser.add_argument(
        "--best_ckpt",
        action="store_true",
        help="Track and save the best checkpoint based on sampling metrics (MSUN for DNG, MR/RMSE for CSP).",
    )
    parser.add_argument(
        "--best_ckpt_selector",
        type=str,
        default="auto",
        choices=BEST_CKPT_SELECTOR_CHOICES,
        help=(
            "Primary metric/tag preference for --best_ckpt. "
            "auto: DNG prefers precise MSUN, CSP uses val MR/RMSE ratio "
            "with sample-first tag order. "
            "legacy restores previous DNG sample-first ordering."
        ),
    )
    # Loss weights
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("A", "F", "Y"),
        help="Loss weights for type/coord/lattice.",
    )
    parser.add_argument(
        "--coord_loss_mode",
        type=str,
        default="cart_metric_vnorm_com",
        choices=["frac_mse", "cart_metric_vnorm_com"],
        help=(
            "Coordinate loss mode. "
            "'frac_mse' is legacy wrapped fractional MSE; "
            "'cart_metric_vnorm_com' uses Cartesian metric error, removes COM drift, "
            "and normalizes by ((V/N)^(1/3))^2."
        ),
    )
    # EDM specific
    parser.add_argument("--edm_P_mean", type=float, default=-1.2)
    parser.add_argument("--edm_P_std", type=float, default=1.2)
    parser.add_argument("--sigma_data_type", type=float, default=1.0)
    parser.add_argument("--sigma_data_coord", type=float, default=0.25)
    parser.add_argument("--sigma_data_lattice", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S_churn", type=float, default=20.0)
    parser.add_argument("--S_min", type=float, default=0.0)
    parser.add_argument("--S_max", type=float, default=999.0)
    parser.add_argument("--S_noise", type=float, default=1.0)
    parser.add_argument(
        "--aa_frac_max_scale",
        type=float,
        default=0.0,
        help="Optional cap on rho-based anti-annealing scale for fractional drift (>0 enables cap).",
    )
    parser.add_argument(
        "--aa_rho_coords",
        type=float,
        default=0.0,
        help=(
            "Rho-based anti-annealing for fractional-coordinate drift. "
            "0 disables coordinate anti-annealing."
        ),
    )
    parser.add_argument(
        "--aa_rho_lattice",
        type=float,
        default=0.0,
        help=(
            "Rho-based anti-annealing for lattice drift. "
            "0 disables lattice anti-annealing."
        ),
    )
    parser.add_argument(
        "--aa_rho_types",
        type=float,
        default=0.0,
        help=(
            "Rho-based anti-annealing for atom-type drift. "
            "0 disables type anti-annealing."
        ),
    )
    parser.add_argument(
        "--stat_samples",
        type=int,
        default=1024,
        help="Number of dataset items to sample for input stats (0 to disable).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 autocast for forward passes (CUDA only).",
    )
    parser.add_argument(
        "--csp",
        action="store_true",
        help="Enable Crystal Structure Prediction mode (fix atom types, zero type loss).",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Normalise aliases, resolve derived fields, and call parser.error for invalid combos."""
    if args.type_encoding == "subatomic_tokenizer_pca":
        args.type_encoding = "subatomic_tokenizer_pca_24"
    args.nequip_relax_mode = str(args.nequip_relax_mode).strip().lower()
    try:
        args.csp_precise_topk_list = _normalize_topk_list(args.csp_precise_topk_list)
    except ValueError as exc:
        parser.error(str(exc))
    if args.csp and args.best_ckpt_selector.startswith("dng_"):
        parser.error(
            f"--best_ckpt_selector={args.best_ckpt_selector} is DNG-only; "
            "use a CSP selector (auto/csp_*) when --csp is enabled."
        )
    if (not args.csp) and args.best_ckpt_selector.startswith("csp_"):
        parser.error(
            f"--best_ckpt_selector={args.best_ckpt_selector} is CSP-only; "
            "use a DNG selector (auto/legacy/dng_*) when --csp is disabled."
        )
    if args.csp_precise_topk_samples < 0:
        parser.error("--csp_precise_topk_samples must be >= 0.")
    if args.seed < 0:
        parser.error("--seed must be >= 0.")
    if args.conditioned_loss_multiplier <= 0.0:
        parser.error("--conditioned_loss_multiplier must be > 0.")
    if not math.isfinite(float(args.cfg_contrast_weight)):
        parser.error("--cfg_contrast_weight must be finite.")
    if float(args.cfg_contrast_weight) < 0.0:
        parser.error("--cfg_contrast_weight must be >= 0.")
    if not math.isfinite(float(args.cfg_contrast_margin)):
        parser.error("--cfg_contrast_margin must be finite.")
    if float(args.cfg_contrast_margin) < 0.0:
        parser.error("--cfg_contrast_margin must be >= 0.")
    if args.sample_compute_novelty:
        print(
            "[warn] --sample_compute_novelty is deprecated and ignored; "
            "ADiT novelty is no longer computed in the runtime metric path."
        )
    try:
        aa_rho_coords, aa_rho_lattice = resolve_aa_rho_pair(
            aa_rho_coords=args.aa_rho_coords,
            aa_rho_lattice=args.aa_rho_lattice,
        )
        aa_rho_types = resolve_nonnegative_scalar("aa_rho_types", args.aa_rho_types)
    except ValueError as exc:
        parser.error(str(exc))
    args.aa_rho_coords = float(aa_rho_coords)
    args.aa_rho_lattice = float(aa_rho_lattice)
    args.aa_rho_types = float(aa_rho_types)
    args.aa_rho_by_target = {
        "types": float(args.aa_rho_types),
        "coords": float(args.aa_rho_coords),
        "lattice": float(args.aa_rho_lattice),
    }

    if args.thermo_stability_check and args.nequip_relax_mode != "sequential":
        if str(args.thermo_mlip).strip().lower() != "nequip":
            parser.error("--nequip_relax_mode=batch requires --thermo_mlip nequip.")
        if args.nequip_relax_mode == "batch":
            if str(args.nequip_optimizer).strip() != "FIRE":
                parser.error(
                    "Batched NequIP currently supports only --nequip_optimizer FIRE."
                )
            if str(args.nequip_cell_filter).strip().lower() not in {"none", "frechet"}:
                parser.error(
                    "Batched NequIP currently supports only "
                    "--nequip_cell_filter none|frechet."
                )
    if args.d_model % args.n_heads != 0:
        parser.error("--d_model must be divisible by --n_heads.")
    if args.edge_bias_n_freqs <= 0:
        parser.error("--edge_bias_n_freqs must be > 0.")
    if args.edge_bias_hidden_dim <= 0:
        parser.error("--edge_bias_hidden_dim must be > 0.")
    if args.edge_bias_n_rbf <= 0:
        parser.error("--edge_bias_n_rbf must be > 0.")
    if args.edge_bias_rbf_max <= 0.0:
        parser.error("--edge_bias_rbf_max must be > 0.")
    explicit_args = getattr(args, "_explicit_args", set())
    if args.property_lr_multiplier is not None:
        if not math.isfinite(float(args.property_lr_multiplier)):
            parser.error("--property_lr_multiplier must be finite.")
        if float(args.property_lr_multiplier) <= 0.0:
            parser.error("--property_lr_multiplier must be > 0.")
        print(
            "[warn] --property_lr_multiplier is deprecated; use "
            "--property_embed_lr_multiplier and --property_delta_lr_multiplier."
        )
        if "property_embed_lr_multiplier" not in explicit_args:
            args.property_embed_lr_multiplier = float(args.property_lr_multiplier)
        if "property_delta_lr_multiplier" not in explicit_args:
            args.property_delta_lr_multiplier = float(args.property_lr_multiplier)
    if not math.isfinite(float(args.property_embed_lr_multiplier)):
        parser.error("--property_embed_lr_multiplier must be finite.")
    if float(args.property_embed_lr_multiplier) <= 0.0:
        parser.error("--property_embed_lr_multiplier must be > 0.")
    if not math.isfinite(float(args.property_delta_lr_multiplier)):
        parser.error("--property_delta_lr_multiplier must be finite.")
    if float(args.property_delta_lr_multiplier) <= 0.0:
        parser.error("--property_delta_lr_multiplier must be > 0.")
    if args.diagnostic_eval_every < 0:
        parser.error("--diagnostic_eval_every must be >= 0.")
    if args.dataset_name == "custom" and args.nmax is None:
        parser.error("--nmax is required when --dataset_name custom.")
    args.property_name = str(args.property_name).strip()
    args.property_transform = str(args.property_transform).strip().lower()
    args.property_encoder_type = str(args.property_encoder_type).strip().lower()
    args.property_fusion_mode = str(args.property_fusion_mode).strip().lower()
    if float(args.cfg_contrast_weight) > 0.0 and not args.property_name:
        parser.error("--cfg_contrast_weight requires --property_name.")
    if args.use_property_film and not args.property_name:
        parser.error("--use_property_film requires --property_name.")
    if args.film_start_layer < -1:
        parser.error("--film_start_layer must be -1 or a non-negative layer index.")
    if args.film_num_layers < -1:
        parser.error("--film_num_layers must be -1 or a positive layer count.")
    if args.film_num_layers == 0:
        parser.error("--film_num_layers must be -1 or a positive layer count.")
    resolved_film_start = args.n_layers // 2 if args.film_start_layer < 0 else args.film_start_layer
    resolved_film_num = args.n_layers - resolved_film_start if args.film_num_layers < 0 else args.film_num_layers
    if args.use_property_film:
        if resolved_film_start < 0 or resolved_film_start >= args.n_layers:
            parser.error("--film_start_layer must select an existing Transformer layer.")
        if resolved_film_num != args.n_layers - resolved_film_start:
            parser.error("--film_num_layers must cover film_start_layer through the final layer.")
    if args.property_name:
        if not math.isfinite(float(args.property_mean)):
            parser.error("--property_mean must be finite.")
        if not math.isfinite(float(args.property_std)) or float(args.property_std) == 0.0:
            parser.error("--property_std must be finite and non-zero.")
        if not (0.0 <= float(args.p_uncond) <= 1.0):
            parser.error("--p_uncond must be in [0, 1].")
        if args.property_num_clusters <= 0:
            parser.error("--property_num_clusters must be > 0.")
        if not math.isfinite(float(args.property_cluster_min)):
            parser.error("--property_cluster_min must be finite.")
        if not math.isfinite(float(args.property_cluster_max)):
            parser.error("--property_cluster_max must be finite.")
        if float(args.property_cluster_min) >= float(args.property_cluster_max):
            parser.error("--property_cluster_min must be < --property_cluster_max.")
        if args.sample_property_value is not None and not math.isfinite(
            float(args.sample_property_value)
        ):
            parser.error("--sample_property_value must be finite when provided.")

    dataset_default_nmax = DATASET_NMAX_DEFAULTS.get(args.dataset_name, DEFAULT_NMAX)
    nmax = int(args.nmax) if args.nmax is not None else int(dataset_default_nmax)
    if nmax <= 0:
        parser.error("--nmax must be a positive integer.")
    # Store resolved value so wandb config captures the effective setting.
    args.nmax = nmax
