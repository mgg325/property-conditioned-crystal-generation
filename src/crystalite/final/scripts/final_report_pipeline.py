from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

from src.utils.sample_stats import make_chgnet_and_relaxer

PYTHON = ROOT / ".venv" / "bin" / "python"
OUTPUT_ROOT = ROOT / "outputs" / "final report output"

TARGETS = (2, 4, 8, 12)
GUIDANCE_SCALES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class GenerationConfig:
    checkpoint: str
    sample_num_steps: int
    sample_mode: str
    atom_count_strategy: str = "empirical"
    sample_seed: int = 123
    num_samples: int = 128
    sample_chunk_size: int = 128
    bf16: bool = False
    data_root: str = "data/anisonet_dielectric_crystalite"
    dataset_name: str = "custom"
    nmax: int = 20
    property_name: str = "dft_dielectric_scalar"


@dataclass
class GroupSpec:
    method: str
    group: str
    guidance_factor: str
    target: str
    checkpoint: str
    raw_source_dir: str | None
    generation: GenerationConfig | None


def _group_name(method: str, guidance: int | None, target: int | None) -> str:
    if guidance is None:
        return f"{method}__uncond"
    return f"{method}__g{guidance}__t{target}"


def _build_group_specs() -> list[GroupSpec]:
    specs: list[GroupSpec] = []

    official_baseline = GenerationConfig(
        checkpoint="checkpoints/best.pt",
        sample_num_steps=150,
        sample_mode="ema",
        bf16=True,
    )
    specs.append(
        GroupSpec(
            method="official_baseline",
            group="official_baseline__uncond",
            guidance_factor="baseline",
            target="uncond",
            checkpoint=official_baseline.checkpoint,
            raw_source_dir=None,
            generation=official_baseline,
        )
    )

    additive_cfg = GenerationConfig(
        checkpoint="outputs/dielectric_cfg_finetune_10epoch_bs32/checkpoints/final.pt",
        sample_num_steps=50,
        sample_mode="ema",
    )
    additive_existing = {
        (None, None): "outputs/dielectric_cfg_samples_g0_uncond_n128",
        (1, 2): "outputs/dielectric_cfg_samples_g1_t2_n128",
        (1, 4): "outputs/dielectric_cfg_samples_g1_t4_n128",
        (1, 8): "outputs/dielectric_cfg_samples_g1_t8_n128",
        (1, 12): "outputs/dielectric_cfg_samples_g1_t12_n128",
        (2, 2): "outputs/dielectric_cfg_samples_g2_t2_n128",
        (2, 4): "outputs/dielectric_cfg_samples_g2_t4_n128",
        (2, 8): "outputs/dielectric_cfg_samples_g2_t8_n128",
        (2, 12): "outputs/dielectric_cfg_samples_g2_t12_n128",
    }
    specs.extend(
        _expand_method_specs(
            method="additive",
            cfg=additive_cfg,
            existing=additive_existing,
        )
    )

    concat_cfg = GenerationConfig(
        checkpoint=(
            "outputs/clean_adaln_softcluster_finetune_10epoch_nmax20_puncond0_ema0995/"
            "checkpoints/final.pt"
        ),
        sample_num_steps=50,
        sample_mode="ema",
    )
    specs.extend(_expand_method_specs(method="concat_mlp", cfg=concat_cfg, existing={}))

    residual_cfg = GenerationConfig(
        checkpoint="outputs/dielectric_residual_delta_v3/checkpoints/step_0004000.pt",
        sample_num_steps=50,
        sample_mode="ema",
    )
    residual_existing = {
        (None, None): "outputs/dielectric_step4000_grid_samples/step4000_g0p0_t4_n128",
        **{
            (g, t): f"outputs/dielectric_step4000_grid_samples/step4000_g{g}p0_t{t}_n128"
            for g in (1, 2, 3, 4)
            for t in TARGETS
        },
    }
    specs.extend(
        _expand_method_specs(
            method="residual_delta",
            cfg=residual_cfg,
            existing=residual_existing,
        )
    )

    add_film_cfg = GenerationConfig(
        checkpoint="outputs/dielectric_cfg_film_upper_from_v1_unfrozen_5epoch_bs32/checkpoints/final.pt",
        sample_num_steps=50,
        sample_mode="ema",
    )
    add_film_existing = {
        (None, None): "outputs/dielectric_cfg_film_g0_uncond_n128",
        (1, 2): "outputs/dielectric_cfg_film_g1_t2_n128",
        (1, 4): "outputs/dielectric_cfg_film_g1_t4_n128",
        (1, 8): "outputs/dielectric_cfg_film_g1_t8_n128",
        (1, 12): "outputs/dielectric_cfg_film_g1_t12_n128",
        (2, 2): "outputs/dielectric_cfg_film_g2_t2_n128",
        (2, 4): "outputs/dielectric_cfg_film_g2_t4_n128",
        (2, 8): "outputs/dielectric_cfg_film_g2_t8_n128",
        (2, 12): "outputs/dielectric_cfg_film_g2_t12_n128",
    }
    specs.extend(
        _expand_method_specs(
            method="additive_plus_film",
            cfg=add_film_cfg,
            existing=add_film_existing,
        )
    )

    fixed_film_cfg = GenerationConfig(
        checkpoint="outputs/film_sole_path/checkpoints/final.pt",
        sample_num_steps=100,
        sample_mode="regular",
        sample_chunk_size=128,
        bf16=True,
    )
    specs.extend(_expand_method_specs(method="fixed_film", cfg=fixed_film_cfg, existing={}))
    return specs


def _expand_method_specs(
    *,
    method: str,
    cfg: GenerationConfig,
    existing: dict[tuple[int | None, int | None], str],
) -> list[GroupSpec]:
    specs = [
        GroupSpec(
            method=method,
            group=_group_name(method, None, None),
            guidance_factor="0",
            target="uncond",
            checkpoint=cfg.checkpoint,
            raw_source_dir=existing.get((None, None)),
            generation=None if (None, None) in existing else cfg,
        )
    ]
    for g in GUIDANCE_SCALES:
        for t in TARGETS:
            key = (g, t)
            specs.append(
                GroupSpec(
                    method=method,
                    group=_group_name(method, g, t),
                    guidance_factor=str(g),
                    target=str(t),
                    checkpoint=cfg.checkpoint,
                    raw_source_dir=existing.get(key),
                    generation=None if key in existing else cfg,
                )
            )
    return specs


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[run]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _generate_group(
    spec: GroupSpec,
    cfg: GenerationConfig,
    *,
    raw_root: Path,
) -> Path:
    out_dir = raw_root / spec.group
    if out_dir.exists():
        return out_dir
    cmd = [
        str(PYTHON),
        "src/sample_crystalite_ckpt.py",
        "--checkpoint",
        str(ROOT / cfg.checkpoint),
        "--output_dir",
        str(out_dir),
        "--device",
        "cuda",
        "--num_samples",
        str(cfg.num_samples),
        "--sample_chunk_size",
        str(cfg.sample_chunk_size),
        "--sample_seed",
        str(cfg.sample_seed),
        "--sample_num_steps",
        str(cfg.sample_num_steps),
        "--sample_mode",
        cfg.sample_mode,
        "--atom_count_strategy",
        cfg.atom_count_strategy,
        "--data_root",
        cfg.data_root,
        "--dataset_name",
        cfg.dataset_name,
        "--nmax",
        str(cfg.nmax),
        "--save_cifs",
        "--no-save_pt",
        "--no-save_extxyz",
        "--cif_limit",
        str(cfg.num_samples),
    ]
    if cfg.bf16:
        cmd.append("--bf16")
    if spec.target != "uncond":
        cmd.extend(
            [
                "--property_name",
                cfg.property_name,
                "--property_value",
                spec.target,
                "--diffusion_guidance_factor",
                spec.guidance_factor,
            ]
        )
    _run(cmd, cwd=ROOT)
    return out_dir


def _resolve_raw_source(spec: GroupSpec, *, raw_root: Path) -> Path:
    if spec.raw_source_dir is not None:
        return ROOT / spec.raw_source_dir
    if spec.generation is None:
        raise ValueError(f"Group {spec.group} has neither a raw source nor generation config.")
    return _generate_group(spec, spec.generation, raw_root=raw_root)


def _iter_cif_paths(group_dir: Path, *, limit_cifs: int) -> list[Path]:
    cif_dir = group_dir / "cifs"
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")
    paths = sorted(cif_dir.glob("sample_*.cif"))
    if len(paths) < limit_cifs:
        raise ValueError(f"Expected at least {limit_cifs} CIFs in {cif_dir}, found {len(paths)}.")
    return paths[:limit_cifs]


def _relax_group(
    spec: GroupSpec,
    *,
    raw_group_dir: Path,
    relaxed_root: Path,
    relaxer: Any,
    relax_steps: int,
    limit_cifs: int,
) -> dict[str, Any]:
    cif_paths = _iter_cif_paths(raw_group_dir, limit_cifs=limit_cifs)
    out_dir = relaxed_root / spec.group
    cif_out_dir = out_dir / "cifs"
    cif_out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    failed = 0
    total_relax_seconds = 0.0
    iterator = tqdm(cif_paths, desc=f"relax {spec.group}", dynamic_ncols=True)
    for cif_path in iterator:
        sample_name = cif_path.name
        out_path = cif_out_dir / sample_name
        if out_path.exists():
            manifest.append(
                {
                    "sample_file": sample_name,
                    "raw_cif_path": str(cif_path),
                    "relaxed_cif_path": str(out_path),
                    "relax_success": True,
                    "relax_seconds": None,
                    "relax_error": "",
                }
            )
            continue
        started = time.monotonic()
        relax_success = False
        relax_error = ""
        try:
            structure = Structure.from_file(cif_path)
            relaxed = relaxer.relax(structure, steps=relax_steps, verbose=False)
            relaxed_structure = relaxed["final_structure"]
            out_path.write_text(str(CifWriter(relaxed_structure)))
            relax_success = True
        except Exception as exc:
            failed += 1
            relax_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        total_relax_seconds += elapsed
        manifest.append(
            {
                "sample_file": sample_name,
                "raw_cif_path": str(cif_path),
                "relaxed_cif_path": str(out_path) if relax_success else "",
                "relax_success": relax_success,
                "relax_seconds": round(elapsed, 6),
                "relax_error": relax_error,
            }
        )
    summary = {
        "group": spec.group,
        "method": spec.method,
        "guidance_factor": spec.guidance_factor,
        "target": spec.target,
        "raw_group_dir": str(raw_group_dir),
        "relaxed_group_dir": str(out_dir),
        "checkpoint": spec.checkpoint,
        "relax_steps": relax_steps,
        "limit_cifs": limit_cifs,
        "num_raw_cifs": len(cif_paths),
        "num_relax_failures": failed,
        "num_relaxed_cifs": len(cif_paths) - failed,
        "total_relax_seconds": round(total_relax_seconds, 6),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "relax_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_group_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_evaluation(
    specs: list[GroupSpec],
    *,
    relaxed_root: Path,
    evaluation_dir: Path,
    limit_cifs: int,
    thermo_ppd_mp: Path | None,
    thermo_ehull_method: str,
    thermo_device: str,
) -> None:
    if evaluation_dir.exists():
        shutil.rmtree(evaluation_dir)
    cmd = [
        str(PYTHON),
        "src/eval_crystalite_samples.py",
        "--output_dir",
        str(evaluation_dir),
        "--train_csv",
        str(ROOT / "data/anisonet_dielectric_crystalite/raw/train.csv"),
        "--anisonet_checkpoint",
        "external/anisonet/anisonet-stock.ckpt",
        "--anisonet_batch_size",
        "8",
    ]
    if thermo_ppd_mp is not None:
        cmd.extend(
            [
                "--thermo_ppd_mp",
                str(thermo_ppd_mp),
                "--thermo_ehull_method",
                str(thermo_ehull_method),
                "--thermo_device",
                str(thermo_device),
            ]
        )
    for spec in specs:
        cmd.extend(
            [
                "--group",
                ",".join(
                    [
                        spec.group,
                        spec.guidance_factor,
                        spec.target,
                        str(relaxed_root / spec.group),
                    ]
                ),
            ]
        )
    _run(cmd, cwd=ROOT)


def _write_consolidated_reports(
    *,
    specs: list[GroupSpec],
    manifest_rows: list[dict[str, Any]],
    evaluation_dir: Path,
    output_root: Path,
) -> None:
    spec_by_group = {spec.group: spec for spec in specs}
    manifest_by_group = {row["group"]: row for row in manifest_rows}

    per_sample_path = evaluation_dir / "per_sample_evaluation.csv"
    group_summary_path = evaluation_dir / "group_summary.csv"

    consolidated_per_sample: list[dict[str, Any]] = []
    with open(per_sample_path, newline="") as f:
        for row in csv.DictReader(f):
            spec = spec_by_group[row["group"]]
            manifest = manifest_by_group[row["group"]]
            merged = {
                "method": spec.method,
                "checkpoint": spec.checkpoint,
                "raw_group_dir": manifest["raw_group_dir"],
                "relaxed_group_dir": manifest["relaxed_group_dir"],
                **row,
            }
            consolidated_per_sample.append(merged)
    _write_group_manifest(
        consolidated_per_sample,
        output_root / "consolidated_per_sample_evaluation.csv",
    )

    consolidated_group_summary: list[dict[str, Any]] = []
    with open(group_summary_path, newline="") as f:
        for row in csv.DictReader(f):
            spec = spec_by_group[row["group"]]
            manifest = manifest_by_group[row["group"]]
            merged = {
                "method": spec.method,
                "checkpoint": spec.checkpoint,
                "raw_group_dir": manifest["raw_group_dir"],
                "relaxed_group_dir": manifest["relaxed_group_dir"],
                "relax_steps": manifest["relax_steps"],
                **row,
            }
            consolidated_group_summary.append(merged)
    _write_group_manifest(
        consolidated_group_summary,
        output_root / "consolidated_group_summary.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final Crystalite report set with GPU CHGNet relaxation."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for raw/reused manifests, relaxed CIFs, and evaluation CSVs.",
    )
    parser.add_argument(
        "--relax_steps",
        type=int,
        default=50,
        help="Number of CHGNet relaxation steps per CIF.",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Reuse only existing raw groups and fail if a missing group would need generation.",
    )
    parser.add_argument(
        "--skip_relaxation",
        action="store_true",
        help="Skip CHGNet relaxation and only run evaluation on already prepared relaxed groups.",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip eval_crystalite_samples.py after relaxation.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional subset of methods to run. "
            "Choices: official_baseline additive concat_mlp residual_delta "
            "additive_plus_film fixed_film"
        ),
    )
    parser.add_argument(
        "--exclude_groups",
        nargs="*",
        default=None,
        help="Optional list of exact group names to skip entirely.",
    )
    parser.add_argument(
        "--limit_cifs",
        type=int,
        default=128,
        help="Number of CIFs per group to relax/evaluate. Default 128; use a smaller value for smoke tests.",
    )
    parser.add_argument(
        "--force_cuda",
        action="store_true",
        help="Force CHGNet relaxation to request cuda even when torch.cuda.is_available() is false.",
    )
    parser.add_argument(
        "--thermo_ppd_mp",
        type=Path,
        default=None,
        help="PatchedPhaseDiagram pickle used to add e_above_hull / SUN / MSUN to evaluation.",
    )
    parser.add_argument(
        "--thermo_ehull_method",
        type=str,
        default="mp2020_like",
        choices=["uncorrected", "mp2020_like"],
        help="How to compute e_above_hull when thermo evaluation is enabled.",
    )
    parser.add_argument(
        "--thermo_device",
        type=str,
        default="cpu",
        help="Device for offline CHGNet thermo energy prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _build_group_specs()
    if args.methods:
        allowed = set(args.methods)
        specs = [spec for spec in specs if spec.method in allowed]
        if not specs:
            raise ValueError("No groups matched --methods.")
    if args.exclude_groups:
        excluded = set(args.exclude_groups)
        specs = [spec for spec in specs if spec.group not in excluded]
        if not specs:
            raise ValueError("All groups were excluded by --exclude_groups.")
    output_root = args.output_root
    raw_root = output_root / "raw_groups"
    relaxed_root = output_root / "relaxed_groups"
    evaluation_dir = output_root / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    relaxed_root.mkdir(parents=True, exist_ok=True)

    source_manifest_rows: list[dict[str, Any]] = []
    for spec in specs:
        if args.skip_generation and spec.raw_source_dir is None:
            generated_dir = raw_root / spec.group
            if not generated_dir.exists():
                raise ValueError(
                    f"Missing raw source for {spec.group}, but --skip_generation was set."
                )
        raw_dir = _resolve_raw_source(spec, raw_root=raw_root)
        source_manifest_rows.append(
            {
                "group": spec.group,
                "method": spec.method,
                "guidance_factor": spec.guidance_factor,
                "target": spec.target,
                "checkpoint": spec.checkpoint,
                "raw_group_dir": str(raw_dir),
                "raw_source_type": "existing" if spec.raw_source_dir is not None else "generated",
                "raw_source_dir_original": spec.raw_source_dir or "",
                "relaxed_group_dir": str(relaxed_root / spec.group),
                "relax_steps": args.relax_steps,
            }
        )
    _write_group_manifest(source_manifest_rows, output_root / "group_manifest.csv")

    relax_summary_rows: list[dict[str, Any]] = []
    if not args.skip_relaxation:
        _, relaxer, resolved_device = make_chgnet_and_relaxer(
            "cuda",
            force_cuda=bool(args.force_cuda),
        )
        print(f"[relax] using CHGNet device={resolved_device}")
        for spec in specs:
            raw_dir = Path(next(row["raw_group_dir"] for row in source_manifest_rows if row["group"] == spec.group))
            relax_summary_rows.append(
                _relax_group(
                    spec,
                    raw_group_dir=raw_dir,
                    relaxed_root=relaxed_root,
                    relaxer=relaxer,
                    relax_steps=args.relax_steps,
                    limit_cifs=args.limit_cifs,
                )
            )
        _write_group_manifest(relax_summary_rows, output_root / "relax_summary.csv")
    else:
        relax_summary_rows = source_manifest_rows

    if not args.skip_evaluation:
        _run_evaluation(
            specs,
            relaxed_root=relaxed_root,
            evaluation_dir=evaluation_dir,
            limit_cifs=args.limit_cifs,
            thermo_ppd_mp=args.thermo_ppd_mp,
            thermo_ehull_method=args.thermo_ehull_method,
            thermo_device=args.thermo_device,
        )
        _write_consolidated_reports(
            specs=specs,
            manifest_rows=source_manifest_rows if args.skip_relaxation else relax_summary_rows,
            evaluation_dir=evaluation_dir,
            output_root=output_root,
        )
    print(f"[done] output root: {output_root}")


if __name__ == "__main__":
    main()
