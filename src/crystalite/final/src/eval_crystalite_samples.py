"""Offline evaluation for generated Crystalite CIF sample folders.

This script intentionally never calls the sampler. It reads existing sample CIFs,
computes lightweight structural metrics, predicts dielectric scalar values with
the local AnisoNet model when available, and writes MatterGen-style CSV/JSON
summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-anisonet"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader

from src.eval.dng_eval import compute_sun_msun_from_thermo_rates
from src.eval.crystal import Crystal
from src.eval.stability import load_phase_diagram
from src.utils.sample_stats import (
    compute_e_above_hull_mp2020_like,
    compute_e_above_hull_uncorrected,
    make_chgnet_and_relaxer,
)


ROOT = REPO_ROOT
ANISONET_ROOT = Path("external/anisonet")
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "dielectric_cfg_evaluation_mattergen_style"
DEFAULT_TRAIN_CSV = ROOT / "data" / "anisonet_dielectric_crystalite" / "raw" / "train.csv"
DEFAULT_ANISONET_CKPT = ANISONET_ROOT / "anisonet-stock.ckpt"

SUMMARY_COLUMNS = [
    "group",
    "guidance_factor",
    "target",
    "n_structures",
    "n_readable",
    "n_struct_valid",
    "n_predicted",
    "mean_predicted_dielectric_scalar",
    "median_predicted_dielectric_scalar",
    "std_predicted_dielectric_scalar",
    "mae",
    "rmse",
    "frac_within_1",
    "frac_within_2",
    "composition_valid_fraction",
    "structure_valid_fraction",
    "structure_comp_valid_fraction",
    "metal_like_proxy_fraction",
    "min_distance_median",
    "volume_per_atom_median",
    "un_rate",
    "thermo_checked",
    "thermo_success",
    "thermo_failed",
    "thermo_success_rate",
    "thermo_stable_rate",
    "thermo_metastable_rate",
    "thermo_e_above_hull_mean",
    "SUN",
    "MSUN",
]

PER_SAMPLE_COLUMNS = [
    "group",
    "guidance_factor",
    "target",
    "sample_id",
    "cif_path",
    "readable",
    "composition_valid",
    "structure_valid",
    "structure_comp_valid",
    "min_distance",
    "volume",
    "volume_per_atom",
    "num_atoms",
    "metal_like_proxy",
    "unique",
    "novel",
    "energy_above_hull_per_atom",
    "stable",
    "metastable",
    "thermo_success",
    "thermo_fail_reason",
    "predicted_dft_dielectric_scalar",
    "prediction_success",
    "prediction_error",
]


@dataclass(frozen=True)
class SampleGroup:
    group: str
    guidance_factor: str
    target: str
    path: Path


@dataclass
class ThermoContext:
    backend: str
    device_requested: str
    device_actual: str
    ehull_method: str
    ppd: Any
    chgnet_model: Any
    mp2020_compat: Any


DEFAULT_GROUPS = [
    SampleGroup("official_baseline", "baseline", "", ROOT / "outputs" / "baseline_128"),
    SampleGroup(
        "finetuned_g0_uncond",
        "0",
        "uncond",
        ROOT / "outputs" / "dielectric_cfg_samples_g0_uncond_n128",
    ),
    SampleGroup("g1_t2", "1", "2", ROOT / "outputs" / "dielectric_cfg_samples_g1_t2_n128"),
    SampleGroup("g1_t4", "1", "4", ROOT / "outputs" / "dielectric_cfg_samples_g1_t4_n128"),
    SampleGroup("g1_t8", "1", "8", ROOT / "outputs" / "dielectric_cfg_samples_g1_t8_n128"),
    SampleGroup("g1_t12", "1", "12", ROOT / "outputs" / "dielectric_cfg_samples_g1_t12_n128"),
    SampleGroup("g2_t2", "2", "2", ROOT / "outputs" / "dielectric_cfg_samples_g2_t2_n128"),
    SampleGroup("g2_t4", "2", "4", ROOT / "outputs" / "dielectric_cfg_samples_g2_t4_n128"),
    SampleGroup("g2_t8", "2", "8", ROOT / "outputs" / "dielectric_cfg_samples_g2_t8_n128"),
    SampleGroup("g2_t12", "2", "12", ROOT / "outputs" / "dielectric_cfg_samples_g2_t12_n128"),
]


def _nan() -> float:
    return float("nan")


def _sample_id_from_path(path: Path) -> int | str:
    match = re.search(r"sample_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return path.stem


def _structure_to_array_dict(structure: Structure, sample_idx: Any) -> dict[str, Any]:
    return {
        "frac_coords": np.asarray(structure.frac_coords, dtype=np.float64),
        "atom_types": np.asarray([site.specie.Z for site in structure], dtype=np.int64),
        "lengths": np.asarray(structure.lattice.abc, dtype=np.float64),
        "angles": np.asarray(structure.lattice.angles, dtype=np.float64),
        "sample_idx": sample_idx,
    }


def _is_finite_positive_lattice(structure: Structure) -> bool:
    try:
        matrix = np.asarray(structure.lattice.matrix, dtype=np.float64)
        lengths = np.asarray(structure.lattice.abc, dtype=np.float64)
        angles = np.asarray(structure.lattice.angles, dtype=np.float64)
        return bool(
            np.isfinite(matrix).all()
            and np.isfinite(lengths).all()
            and np.isfinite(angles).all()
            and np.all(lengths > 0.0)
            and np.isfinite(structure.volume)
            and structure.volume > 0.0
        )
    except Exception:
        return False


def _minimum_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return _nan()
    distances = np.asarray(structure.distance_matrix, dtype=np.float64)
    distances = distances + np.diag(np.full(distances.shape[0], np.inf))
    value = float(np.nanmin(distances))
    return value if math.isfinite(value) else _nan()


def _safe_bool(value: Any) -> bool | float:
    if value is None:
        return _nan()
    return bool(value)


def _metal_like_proxy(structure: Structure) -> bool:
    try:
        import smact

        symbols = [site.specie.symbol for site in structure]
        return bool(symbols) and all(symbol in smact.metals for symbol in symbols)
    except Exception:
        return False


def _build_thermo_context(args: argparse.Namespace) -> ThermoContext | None:
    if args.thermo_ppd_mp is None:
        return None
    if not args.thermo_ppd_mp.exists():
        raise FileNotFoundError(
            f"Thermo phase-diagram pickle not found: {args.thermo_ppd_mp}"
        )
    backend = str(args.thermo_backend).strip().lower()
    if backend != "chgnet":
        raise ValueError(
            "Offline thermo evaluation currently supports only --thermo_backend chgnet."
        )
    chgnet_model, _relaxer, resolved_device = make_chgnet_and_relaxer(
        str(args.thermo_device)
    )
    del _relaxer
    ehull_method = str(args.thermo_ehull_method).strip().lower()
    mp2020_compat = None
    if ehull_method == "mp2020_like":
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

        mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)
    return ThermoContext(
        backend=backend,
        device_requested=str(args.thermo_device),
        device_actual=str(resolved_device),
        ehull_method=ehull_method,
        ppd=load_phase_diagram(args.thermo_ppd_mp),
        chgnet_model=chgnet_model,
        mp2020_compat=mp2020_compat,
    )


def _load_group_records(group: SampleGroup) -> tuple[list[dict[str, Any]], list[Structure | None]]:
    cifs_dir = group.path / "cifs"
    if not group.path.exists():
        raise FileNotFoundError(f"Sample group path does not exist: {group.path}")
    if not cifs_dir.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {cifs_dir}")

    cif_paths = sorted(cifs_dir.glob("*.cif"))
    records: list[dict[str, Any]] = []
    structures: list[Structure | None] = []
    for cif_path in cif_paths:
        sample_id = _sample_id_from_path(cif_path)
        base: dict[str, Any] = {
            "group": group.group,
            "guidance_factor": group.guidance_factor,
            "target": group.target,
            "sample_id": sample_id,
            "cif_path": str(cif_path),
            "readable": False,
            "composition_valid": False,
            "structure_valid": False,
            "structure_comp_valid": False,
            "min_distance": _nan(),
            "volume": _nan(),
            "volume_per_atom": _nan(),
            "num_atoms": _nan(),
            "metal_like_proxy": False,
            "unique": False,
            "novel": False,
            "energy_above_hull_per_atom": _nan(),
            "stable": _nan(),
            "metastable": _nan(),
            "thermo_success": False,
            "thermo_fail_reason": "",
            "predicted_dft_dielectric_scalar": _nan(),
            "prediction_success": False,
            "prediction_error": "",
        }
        try:
            structure = Structure.from_file(cif_path)
            finite_lattice = _is_finite_positive_lattice(structure)
            crystal = Crystal(_structure_to_array_dict(structure, sample_id))
            num_atoms = int(len(structure))
            volume = float(structure.volume)
            base.update(
                {
                    "readable": True,
                    "composition_valid": bool(crystal.comp_valid),
                    "structure_valid": bool(finite_lattice and crystal.struct_valid),
                    "structure_comp_valid": bool(
                        finite_lattice and crystal.comp_valid and crystal.struct_valid
                    ),
                    "min_distance": _minimum_distance(structure),
                    "volume": volume,
                    "volume_per_atom": volume / num_atoms if num_atoms > 0 else _nan(),
                    "num_atoms": num_atoms,
                    "metal_like_proxy": _metal_like_proxy(structure),
                }
            )
            structures.append(structure)
        except Exception as exc:
            base["prediction_error"] = f"parse_error: {exc}"
            structures.append(None)
        records.append(base)
    return records, structures


def _load_train_structures(train_csv: Path) -> list[Structure]:
    df = pd.read_csv(train_csv)
    if "cif" not in df.columns:
        raise ValueError(f"Reference train CSV is missing a cif column: {train_csv}")
    structures: list[Structure] = []
    for idx, cif in enumerate(df["cif"].tolist()):
        if not isinstance(cif, str) or not cif.strip():
            continue
        try:
            structures.append(Structure.from_str(cif, fmt="cif"))
        except Exception as exc:
            print(f"[warn] skipped unparseable train CIF at row {idx}: {exc}")
    return structures


def _compute_thermo_metrics_for_group(
    records: list[dict[str, Any]],
    structures: list[Structure | None],
    *,
    thermo: ThermoContext,
) -> None:
    for idx, structure in enumerate(structures):
        if structure is None or not bool(records[idx]["structure_valid"]):
            records[idx]["thermo_fail_reason"] = "skipped_invalid_structure"
            continue
        try:
            prediction = thermo.chgnet_model.predict_structure(structure)
            e_total = float(prediction["e"]) * float(structure.num_sites)
        except Exception as exc:
            records[idx]["thermo_fail_reason"] = f"predict_exception:{type(exc).__name__}"
            continue

        try:
            if thermo.ehull_method == "mp2020_like":
                e_above, fail_reason = compute_e_above_hull_mp2020_like(
                    thermo.ppd,
                    structure,
                    e_total,
                    mp2020_compat=thermo.mp2020_compat,
                )
            else:
                e_above, fail_reason = compute_e_above_hull_uncorrected(
                    thermo.ppd,
                    structure,
                    e_total,
                )
        except Exception as exc:
            records[idx]["thermo_fail_reason"] = f"ehull_exception:{type(exc).__name__}"
            continue

        if fail_reason is not None or e_above is None or not math.isfinite(float(e_above)):
            records[idx]["thermo_fail_reason"] = str(fail_reason or "nan_eabove")
            continue

        e_above = float(e_above)
        records[idx]["energy_above_hull_per_atom"] = e_above
        records[idx]["stable"] = bool(e_above <= 0.0)
        records[idx]["metastable"] = bool(e_above <= 0.1)
        records[idx]["thermo_success"] = True
        records[idx]["thermo_fail_reason"] = ""


def _composition_key(structure: Structure) -> str:
    return structure.composition.reduced_formula


def _matches(a: Structure, b: Structure, matcher: StructureMatcher) -> bool:
    try:
        return matcher.get_rms_dist(a, b) is not None
    except Exception:
        return False


def _set_uniqueness_and_novelty(
    records: list[dict[str, Any]],
    structures: list[Structure | None],
    train_structures: list[Structure],
) -> None:
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    train_by_comp: dict[str, list[Structure]] = {}
    for train_structure in train_structures:
        if not _is_finite_positive_lattice(train_structure):
            continue
        train_by_comp.setdefault(_composition_key(train_structure), []).append(train_structure)

    seen_by_comp: dict[str, list[Structure]] = {}
    for idx, structure in enumerate(structures):
        if structure is None or not _is_finite_positive_lattice(structure):
            records[idx]["unique"] = False
            records[idx]["novel"] = False
            continue

        comp_key = _composition_key(structure)
        duplicates = seen_by_comp.setdefault(comp_key, [])
        is_unique = not any(_matches(structure, other, matcher) for other in duplicates)
        if is_unique:
            duplicates.append(structure)
        records[idx]["unique"] = bool(is_unique)

        train_candidates = train_by_comp.get(comp_key, [])
        records[idx]["novel"] = not any(
            _matches(structure, train_structure, matcher)
            for train_structure in train_candidates
        )


def _load_anisonet_model(dataset: Any, checkpoint_path: Path, device: torch.device) -> tuple[Any, Any]:
    anisonet_src = ANISONET_ROOT / "src"
    if str(anisonet_src) not in sys.path:
        sys.path.insert(0, str(anisonet_src))

    from e3nn.io import CartesianTensor
    from anisonet.model import E3nnModel

    ct = CartesianTensor("ij=ji")
    net = E3nnModel(
        in_dim=118,
        em_dim=48,
        in_attr_dim=118,
        em_attr_dim=48,
        irreps_out=str(ct),
        layers=2,
        mul=48,
        lmax=3,
        max_radius=dataset.cutoff,
        number_of_basis=15,
        num_neighbors=dataset.num_neighbors,
        reduce_output=True,
        same_em_layer=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    adjusted_state_dict = {
        key.replace("model.", "", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    net.load_state_dict(adjusted_state_dict)
    net.eval().to(device)
    return net, ct


def _predict_with_anisonet(
    records: list[dict[str, Any]],
    structures: list[Structure | None],
    checkpoint_path: Path,
    batch_size: int,
    require_structure_valid: bool,
) -> None:
    readable_indices = []
    for idx, structure in enumerate(structures):
        if structure is None or not bool(records[idx]["readable"]):
            continue
        if require_structure_valid and not bool(records[idx]["structure_valid"]):
            records[idx]["prediction_error"] = (
                "skipped_anisonet: structure_valid=False; "
                "AnisoNet graph construction is unsafe for collapsed geometries"
            )
            continue
        readable_indices.append(idx)
    if not readable_indices:
        return
    if not checkpoint_path.exists():
        for idx in readable_indices:
            records[idx]["prediction_error"] = f"AnisoNet checkpoint not found: {checkpoint_path}"
        return

    try:
        anisonet_src = ANISONET_ROOT / "src"
        if str(anisonet_src) not in sys.path:
            sys.path.insert(0, str(anisonet_src))
        from anisonet.data import BaseDataset, collate_fn

        torch.manual_seed(1234)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")
        adaptor = AseAtomsAdaptor()
        atoms = [adaptor.get_atoms(structures[idx]) for idx in readable_indices]
        df = pd.DataFrame(
            {
                "structure": atoms,
                "target": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in atoms],
            }
        )
        dataset = BaseDataset(df, cutoff=5)
        net, ct = _load_anisonet_model(dataset, checkpoint_path, device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        outputs = []
        with torch.no_grad():
            for batch in loader:
                batch.to(device)
                outputs.append(net(batch).detach().cpu())
        cart_pred = ct.to_cartesian(torch.cat(outputs, dim=0))
        for idx, tensor in zip(readable_indices, cart_pred):
            tensor_np = tensor.numpy()
            eigenvalues = np.linalg.eigvalsh(tensor_np)
            records[idx]["predicted_dft_dielectric_scalar"] = float(eigenvalues.mean())
            records[idx]["prediction_success"] = True
            records[idx]["prediction_error"] = ""
        del loader, dataset, net, ct, outputs, cart_pred
        gc.collect()
    except Exception as exc:
        message = f"AnisoNet prediction failed: {type(exc).__name__}: {exc}"
        for idx in readable_indices:
            records[idx]["prediction_success"] = False
            if not records[idx]["prediction_error"]:
                records[idx]["prediction_error"] = message


def _summarize_group(records: list[dict[str, Any]], group: SampleGroup) -> dict[str, Any]:
    n_structures = len(records)
    n_readable = sum(bool(row["readable"]) for row in records)
    n_struct_valid = sum(bool(row["structure_valid"]) for row in records)
    predicted = [
        float(row["predicted_dft_dielectric_scalar"])
        for row in records
        if bool(row["prediction_success"])
        and math.isfinite(float(row["predicted_dft_dielectric_scalar"]))
    ]
    target_value: float | None
    try:
        target_value = float(group.target)
    except (TypeError, ValueError):
        target_value = None

    errors = [value - target_value for value in predicted] if target_value is not None else []
    min_distances = [
        float(row["min_distance"])
        for row in records
        if math.isfinite(float(row["min_distance"]))
    ]
    volume_per_atom = [
        float(row["volume_per_atom"])
        for row in records
        if math.isfinite(float(row["volume_per_atom"]))
    ]
    thermo_values = [
        float(row["energy_above_hull_per_atom"])
        for row in records
        if bool(row["thermo_success"])
        and math.isfinite(float(row["energy_above_hull_per_atom"]))
    ]
    thermo_checked = sum(bool(row["structure_valid"]) for row in records)
    thermo_success = sum(bool(row["thermo_success"]) for row in records)
    thermo_failed = max(0, thermo_checked - thermo_success)
    thermo_stable_rate = (
        float(np.mean([bool(row["stable"]) for row in records if bool(row["thermo_success"])]))
        if thermo_success
        else _nan()
    )
    thermo_metastable_rate = (
        float(
            np.mean(
                [bool(row["metastable"]) for row in records if bool(row["thermo_success"])]
            )
        )
        if thermo_success
        else _nan()
    )
    un_rate = (
        float(np.mean([bool(row["unique"]) and bool(row["novel"]) for row in records]))
        if records
        else _nan()
    )
    thermo_summary = compute_sun_msun_from_thermo_rates(
        un_rate=un_rate if math.isfinite(un_rate) else None,
        thermo_metrics={
            "eval/thermo_stable_rate": thermo_stable_rate,
            "eval/thermo_metastable_rate": thermo_metastable_rate,
        },
        thermo_tag="eval",
    )

    def frac(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in records])) if records else _nan()

    summary = {
        "group": group.group,
        "guidance_factor": group.guidance_factor,
        "target": group.target,
        "n_structures": n_structures,
        "n_readable": n_readable,
        "n_struct_valid": n_struct_valid,
        "n_predicted": len(predicted),
        "mean_predicted_dielectric_scalar": float(np.mean(predicted)) if predicted else _nan(),
        "median_predicted_dielectric_scalar": float(np.median(predicted)) if predicted else _nan(),
        "std_predicted_dielectric_scalar": float(np.std(predicted, ddof=1)) if len(predicted) > 1 else _nan(),
        "mae": float(np.mean(np.abs(errors))) if errors else _nan(),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else _nan(),
        "frac_within_1": float(np.mean([abs(err) <= 1.0 for err in errors])) if errors else _nan(),
        "frac_within_2": float(np.mean([abs(err) <= 2.0 for err in errors])) if errors else _nan(),
        "composition_valid_fraction": frac("composition_valid"),
        "structure_valid_fraction": frac("structure_valid"),
        "structure_comp_valid_fraction": frac("structure_comp_valid"),
        "metal_like_proxy_fraction": frac("metal_like_proxy"),
        "min_distance_median": float(np.median(min_distances)) if min_distances else _nan(),
        "volume_per_atom_median": float(np.median(volume_per_atom)) if volume_per_atom else _nan(),
        "un_rate": un_rate,
        "thermo_checked": thermo_checked,
        "thermo_success": thermo_success,
        "thermo_failed": thermo_failed,
        "thermo_success_rate": (
            float(thermo_success) / float(thermo_checked) if thermo_checked > 0 else _nan()
        ),
        "thermo_stable_rate": thermo_stable_rate,
        "thermo_metastable_rate": thermo_metastable_rate,
        "thermo_e_above_hull_mean": float(np.mean(thermo_values)) if thermo_values else _nan(),
        "SUN": float(thermo_summary.get("SUN", _nan())),
        "MSUN": float(thermo_summary.get("MSUN", _nan())),
    }
    return {col: summary[col] for col in SUMMARY_COLUMNS}


def _write_outputs(
    output_dir: Path,
    per_sample_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    per_sample = pd.DataFrame(per_sample_rows)
    per_sample = per_sample.reindex(columns=PER_SAMPLE_COLUMNS)
    summary = pd.DataFrame(summary_rows)
    summary = summary.reindex(columns=SUMMARY_COLUMNS)

    per_sample.to_csv(output_dir / "per_sample_evaluation.csv", index=False)
    summary.to_csv(output_dir / "group_summary.csv", index=False)
    thermo_cols = [
        "group",
        "guidance_factor",
        "target",
        "un_rate",
        "thermo_checked",
        "thermo_success",
        "thermo_failed",
        "thermo_success_rate",
        "thermo_stable_rate",
        "thermo_metastable_rate",
        "thermo_e_above_hull_mean",
        "SUN",
        "MSUN",
    ]
    summary.reindex(columns=thermo_cols).to_csv(
        output_dir / "thermo_group_summary.csv",
        index=False,
    )
    with open(output_dir / "group_summary.json", "w") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline MatterGen-style evaluation for existing Crystalite sample CIFs."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--anisonet_checkpoint", type=Path, default=DEFAULT_ANISONET_CKPT)
    parser.add_argument("--anisonet_batch_size", type=int, default=8)
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help=(
            "Custom sample group as group,guidance_factor,target,path. "
            "May be repeated. Defaults to the legacy dielectric CFG groups."
        ),
    )
    parser.add_argument(
        "--anisonet_require_structure_valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Predict only readable CIFs that pass Crystalite structure validity. "
            "Invalid readable CIFs are retained with prediction_success=False."
        ),
    )
    parser.add_argument(
        "--skip_anisonet",
        action="store_true",
        help="Skip AnisoNet dielectric prediction and write structural metrics only.",
    )
    parser.add_argument(
        "--skip_novelty",
        action="store_true",
        help="Skip train-set structure loading and uniqueness/novelty matching.",
    )
    parser.add_argument(
        "--thermo_ppd_mp",
        type=Path,
        default=None,
        help="PatchedPhaseDiagram pickle used for e_above_hull / SUN / MSUN.",
    )
    parser.add_argument(
        "--thermo_backend",
        type=str,
        default="chgnet",
        choices=["chgnet"],
        help="Thermo backend for offline e_above_hull evaluation on existing relaxed CIFs.",
    )
    parser.add_argument(
        "--thermo_device",
        type=str,
        default="cpu",
        help="Device for CHGNet thermo energy prediction. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--thermo_ehull_method",
        type=str,
        default="mp2020_like",
        choices=["uncorrected", "mp2020_like"],
        help="How to compute e_above_hull from the supplied phase diagram pickle.",
    )
    return parser.parse_args()


def _parse_group_specs(specs: list[str]) -> list[SampleGroup]:
    groups: list[SampleGroup] = []
    for spec in specs:
        parts = [part.strip() for part in spec.split(",", 3)]
        if len(parts) != 4 or not parts[0] or not parts[3]:
            raise ValueError(
                "--group must have format group,guidance_factor,target,path; "
                f"got {spec!r}"
            )
        groups.append(SampleGroup(parts[0], parts[1], parts[2], Path(parts[3])))
    return groups


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    groups = _parse_group_specs(args.group) if args.group else DEFAULT_GROUPS
    thermo = _build_thermo_context(args)

    train_structures: list[Structure] = []
    if args.skip_novelty:
        print("Skipping novelty/uniqueness matching.")
    else:
        print("Loading AnisoNet train split for novelty reference...")
        train_structures = _load_train_structures(args.train_csv)
        print(f"Loaded {len(train_structures)} reference train structures.")
    if thermo is not None:
        print(
            "Thermo evaluation enabled: "
            f"backend={thermo.backend} "
            f"ehull_method={thermo.ehull_method} "
            f"device_requested={thermo.device_requested} "
            f"device_actual={thermo.device_actual}"
        )

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for group in groups:
        print(f"Evaluating group={group.group} path={group.path}")
        records, structures = _load_group_records(group)
        if not args.skip_novelty:
            _set_uniqueness_and_novelty(records, structures, train_structures)
        if thermo is not None:
            _compute_thermo_metrics_for_group(records, structures, thermo=thermo)
        if not args.skip_anisonet:
            _predict_with_anisonet(
                records,
                structures,
                checkpoint_path=args.anisonet_checkpoint,
                batch_size=args.anisonet_batch_size,
                require_structure_valid=args.anisonet_require_structure_valid,
            )
        all_rows.extend(records)
        summary = _summarize_group(records, group)
        summary_rows.append(summary)
        print(
            "  "
            f"n={summary['n_structures']} predicted={summary['n_predicted']} "
            f"comp_valid={summary['composition_valid_fraction']:.3f} "
            f"struct_valid={summary['structure_valid_fraction']:.3f} "
            f"metal_like={summary['metal_like_proxy_fraction']:.3f} "
            f"thermo_success={summary['thermo_success']} "
            f"SUN={summary['SUN'] if math.isfinite(float(summary['SUN'])) else float('nan'):.3f}"
        )

    _write_outputs(args.output_dir, all_rows, summary_rows)
    print(f"Wrote per-sample CSV: {args.output_dir / 'per_sample_evaluation.csv'}")
    print(
        "Wrote summary CSV: "
        f"{args.output_dir / 'group_summary.csv'}"
    )
    print(
        "Wrote summary JSON: "
        f"{args.output_dir / 'group_summary.json'}"
    )
    print(f"Wrote thermo summary CSV: {args.output_dir / 'thermo_group_summary.csv'}")


if __name__ == "__main__":
    main()
