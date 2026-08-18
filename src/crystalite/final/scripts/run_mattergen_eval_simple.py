from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import ase.io
import numpy as np
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from monty.serialization import dumpfn
from pymatgen.core import Element, Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.metrics.structure import is_smact_valid, structure_validity
from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)


ROOT = Path(__file__).resolve().parents[1]
MATTERGEN_ROOT = Path("external/mattergen")
DEFAULT_REFERENCE = (
    MATTERGEN_ROOT / "data-release" / "alex-mp" / "reference_TRI2024correction.gz"
)
DEFAULT_SMACT_TIMEOUT_SECONDS = 2.0
DEFAULT_SMACT_JOBS = min(8, max(1, (os.cpu_count() or 1) - 1))
EXCLUDED_METRIC_NAMES = {
    "avg_comp_validity",
    "avg_structure_comp_validity",
}


@dataclass(frozen=True)
class GroupSpec:
    group: str
    guidance_factor: str
    target: str
    cifs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple MatterGen evaluation from CHGNet-relaxed CIF groups without "
            "pre-filtering elements or chunking relaxation."
        )
    )
    parser.add_argument("--method", required=True, help="Method prefix, e.g. fixed_film.")
    parser.add_argument(
        "--relaxed_root",
        type=Path,
        default=ROOT / "outputs" / "final report output" / "relaxed_groups",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=ROOT / "outputs" / "final report output" / "mattergen_eval_fixed_film_simple",
    )
    parser.add_argument(
        "--reference_dataset_path",
        type=Path,
        default=DEFAULT_REFERENCE,
    )
    parser.add_argument(
        "--energy_correction_scheme",
        type=str,
        default="TRI2024",
        choices=["MP2020", "TRI2024"],
    )
    parser.add_argument(
        "--structure_matcher",
        type=str,
        default="disordered",
        choices=["ordered", "disordered"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for MatterSim relaxation.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional exact group names to run.",
    )
    parser.add_argument(
        "--start_group",
        type=str,
        default=None,
        help="Optional exact group name to start from, inclusive.",
    )
    parser.add_argument(
        "--limit_groups",
        type=int,
        default=None,
        help="Optional cap on number of groups to run.",
    )
    parser.add_argument(
        "--limit_cifs",
        type=int,
        default=None,
        help="Optional cap on CIFs per group for smoke tests.",
    )
    parser.add_argument(
        "--relax_timeout_seconds",
        type=float,
        default=180.0,
        help="Per-CIF timeout for MatterSim relaxation. Non-positive disables the timeout.",
    )
    parser.add_argument(
        "--max_terminal_elements",
        type=int,
        default=8,
        help=(
            "Skip samples with more than this many distinct terminal elements before "
            "MatterSim relax/evaluation. This avoids pathological energy-above-hull runs."
        ),
    )
    parser.add_argument(
        "--smact_timeout_seconds",
        type=float,
        default=DEFAULT_SMACT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--smact_jobs",
        type=int,
        default=DEFAULT_SMACT_JOBS,
    )
    return parser.parse_args()


def _parse_group(group_name: str, cifs_dir: Path) -> GroupSpec:
    parts = group_name.split("__")
    if len(parts) == 2 and parts[1] == "uncond":
        return GroupSpec(group=group_name, guidance_factor="0", target="uncond", cifs_dir=cifs_dir)
    if len(parts) != 3:
        raise ValueError(f"Unexpected group format: {group_name}")
    return GroupSpec(
        group=group_name,
        guidance_factor=parts[1].removeprefix("g"),
        target=parts[2].removeprefix("t"),
        cifs_dir=cifs_dir,
    )


def discover_groups(relaxed_root: Path, method: str) -> list[GroupSpec]:
    groups: list[GroupSpec] = []
    for path in sorted(relaxed_root.glob(f"{method}__*")):
        cifs_dir = path / "cifs"
        if cifs_dir.exists():
            groups.append(_parse_group(path.name, cifs_dir))
    return sorted(
        groups,
        key=lambda spec: (
            0 if spec.target == "uncond" else 1,
            int(spec.guidance_factor),
            -1 if spec.target == "uncond" else int(spec.target),
        ),
    )


def load_supported_terminals() -> set[str]:
    return {
        element.symbol
        for element in Element
        if element.Z < 84
        and element.symbol not in {"Tc", "Pm"}
        and not element.is_noble_gas
    }


def _timeout_handler(signum, frame):
    raise TimeoutError("MatterSim relaxation timed out")


def _timed_smact_worker(structure_dict: dict, timeout_seconds: float) -> dict[str, object]:
    structure = Structure.from_dict(structure_dict)
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        comp_valid = bool(is_smact_valid(structure))
        struct_valid = bool(structure_validity(structure))
        return {
            "status": "ok",
            "comp_validity": comp_valid,
            "structure_comp_validity": comp_valid and struct_valid,
        }
    except TimeoutError:
        return {
            "status": "timeout",
            "comp_validity": None,
            "structure_comp_validity": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "comp_validity": None,
            "structure_comp_validity": None,
            "error": repr(exc),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _compute_timed_smact_metrics(
    structures: list[Structure],
    *,
    timeout_seconds: float,
    jobs: int,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    structure_dicts = [structure.as_dict() for structure in structures]
    results: list[dict[str, object]] = [None] * len(structure_dicts)  # type: ignore[assignment]

    with ProcessPoolExecutor(
        max_workers=max(1, int(jobs)),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(_timed_smact_worker, structure_dict, float(timeout_seconds))
            for structure_dict in structure_dicts
        ]
        for idx, future in enumerate(futures):
            results[idx] = future.result()

    ok_results = [row for row in results if row["status"] == "ok"]
    timeout_count = sum(row["status"] == "timeout" for row in results)
    error_count = sum(row["status"] == "error" for row in results)
    total = len(results)
    denominator = len(ok_results)

    avg_comp_validity = (
        float(np.mean([row["comp_validity"] for row in ok_results])) if denominator else None
    )
    avg_structure_comp_validity = (
        float(np.mean([row["structure_comp_validity"] for row in ok_results]))
        if denominator
        else None
    )

    metrics = {
        "avg_comp_validity": {
            "value": avg_comp_validity,
            "description": (
                "Average composition validity (according to smact) of successfully checked "
                "structures in sampled data. Timeout and error samples are excluded from the denominator."
            ),
        },
        "avg_comp_validity_timeout_rate": {
            "value": float(timeout_count / total) if total else 0.0,
            "description": "Fraction of structures whose SMACT composition validity check timed out.",
        },
        "avg_comp_validity_error_rate": {
            "value": float(error_count / total) if total else 0.0,
            "description": "Fraction of structures whose SMACT composition validity check errored.",
        },
        "avg_structure_comp_validity": {
            "value": avg_structure_comp_validity,
            "description": (
                "Average number of structures in sampled data that are both structurally valid "
                "and have a valid smact composition, over successfully checked samples only."
            ),
        },
        "avg_structure_comp_validity_timeout_rate": {
            "value": float(timeout_count / total) if total else 0.0,
            "description": (
                "Fraction of structures whose SMACT-backed structure+composition validity check timed out."
            ),
        },
        "avg_structure_comp_validity_error_rate": {
            "value": float(error_count / total) if total else 0.0,
            "description": (
                "Fraction of structures whose SMACT-backed structure+composition validity check errored."
            ),
        },
        "smact_timeout_seconds": {
            "value": float(timeout_seconds),
            "description": "Per-structure timeout used for SMACT validity checks in seconds.",
        },
        "smact_checked_samples": {
            "value": int(denominator),
            "description": "Number of structures whose SMACT validity check completed successfully.",
        },
    }
    return metrics, results


def _make_relaxer(*, device: str) -> BatchRelaxer:
    potential = Potential.from_checkpoint(
        device=device,
        load_training_state=False,
    )
    return BatchRelaxer(potential=potential, filter="EXPCELLFILTER")


def _relax_one_structure(
    relaxer: BatchRelaxer,
    structure: Structure,
    *,
    timeout_seconds: float,
) -> tuple[Structure, object, float]:
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    if timeout_seconds > 0:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        trajectories = relaxer.relax([atoms])
        relaxed_atoms = next(iter(trajectories.values()))[-1]
        relaxed_structure = AseAtomsAdaptor.get_structure(relaxed_atoms)
        energy = float(relaxed_atoms.info["total_energy"])
        return relaxed_structure, relaxed_atoms, energy
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _make_structure_matcher(name: Literal["ordered", "disordered"]):
    if name == "disordered":
        return DefaultDisorderedStructureMatcher()
    return DefaultOrderedStructureMatcher()


def _make_compatibility(name: Literal["MP2020", "TRI2024"]):
    if name == "MP2020":
        return MaterialsProject2020Compatibility()
    return TRI110Compatibility2024()


def _resolve_metric_name(metric_cls: type) -> str | None:
    name_attr = getattr(metric_cls, "name", None)
    if isinstance(name_attr, str):
        return name_attr
    # Some metric classes expose name only on the instance.
    return None


def _load_reference(reference_dataset_path: Path, *, structure_matcher: str):
    reference = LMDBGZSerializer().deserialize(reference_dataset_path)
    if structure_matcher == "disordered":
        reference.__dict__["is_ordered"] = False
    return reference


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def run_group(
    spec: GroupSpec,
    *,
    output_dir: Path,
    reference,
    compatibility,
    structure_matcher: str,
    relax_timeout_seconds: float,
    smact_timeout_seconds: float,
    smact_jobs: int,
    device: str,
    limit_cifs: int | None,
    max_terminal_elements: int,
    supported_terminals: set[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    detailed_path = output_dir / "detailed.json"
    extxyz_path = output_dir / "mattersim_relaxed.extxyz"
    energies_path = output_dir / "mattersim_relaxed_energies.npy"
    summary_path = output_dir / "group_summary.json"
    skipped_csv_path = output_dir / "skipped_samples.csv"
    skipped_json_path = output_dir / "skipped_samples.json"
    successful_input_dir = output_dir / "successful_input_cifs"
    successful_relaxed_dir = output_dir / "successful_relaxed_cifs"
    relax_results_csv = output_dir / "relax_results.csv"
    relax_results_json = output_dir / "relax_results.json"

    if (
        metrics_path.exists()
        and detailed_path.exists()
        and extxyz_path.exists()
        and energies_path.exists()
        and summary_path.exists()
    ):
        summary = json.loads(summary_path.read_text())
        row = {
            "group": spec.group,
            "guidance_factor": spec.guidance_factor,
            "target": spec.target,
            "structures_path": str(spec.cifs_dir),
            "output_dir": str(output_dir),
            "elapsed_seconds": "",
            "returncode": 0,
            "metrics_json": str(metrics_path),
            "detailed_json": str(detailed_path),
            "relaxed_extxyz": str(extxyz_path),
            "energies_path": str(energies_path),
            "reused_existing": True,
        }
        row.update(summary)
        row.update({k: v["value"] for k, v in json.loads(metrics_path.read_text()).items()})
        return row

    for path in (
        successful_input_dir,
        successful_relaxed_dir,
    ):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        metrics_path,
        detailed_path,
        extxyz_path,
        energies_path,
        summary_path,
        skipped_csv_path,
        skipped_json_path,
        relax_results_csv,
        relax_results_json,
    ):
        if path.exists():
            path.unlink()

    cif_paths = sorted(spec.cifs_dir.glob("*.cif"))
    if limit_cifs is not None:
        cif_paths = cif_paths[: int(limit_cifs)]

    original_structures: list[Structure] = []
    relaxed_structures: list[Structure] = []
    relaxed_atoms: list[object] = []
    energies: list[float] = []
    sample_files: list[str] = []
    relax_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    total_relax_seconds = 0.0
    started_group = time.monotonic()
    relaxer = _make_relaxer(device=device)
    skipped_high_order = 0
    skipped_unsupported_terminals = 0

    for cif_path in cif_paths:
        sample_name = cif_path.name
        input_structure: Structure | None = None
        relaxed_cif_path = successful_relaxed_dir / sample_name
        relax_success = False
        relax_error = ""
        relax_started = time.monotonic()
        try:
            input_structure = Structure.from_file(cif_path)
            terminals = sorted({site.specie.symbol for site in input_structure})
            if len(terminals) > int(max_terminal_elements):
                skipped_high_order += 1
                skipped_rows.append(
                    {
                        "group": spec.group,
                        "sample_file": sample_name,
                        "reason": "too_many_terminal_elements_for_energy_metrics",
                        "num_terminal_elements": len(terminals),
                        "max_terminal_elements": int(max_terminal_elements),
                        "terminals": ",".join(terminals),
                        "relax_error": "",
                    }
                )
                relax_error = (
                    "Skipped before MatterSim relaxation because the sample has "
                    f"{len(terminals)} terminal elements (threshold {int(max_terminal_elements)})."
                )
                continue
            unsupported = [symbol for symbol in terminals if symbol not in supported_terminals]
            if unsupported:
                skipped_unsupported_terminals += 1
                skipped_rows.append(
                    {
                        "group": spec.group,
                        "sample_file": sample_name,
                        "reason": "unsupported_terminal_in_mattergen_reference",
                        "num_terminal_elements": len(terminals),
                        "max_terminal_elements": int(max_terminal_elements),
                        "terminals": ",".join(terminals),
                        "unsupported_terminals": ",".join(unsupported),
                        "relax_error": "",
                    }
                )
                relax_error = (
                    "Skipped before MatterSim relaxation because the sample has "
                    f"unsupported terminals for MatterGen reference: {','.join(unsupported)}."
                )
                continue
            relaxed_structure, relaxed_atom, energy = _relax_one_structure(
                relaxer,
                input_structure,
                timeout_seconds=float(relax_timeout_seconds),
            )
            shutil.copy2(cif_path, successful_input_dir / sample_name)
            relaxed_cif_path.write_text(str(CifWriter(relaxed_structure)))
            original_structures.append(input_structure)
            relaxed_structures.append(relaxed_structure)
            relaxed_atoms.append(relaxed_atom)
            energies.append(float(energy))
            sample_files.append(sample_name)
            relax_success = True
        except Exception as exc:
            relax_error = f"{type(exc).__name__}: {exc}"
            skipped_rows.append(
                {
                    "group": spec.group,
                    "sample_file": sample_name,
                    "reason": "relax_failed",
                    "num_terminal_elements": len(terminals) if input_structure is not None else "",
                    "max_terminal_elements": int(max_terminal_elements),
                    "terminals": ",".join(terminals) if input_structure is not None else "",
                    "unsupported_terminals": "",
                    "relax_error": relax_error,
                }
            )
            # Reset the relaxer after a failed sample.
            relaxer = _make_relaxer(device=device)
        relax_elapsed = time.monotonic() - relax_started
        total_relax_seconds += relax_elapsed
        relax_rows.append(
            {
                "group": spec.group,
                "sample_file": sample_name,
                "input_cif_path": str(cif_path),
                "relaxed_cif_path": str(relaxed_cif_path) if relax_success else "",
                "relax_success": relax_success,
                "relax_seconds": round(relax_elapsed, 6),
                "relax_error": relax_error,
            }
        )

    _write_csv(relax_rows, relax_results_csv)
    relax_results_json.write_text(json.dumps(relax_rows, indent=2))
    skipped_json_path.write_text(json.dumps(skipped_rows, indent=2))
    if skipped_rows:
        _write_csv(skipped_rows, skipped_csv_path)

    group_elapsed = time.monotonic() - started_group
    summary: dict[str, object] = {
        "group": spec.group,
        "guidance_factor": spec.guidance_factor,
        "target": spec.target,
        "total_input_cifs": len(cif_paths),
        "successful_input_cifs": len(sample_files),
        "skipped_input_cifs": len(skipped_rows),
        "skipped_high_order_cifs": skipped_high_order,
        "skipped_unsupported_terminal_cifs": skipped_unsupported_terminals,
        "relax_timeout_seconds": float(relax_timeout_seconds),
        "max_terminal_elements": int(max_terminal_elements),
        "total_relax_seconds": round(total_relax_seconds, 6),
        "avg_relax_seconds_per_input": (
            round(total_relax_seconds / len(cif_paths), 6) if cif_paths else 0.0
        ),
        "avg_relax_seconds_per_success": (
            round(total_relax_seconds / len(sample_files), 6) if sample_files else None
        ),
    }

    if not sample_files:
        summary["group_elapsed_seconds"] = round(group_elapsed, 6)
        summary_path.write_text(json.dumps(summary, indent=2))
        return {
            "group": spec.group,
            "guidance_factor": spec.guidance_factor,
            "target": spec.target,
            "structures_path": str(spec.cifs_dir),
            "output_dir": str(output_dir),
            "elapsed_seconds": round(group_elapsed, 6),
            "returncode": 1,
            "metrics_json": "",
            "detailed_json": "",
            "relaxed_extxyz": "",
            "energies_path": "",
            "reused_existing": False,
            **summary,
        }

    ase.io.write(extxyz_path, relaxed_atoms, format="extxyz")
    np.save(energies_path, np.array(energies, dtype=float))

    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=np.array(energies, dtype=float),
        original_structures=original_structures,
        reference=reference,
        structure_matcher=_make_structure_matcher(
            structure_matcher  # type: ignore[arg-type]
        ),
        energy_correction_scheme=compatibility,
    )

    selected_metrics = []
    for metric_cls in evaluator.available_metrics:
        metric = evaluator._get_metric(metric_cls)
        if metric.name in EXCLUDED_METRIC_NAMES:
            continue
        selected_metrics.append(metric_cls)
    selected_metric_values = evaluator.compute_metrics(
        metrics=selected_metrics,
        save_as=None,
        pretty_print=True,
    )
    for excluded_name in EXCLUDED_METRIC_NAMES:
        selected_metric_values.pop(excluded_name, None)

    metrics_payload: dict[str, dict[str, object]] = {}
    for metric_cls in selected_metrics:
        metric = evaluator._get_metric(metric_cls)
        metric_name = metric.name
        if metric_name in EXCLUDED_METRIC_NAMES or metric_name not in selected_metric_values:
            continue
        metrics_payload[metric_name] = {
            "value": selected_metric_values[metric_name],
            "description": metric.description,
        }

    smact_metrics, smact_rows = _compute_timed_smact_metrics(
        relaxed_structures,
        timeout_seconds=float(smact_timeout_seconds),
        jobs=int(smact_jobs),
    )
    metrics_payload.update(smact_metrics)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    df = evaluator.as_dataframe(
        metrics=selected_metrics,
        save_as=None,
    )
    drop_cols = [col for col in ("comp_validity", "structure_comp_validity") if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df["sample_file"] = sample_files
    df["comp_validity"] = [row["comp_validity"] for row in smact_rows]
    df["structure_comp_validity"] = [row["structure_comp_validity"] for row in smact_rows]
    df["smact_status"] = [row["status"] for row in smact_rows]
    df["smact_timeout"] = [row["status"] == "timeout" for row in smact_rows]
    df["smact_error"] = [row["status"] == "error" for row in smact_rows]
    df["smact_error_detail"] = [row.get("error", "") for row in smact_rows]
    dumpfn(df.to_dict("list"), detailed_path)

    summary["group_elapsed_seconds"] = round(group_elapsed, 6)
    summary_path.write_text(json.dumps(summary, indent=2))

    row = {
        "group": spec.group,
        "guidance_factor": spec.guidance_factor,
        "target": spec.target,
        "structures_path": str(spec.cifs_dir),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(group_elapsed, 6),
        "returncode": 0,
        "metrics_json": str(metrics_path),
        "detailed_json": str(detailed_path),
        "relaxed_extxyz": str(extxyz_path),
        "energies_path": str(energies_path),
        "reused_existing": False,
        **summary,
    }
    row.update({key: value["value"] for key, value in metrics_payload.items()})
    return row


def main() -> None:
    args = parse_args()
    groups = discover_groups(args.relaxed_root, args.method)
    if args.start_group:
        try:
            start_index = next(idx for idx, spec in enumerate(groups) if spec.group == args.start_group)
        except StopIteration as exc:
            raise FileNotFoundError(f"start_group not found: {args.start_group}") from exc
        groups = groups[start_index:]
    if args.groups:
        allowed = set(args.groups)
        groups = [spec for spec in groups if spec.group in allowed]
    if args.limit_groups is not None:
        groups = groups[: int(args.limit_groups)]
    if not groups:
        raise FileNotFoundError(f"No groups found for method={args.method}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.csv"
    rows_by_group: dict[str, dict[str, object]] = {
        row["group"]: row for row in _load_existing_rows(summary_path) if row.get("group")
    }

    reference = _load_reference(args.reference_dataset_path, structure_matcher=args.structure_matcher)
    compatibility = _make_compatibility(args.energy_correction_scheme)
    supported_terminals = load_supported_terminals()

    for spec in groups:
        group_output_dir = args.output_root / spec.group
        print(f"[mattergen-simple] group={spec.group}")
        row = run_group(
            spec,
            output_dir=group_output_dir,
            reference=reference,
            compatibility=compatibility,
            structure_matcher=args.structure_matcher,
            relax_timeout_seconds=args.relax_timeout_seconds,
            smact_timeout_seconds=args.smact_timeout_seconds,
            smact_jobs=args.smact_jobs,
            device=args.device,
            limit_cifs=args.limit_cifs,
            max_terminal_elements=args.max_terminal_elements,
            supported_terminals=supported_terminals,
        )
        rows_by_group[spec.group] = row
        ordered_rows = [rows_by_group[key] for key in sorted(rows_by_group.keys())]
        _write_csv(ordered_rows, summary_path)
        if int(row["returncode"]) != 0:
            print(f"[mattergen-simple] failed group={spec.group}")
            break

    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "relaxed_root": str(args.relaxed_root),
                "reference_dataset_path": str(args.reference_dataset_path),
                "energy_correction_scheme": args.energy_correction_scheme,
                "structure_matcher": args.structure_matcher,
                "device": args.device,
                "relax_timeout_seconds": args.relax_timeout_seconds,
                "max_terminal_elements": args.max_terminal_elements,
                "smact_timeout_seconds": args.smact_timeout_seconds,
                "smact_jobs": args.smact_jobs,
                "groups": [spec.group for spec in groups],
            },
            indent=2,
        )
    )
    print(f"[done] output={args.output_root}")


if __name__ == "__main__":
    main()
