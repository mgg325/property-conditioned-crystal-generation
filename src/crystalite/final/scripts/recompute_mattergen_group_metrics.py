from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import ase.io
import numpy as np
from monty.serialization import dumpfn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.metrics.evaluator import MetricsEvaluator

from run_mattergen_eval_simple import (
    EXCLUDED_METRIC_NAMES,
    _compute_timed_smact_metrics,
    _load_reference,
    _make_compatibility,
    _make_structure_matcher,
    load_supported_terminals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute MatterGen metrics for an existing group output by refiltering "
            "already-relaxed samples and reusing the saved MatterSim outputs."
        )
    )
    parser.add_argument("--group_dir", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument(
        "--reference_dataset_path",
        type=Path,
        required=True,
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
        "--max_terminal_elements",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--smact_timeout_seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--smact_jobs",
        type=int,
        default=1,
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    group_dir = args.group_dir
    metrics_path = group_dir / "metrics.json"
    detailed_path = group_dir / "detailed.json"
    extxyz_path = group_dir / "mattersim_relaxed.extxyz"
    energies_path = group_dir / "mattersim_relaxed_energies.npy"
    summary_path = group_dir / "group_summary.json"
    skipped_csv_path = group_dir / "skipped_samples.csv"
    skipped_json_path = group_dir / "skipped_samples.json"
    successful_input_dir = group_dir / "successful_input_cifs"
    successful_relaxed_dir = group_dir / "successful_relaxed_cifs"

    original_sample_paths = sorted(successful_input_dir.glob("*.cif"))
    if not original_sample_paths:
        raise FileNotFoundError(f"No successful input CIFs found under {successful_input_dir}")

    relaxed_atoms_all = ase.io.read(extxyz_path, ":")
    energies_all = np.load(energies_path)
    if len(relaxed_atoms_all) != len(original_sample_paths) or len(energies_all) != len(original_sample_paths):
        raise ValueError("Existing extxyz/energies do not align with successful_input_cifs.")

    supported_terminals = load_supported_terminals()
    kept_rows: list[tuple[Path, object, float, Structure, list[str]]] = []
    new_skipped_rows: list[dict[str, object]] = []
    for cif_path, relaxed_atom, energy in zip(original_sample_paths, relaxed_atoms_all, energies_all):
        structure = Structure.from_file(cif_path)
        terminals = sorted({site.specie.symbol for site in structure})
        if len(terminals) > int(args.max_terminal_elements):
            new_skipped_rows.append(
                {
                    "group": group_dir.name,
                    "sample_file": cif_path.name,
                    "reason": "too_many_terminal_elements_for_energy_metrics",
                    "num_terminal_elements": len(terminals),
                    "max_terminal_elements": int(args.max_terminal_elements),
                    "terminals": ",".join(terminals),
                    "unsupported_terminals": "",
                    "relax_error": "",
                }
            )
            continue
        unsupported = [symbol for symbol in terminals if symbol not in supported_terminals]
        if unsupported:
            new_skipped_rows.append(
                {
                    "group": group_dir.name,
                    "sample_file": cif_path.name,
                    "reason": "unsupported_terminal_in_mattergen_reference",
                    "num_terminal_elements": len(terminals),
                    "max_terminal_elements": int(args.max_terminal_elements),
                    "terminals": ",".join(terminals),
                    "unsupported_terminals": ",".join(unsupported),
                    "relax_error": "",
                }
            )
            continue
        kept_rows.append((cif_path, relaxed_atom, float(energy), structure, terminals))

    existing_skipped: list[dict[str, object]] = []
    if skipped_json_path.exists():
        existing_skipped = json.loads(skipped_json_path.read_text())
    skipped_by_sample = {row["sample_file"]: row for row in existing_skipped}
    for row in new_skipped_rows:
        skipped_by_sample[row["sample_file"]] = row
    merged_skipped_rows = [skipped_by_sample[key] for key in sorted(skipped_by_sample.keys())]
    skipped_json_path.write_text(json.dumps(merged_skipped_rows, indent=2))
    if merged_skipped_rows:
        _write_csv(merged_skipped_rows, skipped_csv_path)

    temp_input_dir = group_dir / "successful_input_cifs.refiltered"
    temp_relaxed_dir = group_dir / "successful_relaxed_cifs.refiltered"
    for path in (temp_input_dir, temp_relaxed_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    filtered_original_structures: list[Structure] = []
    filtered_relaxed_structures: list[Structure] = []
    filtered_relaxed_atoms: list[object] = []
    filtered_energies: list[float] = []
    sample_files: list[str] = []

    for cif_path, relaxed_atom, energy, structure, _ in kept_rows:
        shutil.copy2(cif_path, temp_input_dir / cif_path.name)
        relaxed_structure = AseAtomsAdaptor.get_structure(relaxed_atom)
        ase.io.write(temp_relaxed_dir / cif_path.name, relaxed_atom, format="cif")
        filtered_original_structures.append(structure)
        filtered_relaxed_structures.append(relaxed_structure)
        filtered_relaxed_atoms.append(relaxed_atom)
        filtered_energies.append(float(energy))
        sample_files.append(cif_path.name)

    ase.io.write(extxyz_path, filtered_relaxed_atoms, format="extxyz")
    np.save(energies_path, np.array(filtered_energies, dtype=float))
    if successful_input_dir.exists():
        shutil.rmtree(successful_input_dir)
    if successful_relaxed_dir.exists():
        shutil.rmtree(successful_relaxed_dir)
    temp_input_dir.rename(successful_input_dir)
    temp_relaxed_dir.rename(successful_relaxed_dir)

    reference = _load_reference(args.reference_dataset_path, structure_matcher=args.structure_matcher)
    compatibility = _make_compatibility(args.energy_correction_scheme)
    structure_matcher = _make_structure_matcher(args.structure_matcher)
    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=filtered_relaxed_structures,
        energies=np.array(filtered_energies, dtype=float),
        original_structures=filtered_original_structures,
        reference=reference,
        structure_matcher=structure_matcher,
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
        filtered_relaxed_structures,
        timeout_seconds=float(args.smact_timeout_seconds),
        jobs=int(args.smact_jobs),
    )
    metrics_payload.update(smact_metrics)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    df = evaluator.as_dataframe(metrics=selected_metrics, save_as=None)
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

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    relaxed_successful_before_filter = summary.get("successful_input_cifs", len(original_sample_paths))
    summary.update(
        {
            "successful_input_cifs": len(sample_files),
            "skipped_input_cifs": len(merged_skipped_rows),
            "skipped_high_order_cifs": sum(
                row.get("reason") == "too_many_terminal_elements_for_energy_metrics"
                for row in merged_skipped_rows
            ),
            "skipped_unsupported_terminal_cifs": sum(
                row.get("reason") == "unsupported_terminal_in_mattergen_reference"
                for row in merged_skipped_rows
            ),
            "max_terminal_elements": int(args.max_terminal_elements),
            "relaxed_successful_cifs_before_reference_filter": relaxed_successful_before_filter,
            "metrics_recomputed_from_existing_relax": True,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    with args.summary_csv.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("group") != group_dir.name:
            continue
        for key, value in summary.items():
            row[key] = value
        for key, value in metrics_payload.items():
            row[key] = value["value"]
        row["metrics_json"] = str(metrics_path)
        row["detailed_json"] = str(detailed_path)
        row["relaxed_extxyz"] = str(extxyz_path)
        row["energies_path"] = str(energies_path)
        break
    _write_csv(rows, args.summary_csv)

    print(f"[updated] group={group_dir.name}")
    print(f"[updated] kept={len(sample_files)} skipped={len(merged_skipped_rows)}")
    print(f"[updated] metrics keys={sorted(metrics_payload.keys())}")


if __name__ == "__main__":
    main()
