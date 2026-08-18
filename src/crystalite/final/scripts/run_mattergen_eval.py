from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Element, Structure


ROOT = Path(__file__).resolve().parents[1]
MATTERGEN_ROOT = Path("external/mattergen")
MATTERGEN_PYTHON = MATTERGEN_ROOT / ".venv" / "bin" / "python"
MATTERGEN_EVAL = ROOT / "scripts" / "mattergen_eval_selected_metrics.py"
MATTERGEN_RELAX_ONLY = ROOT / "scripts" / "mattergen_relax_only.py"
DEFAULT_REFERENCE = (
    MATTERGEN_ROOT / "data-release" / "alex-mp" / "reference_TRI2024correction.gz"
)


@dataclass(frozen=True)
class GroupSpec:
    group: str
    guidance_factor: str
    target: str
    cifs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MatterGen evaluation over existing relaxed CIF groups."
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
        default=ROOT / "outputs" / "final report output",
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
        "--limit_groups",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="Optional exact group names to run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to pass to MatterGen evaluation.",
    )
    parser.add_argument(
        "--relax_chunk_size",
        type=int,
        default=16,
        help="Number of supported CIFs to relax per MatterSim chunk before merging.",
    )
    return parser.parse_args()


def _parse_group(group_name: str, cifs_dir: Path) -> GroupSpec:
    parts = group_name.split("__")
    if len(parts) == 2 and parts[1] == "uncond":
        return GroupSpec(group=group_name, guidance_factor="0", target="uncond", cifs_dir=cifs_dir)
    if len(parts) != 3:
        raise ValueError(f"Unexpected group format: {group_name}")
    guidance_factor = parts[1].removeprefix("g")
    target = parts[2].removeprefix("t")
    return GroupSpec(
        group=group_name,
        guidance_factor=guidance_factor,
        target=target,
        cifs_dir=cifs_dir,
    )


def discover_groups(relaxed_root: Path, method: str) -> list[GroupSpec]:
    groups: list[GroupSpec] = []
    for path in sorted(relaxed_root.glob(f"{method}__*")):
        cifs_dir = path / "cifs"
        if not cifs_dir.exists():
            continue
        groups.append(_parse_group(path.name, cifs_dir))
    return sorted(
        groups,
        key=lambda spec: (
            0 if spec.target == "uncond" else 1,
            int(spec.guidance_factor),
            -1 if spec.target == "uncond" else int(spec.target),
        ),
    )


def load_supported_terminals(reference_dataset_path: Path) -> set[str]:
    _ = reference_dataset_path
    return {
        element.symbol
        for element in Element
        if element.Z < 84
        and element.symbol not in {"Tc", "Pm"}
        and not element.is_noble_gas
    }


def prepare_supported_input(
    spec: GroupSpec,
    *,
    output_dir: Path,
    supported_terminals: set[str],
) -> tuple[Path | None, list[dict[str, object]], int]:
    input_dir = output_dir / "input_cifs"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    skipped_rows: list[dict[str, object]] = []
    supported_count = 0
    total_count = 0

    for cif_path in sorted(spec.cifs_dir.glob("*.cif")):
        total_count += 1
        structure = Structure.from_file(cif_path)
        terminals = sorted({site.specie.symbol for site in structure})
        unsupported = [symbol for symbol in terminals if symbol not in supported_terminals]
        if unsupported:
            skipped_rows.append(
                {
                    "group": spec.group,
                    "sample_file": cif_path.name,
                    "unsupported_terminals": ",".join(unsupported),
                    "all_terminals": ",".join(terminals),
                    "reason": "unsupported_terminal_in_mattergen_reference",
                }
            )
            continue
        shutil.copy2(cif_path, input_dir / cif_path.name)
        supported_count += 1

    (output_dir / "skipped_samples.json").write_text(json.dumps(skipped_rows, indent=2))
    if skipped_rows:
        with (output_dir / "skipped_samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["group", "sample_file", "unsupported_terminals", "all_terminals", "reason"],
            )
            writer.writeheader()
            writer.writerows(skipped_rows)

    return (input_dir if supported_count > 0 else None), skipped_rows, total_count


def chunk_cif_paths(input_dir: Path, chunk_size: int) -> list[list[Path]]:
    cif_paths = sorted(input_dir.glob("*.cif"))
    return [
        cif_paths[idx : idx + chunk_size]
        for idx in range(0, len(cif_paths), chunk_size)
    ]


def relax_supported_inputs_in_chunks(
    *,
    supported_input_dir: Path,
    output_dir: Path,
    device: str,
    relax_chunk_size: int,
) -> tuple[Path, Path]:
    chunk_root = output_dir / "relax_chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)
    chunk_root.mkdir(parents=True, exist_ok=True)

    combined_extxyz = output_dir / "mattersim_relaxed.extxyz"
    combined_energies = output_dir / "mattersim_relaxed_energies.npy"
    if combined_extxyz.exists():
        combined_extxyz.unlink()
    if combined_energies.exists():
        combined_energies.unlink()

    all_energies: list[np.ndarray] = []
    with combined_extxyz.open("w") as combined_f:
        for chunk_idx, chunk_paths in enumerate(
            chunk_cif_paths(supported_input_dir, max(1, int(relax_chunk_size)))
        ):
            chunk_input_dir = chunk_root / f"chunk_{chunk_idx:03d}" / "input_cifs"
            chunk_input_dir.mkdir(parents=True, exist_ok=True)
            for cif_path in chunk_paths:
                shutil.copy2(cif_path, chunk_input_dir / cif_path.name)

            chunk_output_dir = chunk_input_dir.parent
            chunk_extxyz = chunk_output_dir / "relaxed.extxyz"
            chunk_energies = chunk_output_dir / "energies.npy"
            chunk_stdout_log = chunk_output_dir / "relax.stdout.log"
            chunk_stderr_log = chunk_output_dir / "relax.stderr.log"
            cmd = [
                str(MATTERGEN_PYTHON),
                str(MATTERGEN_RELAX_ONLY),
                f"--structures_path={chunk_input_dir}",
                f"--output_structures_path={chunk_extxyz}",
                f"--output_energies_path={chunk_energies}",
                f"--device={device}",
            ]
            with chunk_stdout_log.open("w") as stdout_f, chunk_stderr_log.open("w") as stderr_f:
                completed = subprocess.run(
                    cmd,
                    cwd=str(MATTERGEN_ROOT),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                )
            if completed.returncode != 0:
                stdout_tail = chunk_stdout_log.read_text()[-2000:] if chunk_stdout_log.exists() else ""
                stderr_tail = chunk_stderr_log.read_text()[-2000:] if chunk_stderr_log.exists() else ""
                raise RuntimeError(
                    f"Chunk relaxation failed for {chunk_input_dir}:\n"
                    f"STDOUT:\n{stdout_tail}\nSTDERR:\n{stderr_tail}"
                )

            combined_f.write(chunk_extxyz.read_text())
            all_energies.append(np.load(chunk_energies))

    np.save(combined_energies, np.concatenate(all_energies))
    return combined_extxyz, combined_energies


def run_group(
    spec: GroupSpec,
    *,
    output_dir: Path,
    reference_dataset_path: Path,
    energy_correction_scheme: str,
    structure_matcher: str,
    supported_terminals: set[str],
    device: str,
    relax_chunk_size: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    detailed_path = output_dir / "detailed.json"
    relaxed_extxyz = output_dir / "mattersim_relaxed.extxyz"

    if metrics_path.exists():
        row: dict[str, object] = {
            "group": spec.group,
            "guidance_factor": spec.guidance_factor,
            "target": spec.target,
            "structures_path": str(spec.cifs_dir),
            "output_dir": str(output_dir),
            "elapsed_seconds": "",
            "returncode": 0,
            "metrics_json": str(metrics_path),
            "detailed_json": str(detailed_path) if detailed_path.exists() else "",
            "relaxed_extxyz": str(relaxed_extxyz) if relaxed_extxyz.exists() else "",
            "stderr_tail": "",
            "stdout_tail": "",
            "reused_existing": True,
        }
        row.update(json.loads(metrics_path.read_text()))
        return row

    supported_input_dir, skipped_rows, total_count = prepare_supported_input(
        spec,
        output_dir=output_dir,
        supported_terminals=supported_terminals,
    )
    skipped_count = len(skipped_rows)
    supported_count = total_count - skipped_count

    if supported_input_dir is None:
        return {
            "group": spec.group,
            "guidance_factor": spec.guidance_factor,
            "target": spec.target,
            "structures_path": str(spec.cifs_dir),
            "output_dir": str(output_dir),
            "elapsed_seconds": 0.0,
            "returncode": 0,
            "metrics_json": "",
            "detailed_json": "",
            "relaxed_extxyz": "",
            "stderr_tail": "",
            "stdout_tail": "",
            "reused_existing": False,
            "total_input_cifs": total_count,
            "supported_input_cifs": 0,
            "skipped_input_cifs": skipped_count,
            "skipped_all_for_unsupported_terminals": True,
        }

    try:
        combined_extxyz, combined_energies = relax_supported_inputs_in_chunks(
            supported_input_dir=supported_input_dir,
            output_dir=output_dir,
            device=device,
            relax_chunk_size=relax_chunk_size,
        )
    except Exception as exc:
        return {
            "group": spec.group,
            "guidance_factor": spec.guidance_factor,
            "target": spec.target,
            "structures_path": str(spec.cifs_dir),
            "output_dir": str(output_dir),
            "elapsed_seconds": 0.0,
            "returncode": 1,
            "metrics_json": "",
            "detailed_json": "",
            "relaxed_extxyz": "",
            "stderr_tail": str(exc)[-2000:],
            "stdout_tail": "",
            "reused_existing": False,
            "total_input_cifs": total_count,
            "supported_input_cifs": supported_count,
            "skipped_input_cifs": skipped_count,
            "skipped_all_for_unsupported_terminals": False,
        }

    cmd = [
        str(MATTERGEN_PYTHON),
        str(MATTERGEN_EVAL),
        f"--structures_path={combined_extxyz}",
        "--relax=False",
        f"--energies_path={combined_energies}",
        f"--structure_matcher={structure_matcher}",
        f"--save_as={metrics_path}",
        f"--save_detailed_as={detailed_path}",
        f"--reference_dataset_path={reference_dataset_path}",
        f"--energy_correction_scheme={energy_correction_scheme}",
        f"--device={device}",
    ]
    eval_stdout_log = output_dir / "eval.stdout.log"
    eval_stderr_log = output_dir / "eval.stderr.log"
    started = time.monotonic()
    with eval_stdout_log.open("w") as stdout_f, eval_stderr_log.open("w") as stderr_f:
        completed = subprocess.run(
            cmd,
            cwd=str(MATTERGEN_ROOT),
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
        )
    elapsed = time.monotonic() - started
    stdout_tail = eval_stdout_log.read_text()[-2000:] if eval_stdout_log.exists() else ""
    stderr_tail = eval_stderr_log.read_text()[-2000:] if eval_stderr_log.exists() else ""

    row: dict[str, object] = {
        "group": spec.group,
        "guidance_factor": spec.guidance_factor,
        "target": spec.target,
        "structures_path": str(spec.cifs_dir),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(elapsed, 6),
        "returncode": completed.returncode,
        "metrics_json": str(metrics_path) if metrics_path.exists() else "",
        "detailed_json": str(detailed_path) if detailed_path.exists() else "",
        "relaxed_extxyz": str(combined_extxyz) if combined_extxyz.exists() else "",
        "stderr_tail": stderr_tail,
        "stdout_tail": stdout_tail,
        "reused_existing": False,
        "total_input_cifs": total_count,
        "supported_input_cifs": supported_count,
        "skipped_input_cifs": skipped_count,
        "skipped_all_for_unsupported_terminals": False,
    }

    if completed.returncode == 0 and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        row.update(metrics)
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
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


def load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    groups = discover_groups(args.relaxed_root, args.method)
    if args.groups:
        allowed = set(args.groups)
        groups = [spec for spec in groups if spec.group in allowed]
    if args.limit_groups is not None:
        groups = groups[: int(args.limit_groups)]
    if not groups:
        raise FileNotFoundError(f"No groups found for method={args.method}")
    supported_terminals = load_supported_terminals(args.reference_dataset_path)

    eval_root = args.output_root / f"mattergen_eval_{args.method}"
    eval_root.mkdir(parents=True, exist_ok=True)

    summary_path = eval_root / "summary.csv"
    rows_by_group: dict[str, dict[str, object]] = {
        row["group"]: row for row in load_existing_rows(summary_path) if row.get("group")
    }
    for spec in groups:
        group_output_dir = eval_root / spec.group
        print(f"[mattergen] group={spec.group}")
        row = run_group(
            spec,
            output_dir=group_output_dir,
            reference_dataset_path=args.reference_dataset_path,
            energy_correction_scheme=args.energy_correction_scheme,
            structure_matcher=args.structure_matcher,
            supported_terminals=supported_terminals,
            device=args.device,
            relax_chunk_size=args.relax_chunk_size,
        )
        rows_by_group[spec.group] = row
        ordered_rows = [rows_by_group[key] for key in sorted(rows_by_group.keys())]
        write_csv(ordered_rows, summary_path)
        if int(row["returncode"]) != 0:
            print(f"[mattergen] failed group={spec.group}")
            break

    ordered_rows = [rows_by_group[key] for key in sorted(rows_by_group.keys())]
    write_csv(ordered_rows, summary_path)
    (eval_root / "manifest.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "reference_dataset_path": str(args.reference_dataset_path),
                "energy_correction_scheme": args.energy_correction_scheme,
                "structure_matcher": args.structure_matcher,
                "device": args.device,
                "relax_chunk_size": args.relax_chunk_size,
                "groups": [spec.group for spec in groups],
            },
            indent=2,
        )
    )
    print(f"[done] output={eval_root}")


if __name__ == "__main__":
    main()
