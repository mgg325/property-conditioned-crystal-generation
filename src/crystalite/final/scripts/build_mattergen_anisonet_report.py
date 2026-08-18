from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
DEFAULT_TRAIN_CSV = ROOT / "data" / "anisonet_dielectric_crystalite" / "raw" / "train.csv"
DEFAULT_ANISONET_CKPT = Path("external/anisonet/anisonet-stock.ckpt")

ANISONET_COLUMNS = [
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

MATTERGEN_COLUMNS = [
    "elapsed_seconds",
    "returncode",
    "total_input_cifs",
    "successful_input_cifs",
    "skipped_input_cifs",
    "skipped_high_order_cifs",
    "skipped_unsupported_terminal_cifs",
    "avg_relax_seconds_per_input",
    "avg_relax_seconds_per_success",
    "avg_energy_above_hull_per_atom",
    "avg_rmsd_from_relaxation",
    "frac_novel_unique_stable_structures",
    "frac_stable_structures",
    "frac_successful_jobs",
    "avg_structure_validity",
    "frac_novel_structures",
    "frac_novel_systems",
    "frac_novel_unique_structures",
    "frac_unique_structures",
    "frac_unique_systems",
    "precision",
    "recall",
    "avg_comp_validity",
    "avg_comp_validity_timeout_rate",
    "avg_comp_validity_error_rate",
    "avg_structure_comp_validity",
    "avg_structure_comp_validity_timeout_rate",
    "avg_structure_comp_validity_error_rate",
    "smact_timeout_seconds",
    "smact_checked_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AnisoNet evaluation on MatterSim-successful CIFs and build a "
            "consolidated group summary with skipped-CIF counts."
        )
    )
    parser.add_argument("--mattergen_root", type=Path, required=True)
    parser.add_argument("--eval_output_dir", type=Path, required=True)
    parser.add_argument("--view_root", type=Path, required=True)
    parser.add_argument("--final_csv", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--anisonet_checkpoint", type=Path, default=DEFAULT_ANISONET_CKPT)
    parser.add_argument("--anisonet_batch_size", type=int, default=8)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_view_root(
    mattergen_rows: list[dict[str, str]],
    view_root: Path,
) -> list[tuple[str, str, str, Path]]:
    _clean_dir(view_root)
    group_specs: list[tuple[str, str, str, Path]] = []
    for row in mattergen_rows:
        group = row["group"]
        group_output = Path(row["output_dir"])
        src = group_output / "successful_relaxed_cifs"
        if not src.exists():
            continue
        cif_count = len(list(src.glob("*.cif")))
        if cif_count == 0:
            continue
        dst_group = view_root / group
        dst_group.mkdir(parents=True, exist_ok=True)
        dst_cifs = dst_group / "cifs"
        if dst_cifs.exists() or dst_cifs.is_symlink():
            dst_cifs.unlink()
        dst_cifs.symlink_to(src, target_is_directory=True)
        group_specs.append((group, row["guidance_factor"], row["target"], dst_group))
    return group_specs


def _run_anisonet_eval(
    group_specs: list[tuple[str, str, str, Path]],
    *,
    eval_output_dir: Path,
    train_csv: Path,
    checkpoint: Path,
    batch_size: int,
) -> None:
    if eval_output_dir.exists():
        shutil.rmtree(eval_output_dir)
    cmd = [
        str(PYTHON),
        "src/eval_crystalite_samples.py",
        "--output_dir",
        str(eval_output_dir),
        "--train_csv",
        str(train_csv),
        "--anisonet_checkpoint",
        str(checkpoint),
        "--anisonet_batch_size",
        str(batch_size),
    ]
    for group, guidance_factor, target, path in group_specs:
        cmd.extend(
            [
                "--group",
                f"{group},{guidance_factor},{target},{path}",
            ]
        )
    subprocess.run(cmd, cwd=ROOT, check=True)


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _default_anisonet_row(group: str, guidance_factor: str, target: str) -> dict[str, object]:
    row: dict[str, object] = {
        "group": group,
        "guidance_factor": guidance_factor,
        "target": target,
        "n_structures": 0,
        "n_readable": 0,
        "n_struct_valid": 0,
        "n_predicted": 0,
    }
    for col in ANISONET_COLUMNS:
        row.setdefault(col, "")
    row["n_structures"] = 0
    row["n_readable"] = 0
    row["n_struct_valid"] = 0
    row["n_predicted"] = 0
    return row


def _merge_rows(
    mattergen_rows: list[dict[str, str]],
    anisonet_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    anisonet_by_group = {row["group"]: row for row in anisonet_rows}
    merged_rows: list[dict[str, object]] = []
    for mrow in mattergen_rows:
        group = mrow["group"]
        arow = anisonet_by_group.get(
            group,
            _default_anisonet_row(group, mrow["guidance_factor"], mrow["target"]),
        )
        merged: dict[str, object] = {
            "method": "fixed_film_raw_mattersim",
            "group": group,
            "guidance_factor": mrow["guidance_factor"],
            "target": mrow["target"],
            "raw_group_dir": mrow["structures_path"],
            "mattergen_group_dir": mrow["output_dir"],
            "relaxed_group_dir": str(Path(mrow["output_dir"]) / "successful_relaxed_cifs"),
            "skipped_fraction": (
                _to_float(mrow["skipped_input_cifs"]) / _to_float(mrow["total_input_cifs"])
                if _to_float(mrow["skipped_input_cifs"]) is not None
                and _to_float(mrow["total_input_cifs"]) not in (None, 0.0)
                else ""
            ),
            "anisonet_unpredicted_cifs": (
                int(float(arow["n_structures"])) - int(float(arow["n_predicted"]))
                if str(arow.get("n_structures", "")).strip()
                and str(arow.get("n_predicted", "")).strip()
                else ""
            ),
        }
        for col in MATTERGEN_COLUMNS:
            merged[col] = mrow.get(col, "")
        for col in ANISONET_COLUMNS:
            merged[col] = arow.get(col, "")
        merged_rows.append(merged)
    return merged_rows


def main() -> None:
    args = parse_args()
    summary_csv = args.mattergen_root / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"MatterGen summary not found: {summary_csv}")

    mattergen_rows = _read_csv(summary_csv)
    group_specs = _build_view_root(mattergen_rows, args.view_root)
    _run_anisonet_eval(
        group_specs,
        eval_output_dir=args.eval_output_dir,
        train_csv=args.train_csv,
        checkpoint=args.anisonet_checkpoint,
        batch_size=args.anisonet_batch_size,
    )

    anisonet_rows = _read_csv(args.eval_output_dir / "group_summary.csv")
    merged_rows = _merge_rows(mattergen_rows, anisonet_rows)
    _write_csv(merged_rows, args.final_csv)
    print(f"[done] anisonet_eval={args.eval_output_dir}")
    print(f"[done] final_csv={args.final_csv}")


if __name__ == "__main__":
    main()
