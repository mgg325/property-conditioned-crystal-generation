#!/usr/bin/env python3
# Example usage:
# python structural_validity_check.py --input_dir generated_cifs --output_csv validity.csv

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure


def read_structure(cif_path: Path) -> tuple[Structure | None, str]:
    try:
        return Structure.from_file(cif_path), ""
    except Exception as exc:  # noqa: BLE001 - record bad CIFs and keep processing.
        return None, str(exc)


def get_min_interatomic_distance(structure: Structure) -> float:
    if len(structure) < 2:
        return float("nan")

    distance_matrix = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(distance_matrix, np.inf)
    min_distance = float(np.min(distance_matrix))
    if not np.isfinite(min_distance):
        return float("nan")
    return min_distance


def check_structure(cif_path: Path, min_dist: float, vpa_min: float, vpa_max: float) -> dict[str, Any]:
    structure, error = read_structure(cif_path)
    base_row: dict[str, Any] = {
        "file_name": cif_path.name,
        "file_path": str(cif_path.resolve()),
        "readable": structure is not None,
        "error_message": error,
        "reduced_formula": "",
        "elements": "",
        "num_elements": np.nan,
        "num_sites": np.nan,
        "a": np.nan,
        "b": np.nan,
        "c": np.nan,
        "alpha": np.nan,
        "beta": np.nan,
        "gamma": np.nan,
        "volume": np.nan,
        "volume_per_atom": np.nan,
        "density": np.nan,
        "min_interatomic_distance": np.nan,
        "volume_per_atom_reasonable": False,
        "min_distance_reasonable": False,
        "basic_structural_valid": False,
    }

    if structure is None:
        return base_row

    num_sites = len(structure)
    volume = float(structure.volume)
    volume_per_atom = volume / num_sites if num_sites > 0 else float("nan")
    min_interatomic_distance = get_min_interatomic_distance(structure)
    volume_ok = bool(vpa_min <= volume_per_atom <= vpa_max)
    min_distance_ok = bool(np.isfinite(min_interatomic_distance) and min_interatomic_distance > min_dist)

    base_row.update(
        {
            "reduced_formula": structure.composition.reduced_formula,
            "elements": ",".join(str(element) for element in structure.composition.elements),
            "num_elements": len(structure.composition.elements),
            "num_sites": num_sites,
            "a": float(structure.lattice.a),
            "b": float(structure.lattice.b),
            "c": float(structure.lattice.c),
            "alpha": float(structure.lattice.alpha),
            "beta": float(structure.lattice.beta),
            "gamma": float(structure.lattice.gamma),
            "volume": volume,
            "volume_per_atom": volume_per_atom,
            "density": float(structure.density),
            "min_interatomic_distance": min_interatomic_distance,
            "volume_per_atom_reasonable": volume_ok,
            "min_distance_reasonable": min_distance_ok,
            "basic_structural_valid": bool(volume_ok and min_distance_ok),
        }
    )
    return base_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic structural validity check for CIF files.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .cif files.")
    parser.add_argument(
        "--output_csv",
        default="generated_structure_validity_check.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--min_dist", type=float, default=1.0, help="Minimum allowed pair distance.")
    parser.add_argument(
        "--vpa_min",
        type=float,
        default=5.0,
        help="Minimum allowed volume per atom in Angstrom^3/atom.",
    )
    parser.add_argument(
        "--vpa_max",
        type=float,
        default=100.0,
        help="Maximum allowed volume per atom in Angstrom^3/atom.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists():
        print(f"Input directory missing: {input_dir}")
        results = pd.DataFrame()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        return

    cif_paths = sorted(input_dir.glob("*.cif"))
    rows = [
        check_structure(cif_path, min_dist=args.min_dist, vpa_min=args.vpa_min, vpa_max=args.vpa_max)
        for cif_path in cif_paths
    ]
    results = pd.DataFrame(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    if results.empty:
        print(f"number of CIF files found: 0")
        print(f"wrote results: {output_csv}")
        return

    print(f"number of CIF files found: {len(results)}")
    print(f"number readable: {int(results['readable'].sum())}")
    print(
        "number passing volume-per-atom check: "
        f"{int(results['volume_per_atom_reasonable'].sum())}"
    )
    print(f"number passing min-distance check: {int(results['min_distance_reasonable'].sum())}")
    print(f"number basic_structural_valid: {int(results['basic_structural_valid'].sum())}")
    print(f"wrote results: {output_csv}")

    suspicious = results.sort_values(
        by=["min_interatomic_distance"],
        ascending=True,
        na_position="last",
    ).head(10)
    print("\n10 most suspicious structures by minimum interatomic distance:")
    print(
        suspicious[
            [
                "file_name",
                "reduced_formula",
                "min_interatomic_distance",
                "volume_per_atom",
                "readable",
                "volume_per_atom_reasonable",
                "min_distance_reasonable",
                "basic_structural_valid",
                "error_message",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
