#!/usr/bin/env python3
"""Prepare the AnisoNet dielectric scalar dataset for MatterGen fine-tuning."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd
from ase import Atoms
from ase.io import write


DEFAULT_DATASET_PATH = Path("external/anisonet/dataset/train_dataset.p")
DEFAULT_CIF_DIR = Path("external/anisonet/mattergen_dielectric_cifs")
DEFAULT_CSV_PATH = Path("external/anisonet/mattergen_dielectric_input.csv")
REQUIRED_COLUMNS = (
    "structure",
    "dielectric_scalar",
    "space_group_number",
    "crystal_system",
    "subset",
)
CSV_COLUMNS = (
    "material_id",
    "cif_path",
    "dft_dielectric_scalar",
    "dielectric_scalar_raw",
    "space_group_number",
    "crystal_system",
    "subset",
)


def load_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        obj = pickle.load(handle)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame from {path}, got {type(obj)!r}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in obj.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return obj


def safe_filename(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "material"


def material_id_for_row(row: pd.Series, original_index: Any) -> str:
    if "material_id" in row.index and pd.notna(row["material_id"]):
        return str(row["material_id"])
    return f"anisonet_train_{int(original_index):06d}"


def write_cif(atoms: Any, path: Path) -> None:
    if not isinstance(atoms, Atoms):
        raise TypeError(f"Expected an ase.Atoms object, got {type(atoms)!r}")
    write(path, atoms, format="cif")


def test_first_five_cifs(df: pd.DataFrame, cif_dir: Path) -> None:
    sample = df.head(5)
    print(f"Testing CIF writing for {len(sample)} structures...")

    for position, (original_index, row) in enumerate(sample.iterrows()):
        material_id = material_id_for_row(row, original_index)
        cif_path = cif_dir / f"{position:06d}_{safe_filename(material_id)}.cif"
        write_cif(row["structure"], cif_path)
        if not cif_path.exists():
            raise FileNotFoundError(f"Test CIF was not created: {cif_path}")
        print(f"  confirmed: {cif_path}")


def build_mattergen_input(df: pd.DataFrame, cif_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for position, (original_index, row) in enumerate(df.iterrows()):
        material_id = material_id_for_row(row, original_index)
        cif_path = cif_dir / f"{position:06d}_{safe_filename(material_id)}.cif"
        dielectric_scalar = row["dielectric_scalar"]

        write_cif(row["structure"], cif_path)

        rows.append(
            {
                "material_id": material_id,
                "cif_path": str(cif_path),
                "dft_dielectric_scalar": dielectric_scalar,
                "dielectric_scalar_raw": dielectric_scalar,
                "space_group_number": row["space_group_number"],
                "crystal_system": row["crystal_system"],
                "subset": row["subset"],
            }
        )

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one labelled subset from an AnisoNet dielectric dataset."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument(
        "--subset",
        default="train",
        help="Value in the dataset's subset column to export (default: train).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.dataset_path)
    selected_df = df[df["subset"] == args.subset].copy()
    if selected_df.empty:
        raise ValueError(f"No rows found with subset == {args.subset!r}")

    args.cif_dir.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    test_first_five_cifs(selected_df, args.cif_dir)

    mattergen_df = build_mattergen_input(selected_df, args.cif_dir)
    mattergen_df.to_csv(args.csv_path, index=False)

    print(f"Wrote CIF directory: {args.cif_dir}")
    print(f"Wrote CSV: {args.csv_path}")
    print(f"Final DataFrame shape: {mattergen_df.shape}")
    print("First 5 rows:")
    print(mattergen_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
