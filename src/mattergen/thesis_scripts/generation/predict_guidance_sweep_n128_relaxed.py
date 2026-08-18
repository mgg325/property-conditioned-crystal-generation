#!/usr/bin/env python3
"""Run CPU-only AnisoNet inference on relaxed MatterGen guidance sweep n128 samples."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-anisonet"))

import numpy as np
import pandas as pd
import torch
from ase import Atoms
from ase.io import read
from e3nn.io import CartesianTensor
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import DataLoader

from anisonet.data import BaseDataset, collate_fn
from anisonet.model import E3nnModel

ROOT = Path("external/anisonet")
SWEEP_DIR = ROOT / "generation_tests" / "Best Checkpoint generation" / "guidance_sweep_n128"
CHECKPOINT_PATH = ROOT / "anisonet-stock.ckpt"
RELAXED_EXTXYZ = SWEEP_DIR / "all_guidance_conditions_relaxed.extxyz"
PREDICTIONS_CSV = SWEEP_DIR / "anisonet_guidance_sweep_n128_relaxed_predictions.csv"
GUIDANCE_ORDER = [0, 1, 2, 5]
CONDITION_ORDER = [2, 4, 8, 12]
STRUCTURES_PER_GROUP = 128
EXPECTED_STRUCTURES = len(GUIDANCE_ORDER) * len(CONDITION_ORDER) * STRUCTURES_PER_GROUP
BATCH_SIZE = 8


def normalize_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_group(global_index: int) -> tuple[int, int, int]:
    group_index, structure_index = divmod(global_index, STRUCTURES_PER_GROUP)
    guidance_index, condition_index = divmod(group_index, len(CONDITION_ORDER))
    if guidance_index >= len(GUIDANCE_ORDER):
        raise ValueError(f"Cannot infer metadata for structure index {global_index}")
    return GUIDANCE_ORDER[guidance_index], CONDITION_ORDER[condition_index], structure_index


def metadata_from_atoms(atoms: Atoms, global_index: int) -> dict[str, Any]:
    structure = AseAtomsAdaptor.get_structure(atoms)
    inferred_guidance, inferred_condition, inferred_structure_index = infer_group(global_index)

    guidance_factor = normalize_int(atoms.info.get("guidance_factor"))
    condition = normalize_int(atoms.info.get("condition"))
    structure_index = normalize_int(atoms.info.get("structure_index"))

    if guidance_factor is None:
        guidance_factor = inferred_guidance
    if condition is None:
        condition = inferred_condition
    if structure_index is None:
        structure_index = inferred_structure_index

    return {
        "guidance_factor": guidance_factor,
        "condition": condition,
        "structure_index": structure_index,
        "formula": atoms.get_chemical_formula(),
        "chemical_system": structure.composition.chemical_system,
        "num_sites": len(atoms),
        "volume": float(atoms.get_volume()),
    }


def load_structures() -> tuple[list[dict[str, Any]], list[Atoms]]:
    structures = read(RELAXED_EXTXYZ, ":")
    if len(structures) != EXPECTED_STRUCTURES:
        raise ValueError(f"Expected {EXPECTED_STRUCTURES} relaxed structures, found {len(structures)}")
    rows = [metadata_from_atoms(atoms, index) for index, atoms in enumerate(structures)]
    return rows, structures


def load_model(dataset: BaseDataset, device: torch.device) -> tuple[E3nnModel, CartesianTensor]:
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

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) else checkpoint
    adjusted_state_dict = {
        key.replace("model.", "", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    net.load_state_dict(adjusted_state_dict)
    net.eval().to(device)
    return net, ct


def run_predictions(rows: list[dict[str, Any]], structures: list[Atoms]) -> pd.DataFrame:
    torch.manual_seed(1234)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    df = pd.DataFrame(
        {
            "structure": structures,
            "target": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in structures],
        }
    )
    dataset = BaseDataset(df, cutoff=5)
    net, ct = load_model(dataset, device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    outputs = []
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            outputs.append(net(batch).detach().cpu())

    cart_pred = ct.to_cartesian(torch.cat(outputs, dim=0))
    prediction_rows = []
    for metadata, tensor in zip(rows, cart_pred):
        tensor_np = tensor.numpy()
        eigenvalues = np.linalg.eigvalsh(tensor_np)
        prediction_rows.append(
            {
                **metadata,
                "predicted_dielectric_scalar": float(eigenvalues.mean()),
                "predicted_tensor": json.dumps(tensor_np.tolist()),
            }
        )
    return pd.DataFrame(prediction_rows)


def grouped_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["guidance_factor", "condition"])["predicted_dielectric_scalar"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not RELAXED_EXTXYZ.exists():
        raise FileNotFoundError(f"Relaxed extxyz not found: {RELAXED_EXTXYZ}")

    rows, structures = load_structures()
    predictions = run_predictions(rows, structures)
    output_columns = [
        "guidance_factor",
        "condition",
        "structure_index",
        "formula",
        "chemical_system",
        "num_sites",
        "volume",
        "predicted_dielectric_scalar",
        "predicted_tensor",
    ]
    predictions[output_columns].to_csv(PREDICTIONS_CSV, index=False)

    print(f"prediction CSV path: {PREDICTIONS_CSV}")
    print(f"prediction row count: {len(predictions)}")
    print("\nGrouped summary by guidance_factor and condition:")
    print(grouped_summary(predictions).to_string(index=False))


if __name__ == "__main__":
    main()
