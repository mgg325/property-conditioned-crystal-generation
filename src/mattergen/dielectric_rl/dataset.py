from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset

from mattergen.common.data.dataset import CrystalDataset, structures_to_numpy


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or value == ""


def _load_structure_from_zip_uri(uri: str) -> Structure:
    if not uri.startswith("zip://"):
        raise ValueError(f"Unsupported zip URI: {uri}")
    archive_and_member = uri[len("zip://") :]
    archive_path_str, member_name = archive_and_member.split("!", 1)
    archive_path = Path(archive_path_str)
    with ZipFile(archive_path) as zip_file:
        cif_text = zip_file.read(member_name).decode("utf-8")
    return Structure.from_str(cif_text, fmt="cif")


def load_structure_from_pathlike(path_or_uri: str) -> Structure:
    if path_or_uri.startswith("zip://"):
        return _load_structure_from_zip_uri(path_or_uri)
    return Structure.from_file(path_or_uri)


@dataclass(frozen=True)
class DatasetBuildResult:
    dataset: CrystalDataset
    failed_structure_ids: list[str]


def records_to_structures(
    records: Iterable[dict[str, Any]],
    target_dielectric_scalar: float,
    use_relaxed_if_available: bool = True,
) -> tuple[list[Structure], list[str]]:
    structures: list[Structure] = []
    failed_structure_ids: list[str] = []
    for record in records:
        structure_path = record.get("relaxed_cif_path") if use_relaxed_if_available else None
        if _is_missing(structure_path):
            structure_path = record.get("cif_path")
        try:
            structure = load_structure_from_pathlike(str(structure_path))
        except Exception:
            failed_structure_ids.append(str(record.get("structure_id")))
            continue
        structure.properties["material_id"] = str(record["structure_id"])
        structure.properties["dft_dielectric_scalar"] = float(target_dielectric_scalar)
        structure.properties["reward_final"] = float(record.get("reward_final", 0.0))
        structures.append(structure)
    return structures, failed_structure_ids


def build_crystal_dataset_from_records(
    records: Iterable[dict[str, Any]],
    target_dielectric_scalar: float,
    use_relaxed_if_available: bool = True,
) -> DatasetBuildResult:
    structures, failed_structure_ids = records_to_structures(
        records=records,
        target_dielectric_scalar=target_dielectric_scalar,
        use_relaxed_if_available=use_relaxed_if_available,
    )
    structure_infos, properties = structures_to_numpy(structures)
    dataset = CrystalDataset(
        pos=structure_infos["pos"],
        cell=structure_infos["cell"],
        atomic_numbers=structure_infos["atomic_numbers"],
        num_atoms=structure_infos["num_atoms"],
        structure_id=structure_infos["structure_id"],
        properties=properties,
    )
    return DatasetBuildResult(dataset=dataset, failed_structure_ids=failed_structure_ids)


class RewardWeightedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, sample_weights: np.ndarray):
        if len(base_dataset) != len(sample_weights):
            raise ValueError("sample_weights length must match base dataset length")
        self.base_dataset = base_dataset
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        graph = self.base_dataset[index]
        return graph.replace(
            sample_weight=torch.tensor(self.sample_weights[index], dtype=torch.float32)
        )
