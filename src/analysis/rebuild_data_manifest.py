#!/usr/bin/env python3
"""Rebuild checksums for released data files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix[1:] if suffix else "file"


def workflow(path: Path) -> str:
    parts = path.parts
    for name in ("mattergen", "crystalite", "reward_guided"):
        if name in parts:
            return name
    return "shared"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.release_root.resolve()
    data_root = root / "data"
    records = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.name == "DATA_MANIFEST.csv":
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "relative_path": relative,
                "data_type": classify(path),
                "workflow": workflow(path),
                "run_id": "",
                "sample_id_if_applicable": "",
                "source_label": "released research data",
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "notes": "",
            }
        )
    pd.DataFrame(records).to_csv(data_root / "DATA_MANIFEST.csv", index=False)


if __name__ == "__main__":
    main()
