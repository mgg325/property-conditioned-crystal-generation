#!/usr/bin/env python3
"""Build the Crystalite thermodynamic additions for Table 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(
        args.release_root
        / "data/final_evaluations/crystalite/source/final_film_group_summary.csv"
    )
    rows: list[dict[str, object]] = []
    for record in source.itertuples(index=False):
        targets = [2, 4, 8, 12] if record.guidance_factor == 0 else [record.target]
        for target in targets:
            n_generated = int(record.total_input_cifs)
            n_thermo = int(record.successful_input_cifs)
            rows.append(
                {
                    "guidance_factor": record.guidance_factor,
                    "target": target,
                    "n_generated": n_generated,
                    "n_thermo": n_thermo,
                    "thermodynamic_coverage_percent": 100 * n_thermo / n_generated,
                    "mean_energy_above_hull_eV_per_atom": record.avg_energy_above_hull_per_atom,
                }
            )

    output = args.release_root / "data/processed/crystalite/table2_thermodynamic_metrics.csv"
    pd.DataFrame(rows).sort_values(["guidance_factor", "target"]).to_csv(output, index=False)


if __name__ == "__main__":
    main()
