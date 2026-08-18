# Property-conditioned Generation of Crystalline Materials

This repository accompanies the MSc thesis *Property-conditioned Generation of Crystalline Materials* by Aijing Chen, Department of Chemistry, Imperial College London. It contains code, configurations, and research data for dielectric-property-conditioned crystal generation and reward-guided MatterGen fine-tuning.

The work has three components:

- MatterGen dielectric conditioning;
- Crystalite dielectric conditioning; and
- reward-guided MatterGen fine-tuning.

## Repository structure

- `src/mattergen/` — MatterGen preprocessing, conditioning, generation, and evaluation code. Reward-guided code is in `src/mattergen/dielectric_rl/`.
- `src/crystalite/` — Crystalite dielectric-conditioning, generation, and evaluation code.
- `src/analysis/` — analysis and thesis-result verification scripts.
- `configs/` — configuration files used by the released workflows.
- `data/final_evaluations/` — sample-level evaluation records supporting reported results.
- `data/structures/` — generated and relaxed CIF files from final reported experiments.
- `data/figure_data/` — processed data used for quantitative thesis figures.
- `data/verification/` — recomputed checks of reported thesis metrics.
- `THESIS_DATA_MAP.md` — mapping between thesis tables/figures and released source data.
- `THIRD_PARTY.md` — external software and model dependencies.

## Data supporting the thesis

Released data include sample-level MatterGen and Crystalite evaluation results, reward-guided evaluation records, generated and relaxed CIF files, representative structures, and processed figure/table source data. The canonical evaluation tables and `THESIS_DATA_MAP.md` map thesis outputs to these data.

The MatterGen dielectric-adapter workflow uses prepared training, validation, and test splits derived from AnisoNet-labelled dielectric data.

## Verifying reported results

From the repository root, run:

```bash
python src/analysis/verify_thesis_results.py --release-root .
```

The script reads canonical sample-level data and writes checked summary metrics to `data/verification/thesis_result_verification.csv`.

## External dependencies and data

This repository does not redistribute pretrained MatterGen or Crystalite weights, AnisoNet model assets, MatterSim model assets, raw external training datasets, or external thermodynamic reference data. `THIRD_PARTY.md` records the corresponding upstream resources and retained project-specific modifications.

All retained result data and structures use repository-relative paths. The empty `external/` directory identifies inputs that must be obtained separately.

## Installation

Install MatterGen and Crystalite from the upstream versions documented in `THIRD_PARTY.md`. Install AnisoNet and obtain required MatterSim assets separately. Then install the dependencies for the retained project-specific scripts:

```bash
python -m pip install -r requirements.txt
```

`configs/crystalite/pyproject.toml` retains the upstream Crystalite environment metadata. The shared `requirements.txt` is not a replacement for either upstream model environment.

## Licence

Project-specific code in this repository is released under the MIT License. Third-party components remain subject to their original licences. Third-party software and modified upstream files retain their original copyright and licence notices. External datasets, pretrained models and other third-party assets are not relicensed by this repository.

## Citation

Please cite this work as:

> Chen, Aijing. *Property-conditioned Generation of Crystalline Materials*. MSc Digital Chemistry thesis, Department of Chemistry, Imperial College London, 2026.

The same citation metadata is provided in `CITATION.cff`.
