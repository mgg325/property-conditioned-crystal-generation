# Thesis data map

| Thesis output | Analysis script | Source data | Sample-level data | Structures | Status |
|---|---|---|---|---|---|
| Table 1 — MatterGen guidance sweep | `src/mattergen/thesis_scripts/analysis/plot_target_distributions.py` | `data/final_evaluations/mattergen/source/adapter_group_summary_indexfixed.csv` | `data/final_evaluations/mattergen/mattergen_final_samples.csv` | `data/structures/mattergen/final_guidance_sweep/` | COMPLETE |
| Table 2 — Crystalite FiLM guidance sweep | Crystalite final-report analysis scripts; `src/analysis/build_crystalite_thermo_table.py` | `data/final_evaluations/crystalite/source/consolidated_group_summary.csv`; `data/processed/crystalite/table2_thermodynamic_metrics.csv` | `data/final_evaluations/crystalite/crystalite_final_samples.csv` | `data/structures/crystalite/final_film_guidance_sweep/` | COMPLETE |
| Table 3 — MatterGen vs Crystalite | common evaluation analysis | final MatterGen and Crystalite source tables | final MatterGen and Crystalite ledgers | final sweep structures | COMPLETE |
| Table 4 — reward-guided MatterGen | `src/mattergen/dielectric_rl/online_matinvent_loop.py` | `data/final_evaluations/reward_guided/` | `reward_guided_target12_samples.csv` | `data/structures/reward_guided/` | COMPLETE |
| Table S2 — complete MatterGen guidance sweep | MatterGen guidance analysis | MatterGen source tables | `mattergen_final_samples.csv` | MatterGen final sweep | COMPLETE |
| Table S3 — MatterGen diversity | `src/mattergen/thesis_scripts/analysis/analyze_guidance_family_audit.py` | `data/final_evaluations/mattergen/diversity/` | `mattergen_final_samples.csv` | MatterGen relaxed CIFs | COMPLETE |
| Table S4 — Crystalite conditioning methods | Crystalite distribution scripts | `data/final_evaluations/crystalite/development/` | development per-sample tables | not retained for development methods | COMPLETE |
| Table S6 — complete Crystalite FiLM evaluation | Crystalite final-report analysis scripts | Crystalite source tables | `crystalite_final_samples.csv` | Crystalite final FiLM sweep | COMPLETE |
| Table S7 — exact-count comparison | common evaluation analysis | final MatterGen and Crystalite ledgers | final ledgers | final sweep structures | COMPLETE |
| Table S9 — reward-guided settings/results | reward-guided analysis | branch metadata and loop summaries | `reward_guided_target12_samples.csv` | reward-guided structures | COMPLETE |
