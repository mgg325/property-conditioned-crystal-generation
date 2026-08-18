from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.spatial.distance import pdist

from mattergen.evaluation.utils.structure_matcher import OrderedStructureMatcher


RELEASE_ROOT = Path(__file__).resolve().parents[4]
ROOT = RELEASE_ROOT / "data/final_evaluations/mattergen/source"
OUTDIR = RELEASE_ROOT / "data/final_evaluations/mattergen/diversity"
PLOTDIR = OUTDIR / "plots"

MASTER_CSV = ROOT / "adapter_evaluation_master_indexfixed.csv"
GROUP_SUMMARY_CSV = ROOT / "adapter_group_summary_indexfixed.csv"
RAW_MASTER_CSV = ROOT / "adapter_evaluation_master.csv"
RAW_GROUP_SUMMARY_CSV = ROOT / "adapter_group_summary.csv"
INDEXFIX_REPORT = ROOT / "adapter_evaluation_indexfix_report.txt"
PREP_LOG = ROOT / "adapter_evaluation_preparation_log.txt"
AUDIT_REPORT = ROOT / "adapter_evaluation_audit_report.txt"
METRICS_JSON = ROOT / "mattergen_guidance_sweep_n128_metrics.json"
DETAILED_JSON = ROOT / "mattergen_guidance_sweep_n128_detailed_metrics.json"
COMBINED_EVAL_CSV = ROOT / "mattergen_guidance_sweep_n128_combined_evaluation.csv"
EVAL_SUMMARY_CSV = ROOT / "mattergen_guidance_sweep_n128_evaluation_summary.csv"
RELAXED_EXTXYZ = ROOT / "all_guidance_conditions_relaxed.extxyz"
PREDICTIONS_CSV = ROOT / "anisonet_guidance_sweep_n128_relaxed_predictions.csv"
LOG_DIR = ROOT / "logs"

BOOTSTRAP_SEED = 12345
BOOTSTRAP_SAMPLES = 1000

TARGETS = [2, 4, 8, 12]
GUIDANCES = [0, 1, 2, 5]
KEY_GROUPS = [(2, 8), (5, 8), (2, 12), (5, 12)]


@dataclass
class FamilyMetricBundle:
    n_items: int
    n_families: int
    largest_family_fraction: float
    top5_cumulative_fraction: float
    shannon_entropy: float
    effective_num_families: float
    hhi: float
    mean_family_size: float
    median_family_size: float
    frac_in_families_ge_5: float


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV)
    df["group_key"] = list(zip(df["guidance_factor"], df["condition"]))
    df["guidance_label"] = df["guidance_factor"].map(lambda x: f"g={x}")
    df["target_label"] = df["condition"].map(lambda x: f"target={x}")
    df["subset_all_relaxed"] = True
    df["subset_sun_positive"] = df["novel_unique_stable"].astype(bool)
    return df


def load_structures() -> dict[int, Structure]:
    with open(DETAILED_JSON, "r") as f:
        raw = json.load(f)
    entries = raw["entry"]
    structures = {}
    for entry in entries:
        entry_id = int(entry["entry_id"])
        structures[entry_id] = Structure.from_dict(entry["structure"])
    return structures


def build_structure_df(master: pd.DataFrame, structures: dict[int, Structure]) -> pd.DataFrame:
    records = []
    for row in master.itertuples(index=False):
        global_idx = int(row.global_structure_index)
        structure = structures[global_idx]
        comp = structure.composition
        try:
            spg_symbol = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0).get_space_group_symbol()
        except Exception:
            spg_symbol = "P1"
        records.append(
            {
                "global_structure_index": global_idx,
                "reduced_formula": comp.reduced_formula,
                "anonymized_formula": comp.anonymized_formula,
                "space_group": spg_symbol,
                "structure_obj": structure,
            }
        )
    extra = pd.DataFrame.from_records(records)
    merged = master.merge(extra, on="global_structure_index", how="left", validate="one_to_one")
    return merged


def composition_vector(structure: Structure, element_order: list[str]) -> np.ndarray:
    frac = structure.composition.fractional_composition.as_dict()
    return np.array([float(frac.get(el, 0.0)) for el in element_order], dtype=float)


def family_metrics_from_labels(labels: Iterable[str]) -> FamilyMetricBundle:
    labels = list(labels)
    n = len(labels)
    counts = Counter(labels)
    family_sizes = np.array(sorted(counts.values(), reverse=True), dtype=float)
    if n == 0:
        return FamilyMetricBundle(0, 0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    p = family_sizes / n
    shannon = float(-(p * np.log(p)).sum())
    effective = float(math.exp(shannon))
    hhi = float((p ** 2).sum())
    largest = float(p[0])
    top5 = float(p[:5].sum())
    mean_size = float(family_sizes.mean())
    median_size = float(np.median(family_sizes))
    frac_ge5 = float(family_sizes[family_sizes >= 5].sum() / n)
    return FamilyMetricBundle(
        n_items=n,
        n_families=len(counts),
        largest_family_fraction=largest,
        top5_cumulative_fraction=top5,
        shannon_entropy=shannon,
        effective_num_families=effective,
        hhi=hhi,
        mean_family_size=mean_size,
        median_family_size=median_size,
        frac_in_families_ge_5=frac_ge5,
    )


def bootstrap_family_metrics(labels: list[str], n_boot: int = BOOTSTRAP_SAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return {
            "boot_largest_family_fraction_lo": math.nan,
            "boot_largest_family_fraction_hi": math.nan,
            "boot_effective_num_families_lo": math.nan,
            "boot_effective_num_families_hi": math.nan,
            "boot_hhi_lo": math.nan,
            "boot_hhi_hi": math.nan,
        }
    arr = np.array(labels, dtype=object)
    largest_vals = []
    effective_vals = []
    hhi_vals = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True).tolist()
        m = family_metrics_from_labels(sample)
        largest_vals.append(m.largest_family_fraction)
        effective_vals.append(m.effective_num_families)
        hhi_vals.append(m.hhi)
    def q(vals: list[float], p: float) -> float:
        return float(np.quantile(np.array(vals, dtype=float), p))
    return {
        "boot_largest_family_fraction_lo": q(largest_vals, 0.025),
        "boot_largest_family_fraction_hi": q(largest_vals, 0.975),
        "boot_effective_num_families_lo": q(effective_vals, 0.025),
        "boot_effective_num_families_hi": q(effective_vals, 0.975),
        "boot_hhi_lo": q(hhi_vals, 0.025),
        "boot_hhi_hi": q(hhi_vals, 0.975),
    }


def group_structures_by_matcher(df: pd.DataFrame, matcher: OrderedStructureMatcher) -> list[int]:
    clusters: list[list[Structure]] = []
    labels: list[int] = []
    for structure in df["structure_obj"]:
        assigned = False
        for idx, cluster in enumerate(clusters):
            if any(matcher.fit(structure, other) for other in cluster):
                cluster.append(structure)
                labels.append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append([structure])
            labels.append(len(clusters) - 1)
    return labels


def pairwise_same_family_fraction(labels: list[str | int]) -> float:
    n = len(labels)
    if n < 2:
        return math.nan
    counts = Counter(labels)
    num = sum(v * (v - 1) / 2 for v in counts.values())
    den = n * (n - 1) / 2
    return float(num / den)


def pairwise_distance_summary(matrix: np.ndarray) -> dict[str, float]:
    if matrix.shape[0] < 2:
        return {
            "pairwise_l1_mean": math.nan,
            "pairwise_l1_median": math.nan,
            "pairwise_l1_q25": math.nan,
            "pairwise_l1_q75": math.nan,
            "pairwise_cosine_mean": math.nan,
            "pairwise_cosine_median": math.nan,
            "pairwise_cosine_q25": math.nan,
            "pairwise_cosine_q75": math.nan,
        }
    l1 = pdist(matrix, metric="cityblock")
    cosine = pdist(matrix, metric="cosine")
    def summarize(vals: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(np.mean(vals)),
            f"{prefix}_median": float(np.median(vals)),
            f"{prefix}_q25": float(np.quantile(vals, 0.25)),
            f"{prefix}_q75": float(np.quantile(vals, 0.75)),
        }
    out = {}
    out.update(summarize(l1, "pairwise_l1"))
    out.update(summarize(cosine, "pairwise_cosine"))
    return out


def analyze_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matcher = OrderedStructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    element_order = sorted({el.symbol for s in df["structure_obj"] for el in s.composition.elements})

    family_rows: list[dict] = []
    top_rows: list[dict] = []
    subset_rows: list[dict] = []

    for guidance in GUIDANCES:
        for target in TARGETS:
            group = df[(df["guidance_factor"] == guidance) & (df["condition"] == target)].copy()
            group = group.sort_values("local_structure_index").reset_index(drop=True)
            group["structure_family_id_all"] = group_structures_by_matcher(group, matcher)
            group["structure_family_label_all"] = group["structure_family_id_all"].map(lambda x: f"SF{x:03d}")

            for subset_name, subset_mask_col in [("all_relaxed", "subset_all_relaxed"), ("sun_positive", "subset_sun_positive")]:
                subset = group[group[subset_mask_col]].copy().reset_index(drop=True)
                if subset_name == "sun_positive":
                    subset["structure_family_id_subset"] = group_structures_by_matcher(subset, matcher) if len(subset) else []
                    subset["structure_family_label_subset"] = subset["structure_family_id_subset"].map(lambda x: f"SF{x:03d}") if len(subset) else pd.Series(dtype=object)
                    structure_family_col = "structure_family_label_subset"
                else:
                    structure_family_col = "structure_family_label_all"

                n_generated = len(group)
                n_readable = len(group)
                n_relaxed = len(group)
                n_stable = int(subset["stable"].sum()) if len(subset) else 0
                n_unique = int(subset["is_unique"].sum()) if len(subset) else 0
                n_novel = int(subset["is_novel"].sum()) if len(subset) else 0
                n_sun = int(subset["novel_unique_stable"].sum()) if len(subset) else 0

                subset_row = {
                    "guidance_factor": guidance,
                    "condition": target,
                    "subset": subset_name,
                    "n_generated": n_generated,
                    "n_readable": n_readable,
                    "n_relaxed": n_relaxed,
                    "n_subset": len(subset),
                    "n_stable": n_stable,
                    "n_unique": n_unique,
                    "n_novel": n_novel,
                    "n_SUN": n_sun,
                }

                if len(subset):
                    comp_matrix = np.vstack([composition_vector(s, element_order) for s in subset["structure_obj"]])
                    subset_row.update(pairwise_distance_summary(comp_matrix))
                    subset_row["pair_same_structure_family_fraction"] = pairwise_same_family_fraction(subset[structure_family_col].tolist())
                    subset_row["pair_same_space_group_fraction"] = pairwise_same_family_fraction(subset["space_group"].tolist())
                else:
                    subset_row.update(pairwise_distance_summary(np.zeros((0, len(element_order)))))
                    subset_row["pair_same_structure_family_fraction"] = math.nan
                    subset_row["pair_same_space_group_fraction"] = math.nan
                subset_rows.append(subset_row)

                for family_type, labels in [
                    ("chemical_system", subset["chemical_system"].tolist()),
                    ("reduced_formula", subset["reduced_formula"].tolist()),
                    ("anonymized_formula", subset["anonymized_formula"].tolist()),
                    ("structure_family", subset[structure_family_col].tolist() if len(subset) else []),
                    ("space_group", subset["space_group"].tolist()),
                ]:
                    metrics = family_metrics_from_labels(labels)
                    row = {
                        "guidance_factor": guidance,
                        "condition": target,
                        "subset": subset_name,
                        "family_type": family_type,
                        "n_generated_group": len(group),
                        "n_subset": len(subset),
                        "n_stable_in_subset": int(subset["stable"].sum()) if len(subset) else 0,
                        "n_unique_in_subset": int(subset["is_unique"].sum()) if len(subset) else 0,
                        "n_novel_in_subset": int(subset["is_novel"].sum()) if len(subset) else 0,
                        "n_sun_in_subset": int(subset["novel_unique_stable"].sum()) if len(subset) else 0,
                        "n_families": metrics.n_families,
                        "largest_family_fraction": metrics.largest_family_fraction,
                        "top5_cumulative_fraction": metrics.top5_cumulative_fraction,
                        "shannon_entropy": metrics.shannon_entropy,
                        "effective_num_families": metrics.effective_num_families,
                        "hhi": metrics.hhi,
                        "mean_family_size": metrics.mean_family_size,
                        "median_family_size": metrics.median_family_size,
                        "frac_in_families_ge_5": metrics.frac_in_families_ge_5,
                    }
                    if family_type in {"chemical_system", "structure_family"}:
                        row.update(bootstrap_family_metrics(labels))
                    else:
                        row.update(
                            {
                                "boot_largest_family_fraction_lo": math.nan,
                                "boot_largest_family_fraction_hi": math.nan,
                                "boot_effective_num_families_lo": math.nan,
                                "boot_effective_num_families_hi": math.nan,
                                "boot_hhi_lo": math.nan,
                                "boot_hhi_hi": math.nan,
                            }
                        )
                    family_rows.append(row)

                    counts = Counter(labels)
                    if len(subset):
                        top_items = counts.most_common(10)
                        for rank, (fam_label, fam_count) in enumerate(top_items, start=1):
                            fam_df = subset[
                                subset[structure_family_col].eq(fam_label)
                                if family_type == "structure_family"
                                else subset[family_type].eq(fam_label)
                            ]
                            rep_formula = fam_df["reduced_formula"].mode().iloc[0]
                            rep_space_group = fam_df["space_group"].mode().iloc[0]
                            top_rows.append(
                                {
                                    "guidance_factor": guidance,
                                    "condition": target,
                                    "subset": subset_name,
                                    "family_type": family_type,
                                    "rank": rank,
                                    "family_label": fam_label,
                                    "count": int(fam_count),
                                    "fraction": float(fam_count / len(subset)),
                                    "representative_formula": rep_formula,
                                    "representative_space_group": rep_space_group,
                                    "mean_predicted_dielectric_scalar": float(fam_df["predicted_dielectric_scalar"].mean()),
                                    "mean_energy_above_hull_per_atom": float(fam_df["energy_above_hull_per_atom"].mean()),
                                    "stable_fraction_within_family": float(fam_df["stable"].mean()),
                                    "sun_fraction_within_family": float(fam_df["novel_unique_stable"].mean()),
                                }
                            )

    return (
        pd.DataFrame(family_rows),
        pd.DataFrame(top_rows),
        pd.DataFrame(subset_rows),
    )


def plot_heatmap(metric_df: pd.DataFrame, family_type: str, subset: str, value_col: str, title: str, filename: str) -> None:
    piv = metric_df[(metric_df["family_type"] == family_type) & (metric_df["subset"] == subset)].pivot(
        index="condition", columns="guidance_factor", values=value_col
    ).sort_index().sort_index(axis=1)
    plt.figure(figsize=(7, 4.5))
    im = plt.imshow(piv.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label=value_col)
    plt.xticks(range(len(piv.columns)), [f"g={c}" for c in piv.columns])
    plt.yticks(range(len(piv.index)), [f"target={i}" for i in piv.index])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTDIR / filename, dpi=200)
    plt.close()


def plot_rank_curves(top_df: pd.DataFrame, family_type: str, subset: str, filename: str, title: str) -> None:
    plt.figure(figsize=(8, 5.5))
    for guidance, target in KEY_GROUPS:
        group = top_df[
            (top_df["guidance_factor"] == guidance)
            & (top_df["condition"] == target)
            & (top_df["subset"] == subset)
            & (top_df["family_type"] == family_type)
        ].sort_values("rank")
        if group.empty:
            continue
        plt.plot(group["rank"], group["fraction"], marker="o", label=f"g={guidance}, target={target}")
    plt.xlabel("Family rank")
    plt.ylabel("Fraction of subset")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTDIR / filename, dpi=200)
    plt.close()


def plot_subset_metric_lines(subset_df: pd.DataFrame, value_col: str, filename: str, title: str) -> None:
    plt.figure(figsize=(8, 5.5))
    for target in TARGETS:
        grp = subset_df[(subset_df["subset"] == "all_relaxed") & (subset_df["condition"] == target)].sort_values("guidance_factor")
        plt.plot(grp["guidance_factor"], grp[value_col], marker="o", label=f"target={target}")
    plt.xlabel("Guidance factor")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTDIR / filename, dpi=200)
    plt.close()


def conclusion_from_metrics(metric_df: pd.DataFrame, subset_df: pd.DataFrame) -> str:
    chem = metric_df[(metric_df["family_type"] == "chemical_system") & (metric_df["subset"] == "all_relaxed")]
    struct = metric_df[(metric_df["family_type"] == "structure_family") & (metric_df["subset"] == "all_relaxed")]
    sun_struct = metric_df[(metric_df["family_type"] == "structure_family") & (metric_df["subset"] == "sun_positive")]

    support_votes = 0
    total_checks = 0
    for target in [8, 12]:
        c2 = chem[(chem["condition"] == target) & (chem["guidance_factor"] == 2)].iloc[0]
        c5 = chem[(chem["condition"] == target) & (chem["guidance_factor"] == 5)].iloc[0]
        s2 = struct[(struct["condition"] == target) & (struct["guidance_factor"] == 2)].iloc[0]
        s5 = struct[(struct["condition"] == target) & (struct["guidance_factor"] == 5)].iloc[0]
        p2 = subset_df[(subset_df["condition"] == target) & (subset_df["guidance_factor"] == 2) & (subset_df["subset"] == "all_relaxed")].iloc[0]
        p5 = subset_df[(subset_df["condition"] == target) & (subset_df["guidance_factor"] == 5) & (subset_df["subset"] == "all_relaxed")].iloc[0]

        checks = [
            c5["n_families"] < c2["n_families"],
            c5["effective_num_families"] < c2["effective_num_families"],
            c5["largest_family_fraction"] > c2["largest_family_fraction"],
            c5["hhi"] > c2["hhi"],
            p5["pairwise_l1_mean"] < p2["pairwise_l1_mean"],
            s5["largest_family_fraction"] > s2["largest_family_fraction"],
        ]
        support_votes += sum(checks)
        total_checks += len(checks)

        sun2 = sun_struct[(sun_struct["condition"] == target) & (sun_struct["guidance_factor"] == 2)]
        sun5 = sun_struct[(sun_struct["condition"] == target) & (sun_struct["guidance_factor"] == 5)]
        if not sun2.empty and not sun5.empty and sun2.iloc[0]["n_subset"] >= 10 and sun5.iloc[0]["n_subset"] >= 10:
            more_checks = [
                sun5.iloc[0]["effective_num_families"] < sun2.iloc[0]["effective_num_families"],
                sun5.iloc[0]["largest_family_fraction"] > sun2.iloc[0]["largest_family_fraction"],
            ]
            support_votes += sum(more_checks)
            total_checks += len(more_checks)

    if total_checks == 0:
        return "Inconclusive"
    frac = support_votes / total_checks
    if frac >= 0.7:
        return "Supported"
    if frac <= 0.4:
        return "Not supported"
    return "Inconclusive"


def fmt(x: float | int | str) -> str:
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.3f}"


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    work = df.loc[:, cols]
    if max_rows is not None:
        work = work.head(max_rows)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in work.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(metric_df: pd.DataFrame, top_df: pd.DataFrame, subset_df: pd.DataFrame) -> None:
    conclusion = conclusion_from_metrics(metric_df, subset_df)
    chem_relaxed = metric_df[(metric_df["family_type"] == "chemical_system") & (metric_df["subset"] == "all_relaxed")].copy()
    struct_relaxed = metric_df[(metric_df["family_type"] == "structure_family") & (metric_df["subset"] == "all_relaxed")].copy()
    sun_struct = metric_df[(metric_df["family_type"] == "structure_family") & (metric_df["subset"] == "sun_positive")].copy()
    key_compare = subset_df[(subset_df["condition"].isin([8, 12])) & (subset_df["guidance_factor"].isin([2, 5])) & (subset_df["subset"] == "all_relaxed")].copy()
    key_compare = key_compare.sort_values(["condition", "guidance_factor"])

    top_focus = top_df[
        (top_df["guidance_factor"].isin([2, 5]))
        & (top_df["condition"].isin([8, 12]))
        & (top_df["subset"] == "all_relaxed")
        & (top_df["family_type"].isin(["chemical_system", "structure_family"]))
    ].copy()

    report = f"""# MatterGen Dielectric-Adapter Guidance Sweep Family Audit

Date: 2026-08-01

## 1. Audit Scope

This read-only audit evaluates whether high classifier-free guidance, especially `g=5`, narrows the generated distribution into fewer chemical or structural families even when the standard MatterGen `SUN` fraction remains competitive.

Primary working directory:

- `{ROOT}`

Primary raw MatterGen evaluation outputs:

- `{COMBINED_EVAL_CSV}`
- `{EVAL_SUMMARY_CSV}`
- `{METRICS_JSON}`
- `{DETAILED_JSON}`
- `{RELAXED_EXTXYZ}`
- `{PREDICTIONS_CSV}`

Post-processed adapter evaluation tables used for paper-facing summaries:

- raw merged master: `{RAW_MASTER_CSV}`
- raw grouped summary: `{RAW_GROUP_SUMMARY_CSV}`
- corrected merged master: `{MASTER_CSV}`
- corrected grouped summary: `{GROUP_SUMMARY_CSV}`
- index-fix audit: `{INDEXFIX_REPORT}`
- preparation log: `{PREP_LOG}`
- audit report: `{AUDIT_REPORT}`

This audit uses the corrected per-sample table `{MASTER_CSV}` plus the relaxed structures from `{DETAILED_JSON}`.
That choice is evidence-based rather than name-based:

- `adapter_evaluation_preparation_log.txt` states that `adapter_evaluation_master.csv` was created by merging the relaxed AnisoNet prediction CSV with the MatterGen detailed metrics JSON.
- `adapter_evaluation_indexfix_report.txt` states that `adapter_evaluation_master_indexfixed.csv` preserves all non-index columns, adds `global_structure_index`, and replaces the original global `structure_index` with a local `0..127` index inside each `(guidance_factor, condition)` group.
- `adapter_group_summary_indexfixed.csv` is numerically unchanged from the original grouped summary within tolerance `1e-8`.

Analyzed groups:

- targets: `2, 4, 8, 12`
- guidance factors: `0, 1, 2, 5`

Primary comparison emphasis:

- `g=2` versus `g=5`
- `target=8` versus `target=12`

## 2. Current SUN / Uniqueness / Novelty Definitions

Code definitions inspected:

- uniqueness / novelty: `src/mattergen/evaluation/metrics/structure.py`
- dataset matching: `src/mattergen/evaluation/utils/dataset_matcher.py`
- structure matcher defaults: `src/mattergen/evaluation/utils/structure_matcher.py`
- energy / stable / SUN fractions: `src/mattergen/evaluation/metrics/energy.py`
- evaluator reference presets: `src/mattergen/evaluation/metrics/evaluator.py`

Verified implementation:

- `stable` is `energy_above_hull <= 0.1 eV/atom`, with denominator `total_submitted_jobs`.
- `SUN` is `is_novel & is_unique & is_stable`, again divided by `total_submitted_jobs`.
- `frac_novel_unique_stable_structures` in `{METRICS_JSON}` explicitly reports the reference as `TRI2024correction`.
- for ordered structures, uniqueness and novelty are both StructureMatcher-based, grouped by **reduced formula**.
- the default ordered matcher is:
  - `ltol=0.2`
  - `stol=0.3`
  - `angle_tol=5`
  - `primitive_cell=True`
  - `scale=True`
  - `attempt_supercell=False`
  - `allow_subset=False`
- space-group summaries in MatterGen use `SpacegroupAnalyzer(symprec=0.1, angle_tolerance=5.0)`.

Interpretive consequence:

- `is_unique` only removes near duplicates that the StructureMatcher considers equivalent inside the same reduced-formula bucket.
- `is_novel` only checks whether a generated structure has a StructureMatcher-equivalent match in the `TRI2024correction` reference set, again within the same reduced-formula grouping.
- therefore multiple samples from the same broad chemical family, prototype family, or composition neighborhood can all still be counted as `unique` and `novel` as long as they are not near duplicates under this matcher.

This is exactly why family-level diagnostics are needed: the current SUN definition is intentionally stricter than random deduplication, but still much narrower than “chemical family coverage”.

## 3. Family Definitions Used Here

The audit computes diversity diagnostics at multiple levels:

- `chemical_system`: exact element set such as `Ba-Ti-O`
- `reduced_formula`: exact reduced stoichiometry from `pymatgen.Composition.reduced_formula`
- `anonymized_formula`: stoichiometry pattern from `Composition.anonymized_formula`
- `composition distance`: pairwise distance between fractional-composition vectors over the global element vocabulary
- `structure_family`: relaxed-structure clustering with MatterGen’s ordered `StructureMatcher` defaults
- `space_group`: coarse crystallographic grouping with `SpacegroupAnalyzer(symprec=0.1, angle_tolerance=5.0)`

Fingerprint-based structural clustering:

- `matminer` is **not installed** in the current environment.
- therefore `CrystalNNFingerprint` / `SiteStatsFingerprint` clustering was **not** computed.
- instead, the structural-family analysis is based on `StructureMatcher` clustering, and pairwise structure concentration is summarized by the fraction of sample pairs that fall into the same StructureMatcher family.

This is reported as a **diagnostic similarity statistic**, not as a new formal paper metric.

## 4. Group-Level Accounting

{md_table(subset_df.sort_values(["subset", "condition", "guidance_factor"]), [
    "guidance_factor", "condition", "subset", "n_generated", "n_readable", "n_relaxed",
    "n_subset", "n_stable", "n_unique", "n_novel", "n_SUN"
])}

Notes:

- all groups have `128` generated / readable / relaxed structures in the corrected master table.
- for the `sun_positive` subset, `n_subset = n_SUN` by construction.

## 5. Family-Diversity Tables

### 5.1 Chemical-System Families on All Relaxed Structures

{md_table(chem_relaxed.sort_values(["condition", "guidance_factor"]), [
    "guidance_factor", "condition", "n_subset", "n_families",
    "largest_family_fraction", "top5_cumulative_fraction",
    "effective_num_families", "hhi",
    "boot_largest_family_fraction_lo", "boot_largest_family_fraction_hi",
    "boot_effective_num_families_lo", "boot_effective_num_families_hi",
    "boot_hhi_lo", "boot_hhi_hi"
])}

### 5.2 StructureMatcher Structural Families on All Relaxed Structures

{md_table(struct_relaxed.sort_values(["condition", "guidance_factor"]), [
    "guidance_factor", "condition", "n_subset", "n_families",
    "largest_family_fraction", "top5_cumulative_fraction",
    "effective_num_families", "hhi",
    "boot_largest_family_fraction_lo", "boot_largest_family_fraction_hi",
    "boot_effective_num_families_lo", "boot_effective_num_families_hi",
    "boot_hhi_lo", "boot_hhi_hi"
])}

### 5.3 StructureMatcher Structural Families on SUN-Positive Structures

{md_table(sun_struct.sort_values(["condition", "guidance_factor"]), [
    "guidance_factor", "condition", "n_subset", "n_families",
    "largest_family_fraction", "top5_cumulative_fraction",
    "effective_num_families", "hhi",
    "boot_largest_family_fraction_lo", "boot_largest_family_fraction_hi",
    "boot_effective_num_families_lo", "boot_effective_num_families_hi",
    "boot_hhi_lo", "boot_hhi_hi"
])}

## 6. Pairwise Composition / Structure Similarity Diagnostics

For continuous composition distances, the audit uses fractional-composition vectors and reports pairwise `L1` and cosine distances.

For structure concentration, because `matminer` fingerprints are unavailable, the audit reports:

- `pair_same_structure_family_fraction`: fraction of all sample pairs that land in the same StructureMatcher family
- `pair_same_space_group_fraction`: fraction of all sample pairs that share the same space group

Key comparison groups:

{md_table(key_compare, [
    "guidance_factor", "condition",
    "pairwise_l1_mean", "pairwise_l1_median",
    "pairwise_cosine_mean", "pairwise_cosine_median",
    "pair_same_structure_family_fraction",
    "pair_same_space_group_fraction"
])}

## 7. Focused `g=2` vs `g=5` Comparison for Targets 8 and 12

### 7.1 Chemical-system concentration

{md_table(
    chem_relaxed[chem_relaxed["condition"].isin([8, 12]) & chem_relaxed["guidance_factor"].isin([2, 5])].sort_values(["condition", "guidance_factor"]),
    ["guidance_factor", "condition", "n_families", "largest_family_fraction", "effective_num_families", "hhi"]
)}

### 7.2 Structural-family concentration

{md_table(
    struct_relaxed[struct_relaxed["condition"].isin([8, 12]) & struct_relaxed["guidance_factor"].isin([2, 5])].sort_values(["condition", "guidance_factor"]),
    ["guidance_factor", "condition", "n_families", "largest_family_fraction", "effective_num_families", "hhi"]
)}

### 7.3 SUN-positive structural concentration

{md_table(
    sun_struct[sun_struct["condition"].isin([8, 12]) & sun_struct["guidance_factor"].isin([2, 5])].sort_values(["condition", "guidance_factor"]),
    ["guidance_factor", "condition", "n_subset", "n_families", "largest_family_fraction", "effective_num_families", "hhi"]
)}

Interpretive summary for the focus groups:

- compare whether `g=5` lowers `n_families` and `effective_num_families`
- compare whether `g=5` raises `largest_family_fraction` and `HHI`
- compare whether `g=5` lowers mean pairwise composition distance or raises same-family pair fractions
- inspect whether the SUN-positive subset itself becomes concentrated even when the aggregate `frac_novel_unique_stable` does not collapse

## 8. Top Families

The complete top-family tables are stored in:

- `{OUTDIR / "top_families_by_group.csv"}`

Below is an excerpt for the key groups (`g=2/5`, `target=8/12`) and the two most directly interpretable family definitions (`chemical_system` and `structure_family`):

{md_table(
    top_focus.sort_values(["family_type", "condition", "guidance_factor", "rank"]),
    ["guidance_factor", "condition", "subset", "family_type", "rank", "family_label", "count", "fraction", "representative_formula", "representative_space_group", "mean_predicted_dielectric_scalar", "mean_energy_above_hull_per_atom", "stable_fraction_within_family", "sun_fraction_within_family"],
    max_rows=80
)}

## 9. Plots Written

Plots saved under:

- `{PLOTDIR}`

Produced files:

- `chem_system_effective_num_all_relaxed.png`
- `chem_system_hhi_all_relaxed.png`
- `structure_family_effective_num_all_relaxed.png`
- `chemical_system_rank_curves_all_relaxed_key_groups.png`
- `structure_family_rank_curves_all_relaxed_key_groups.png`
- `structure_family_rank_curves_sun_key_groups.png`

These figures provide:

- chemical-system concentration trends across all targets and guidance factors
- structural-family concentration trends across all targets and guidance factors
- rank-frequency curves for the main `g=2` versus `g=5` comparisons

## 10. Main Conclusion

**{conclusion}**

Interpretation guide:

- `Supported` means the audit found a consistent increase in family concentration and/or a decrease in effective family diversity at high guidance, especially in the `g=2` vs `g=5` comparisons for `target=8` and `target=12`.
- `Not supported` means those diagnostics did not systematically worsen at high guidance.
- `Inconclusive` means the evidence was mixed across chemical-system, structural-family, pairwise-composition, and SUN-subset diagnostics.

## 11. Caveats

- The current environment does not provide `matminer`, so continuous fingerprint-based structure distances were not computed.
- The structural-family analysis therefore rests on StructureMatcher clusters and same-family pair fractions rather than on `CrystalNNFingerprint` / `SiteStatsFingerprint` embeddings.
- The corrected adapter tables preserve values from the original merged evaluation, but the original raw `structure_index` column in `adapter_evaluation_master.csv` was global `0..2047`; for any within-group structure audit the corrected `local_structure_index` in `{MASTER_CSV}` should be used instead.
"""
    (OUTDIR / "mattergen_guidance_family_audit_2026-08-01.md").write_text(report)


def main() -> None:
    master = load_master()
    structures = load_structures()
    full_df = build_structure_df(master, structures)

    family_df, top_df, subset_df = analyze_groups(full_df)

    family_df.to_csv(OUTDIR / "family_diversity_tables.csv", index=False)
    top_df.to_csv(OUTDIR / "top_families_by_group.csv", index=False)
    subset_df.to_csv(OUTDIR / "group_accounting_and_pairwise_diagnostics.csv", index=False)

    plot_heatmap(
        family_df,
        family_type="chemical_system",
        subset="all_relaxed",
        value_col="effective_num_families",
        title="Chemical-system effective family number (all relaxed)",
        filename="chem_system_effective_num_all_relaxed.png",
    )
    plot_heatmap(
        family_df,
        family_type="chemical_system",
        subset="all_relaxed",
        value_col="hhi",
        title="Chemical-system HHI (all relaxed)",
        filename="chem_system_hhi_all_relaxed.png",
    )
    plot_heatmap(
        family_df,
        family_type="structure_family",
        subset="all_relaxed",
        value_col="effective_num_families",
        title="StructureMatcher effective family number (all relaxed)",
        filename="structure_family_effective_num_all_relaxed.png",
    )
    plot_rank_curves(
        top_df,
        family_type="chemical_system",
        subset="all_relaxed",
        filename="chemical_system_rank_curves_all_relaxed_key_groups.png",
        title="Chemical-system family rank curves: key groups",
    )
    plot_rank_curves(
        top_df,
        family_type="structure_family",
        subset="all_relaxed",
        filename="structure_family_rank_curves_all_relaxed_key_groups.png",
        title="StructureMatcher family rank curves: key groups",
    )
    plot_rank_curves(
        top_df,
        family_type="structure_family",
        subset="sun_positive",
        filename="structure_family_rank_curves_sun_key_groups.png",
        title="StructureMatcher family rank curves: SUN-positive key groups",
    )

    write_report(family_df, top_df, subset_df)


if __name__ == "__main__":
    main()
