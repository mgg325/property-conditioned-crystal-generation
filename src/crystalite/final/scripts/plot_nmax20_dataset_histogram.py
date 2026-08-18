from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_PICKLE = Path("external/anisonet/dataset/train_dataset.p")
OUTPUT_DIR = Path(__file__).resolve().parents[4] / "outputs/crystalite_figures"
PNG_OUT = OUTPUT_DIR / "anisonet_dataset_nmax20_histogram.png"
PDF_OUT = OUTPUT_DIR / "anisonet_dataset_nmax20_histogram.pdf"


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 16,
        }
    )

    df = pickle.load(open(DATASET_PICKLE, "rb"))

    dielectric = np.asarray(df["dielectric_scalar"], dtype=float)
    n_sites = np.asarray(df["structure"].map(len), dtype=int)
    keep_mask = n_sites <= 20

    original = dielectric[np.isfinite(dielectric)]
    filtered = dielectric[np.isfinite(dielectric) & keep_mask]

    bins = np.arange(1.0, 16.0 + 1.0, 1.0)
    counts_original, edges = np.histogram(original, bins=bins)
    counts_filtered, _ = np.histogram(filtered, bins=bins)
    widths = np.diff(edges)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 5.8))

    # Use the reported histogram style.
    ax.bar(
        edges[:-1],
        counts_original,
        width=widths,
        align="edge",
        color="#c7b6e2",
        alpha=0.75,
        linewidth=0,
        label=f"original (n = {len(original):,})",
        zorder=1,
    )
    ax.bar(
        edges[:-1],
        counts_filtered,
        width=widths,
        align="edge",
        color="#6f42c1",
        alpha=0.82,
        linewidth=0,
        label=f"retained after nmax = 20 (n = {len(filtered):,})",
        zorder=2,
    )

    ax.set_xlabel("AnisoNet-labelled dielectric scalar interval")
    ax.set_ylabel("Number of structures")
    ax.set_xlim(1, 16)
    ax.set_xticks(np.arange(1, 16, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")

    print(f"Saved: {PNG_OUT}")
    print(f"Saved: {PDF_OUT}")
    print(f"Original count: {len(original)}")
    print(f"Retained count: {len(filtered)}")


if __name__ == "__main__":
    main()
