from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
RESULTS_CSV = ROOT / "beam_exhaustive_noise_results.csv"
OUTPUT_PATH = ROOT / "recoverability_analysis.png"
WIDTHS = [1, 3, 5, 10, 20, 50, 110]
NOISE_LEVELS = [0.4, 0.6]

COLORS = {
    "pruned": "#C8CDD4",
    "ranked_low": "#F3B562",
    "ranked_best": "#58A96A",
}


def load_rows() -> list[dict]:
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(rows: list[dict], noise: float, width: int) -> dict:
    for row in rows:
        if float(row["noise_sigma"]) == noise and row["beam_width"] == str(width):
            return row
    raise KeyError((noise, width))


def category_rates(rows: list[dict], noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranked_best = []
    ranked_low = []
    pruned = []
    for width in WIDTHS:
        row = row_for(rows, noise, width)
        actual = float(row["equation_accuracy"])
        oracle = float(row["oracle_accuracy"])
        ranked_best.append(actual)
        ranked_low.append(max(0.0, oracle - actual))
        pruned.append(max(0.0, 1.0 - oracle))
    return np.asarray(pruned), np.asarray(ranked_low), np.asarray(ranked_best)


def draw_panel(ax: plt.Axes, rows: list[dict], noise: float) -> None:
    x = np.arange(len(WIDTHS))
    pruned, ranked_low, ranked_best = category_rates(rows, noise)

    ax.bar(
        x,
        100.0 * pruned,
        color=COLORS["pruned"],
        edgecolor="white",
        label="Pruned before final beam",
    )
    ax.bar(
        x,
        100.0 * ranked_low,
        bottom=100.0 * pruned,
        color=COLORS["ranked_low"],
        edgecolor="white",
        label="In final beam, ranked below another valid equation",
    )
    ax.bar(
        x,
        100.0 * ranked_best,
        bottom=100.0 * (pruned + ranked_low),
        color=COLORS["ranked_best"],
        edgecolor="white",
        label="Ranked best",
    )

    for idx, width in enumerate(WIDTHS):
        ax.text(
            idx,
            100.0 * (pruned[idx] + ranked_low[idx]) + 100.0 * ranked_best[idx] / 2,
            f"{100.0 * ranked_best[idx]:.1f}",
            ha="center",
            va="center",
            fontsize=7.7,
            color="#12301C",
            fontweight="bold",
        )
        if ranked_low[idx] > 0.025:
            ax.text(
                idx,
                100.0 * pruned[idx] + 100.0 * ranked_low[idx] / 2,
                f"{100.0 * ranked_low[idx]:.1f}",
                ha="center",
                va="center",
                fontsize=7.4,
                color="#5A3400",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(width) for width in WIDTHS])
    ax.set_ylim(0, 100)
    ax.set_title(fr"Gaussian noise $\sigma={noise:.1f}$", fontweight="bold")
    ax.set_xlabel("Beam width")
    ax.grid(True, axis="y", alpha=0.22)


def main() -> None:
    rows = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    for ax, noise in zip(axes, NOISE_LEVELS):
        draw_panel(ax, rows, noise)
    axes[0].set_ylabel("Share of examples (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=8.8,
    )
    fig.suptitle(
        "Recoverability: wider beams reduce pruning, but high noise leaves ranking failures",
        fontsize=13.3,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "Pruned before final beam includes examples where the true prefix was outside the retained beam at any position. The orange segment is the oracle-actual gap.",
        ha="center",
        fontsize=8.4,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.88))
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
