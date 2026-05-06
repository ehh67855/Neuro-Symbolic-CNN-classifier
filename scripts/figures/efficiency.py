from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import FIGURES_DIR, RESULTS_DIR


EFFICIENCY_CSV = RESULTS_DIR / "beam_efficiency_results.csv"
OUTPUT_PATH = FIGURES_DIR / "efficiency_tradeoff.png"
FINITE_WIDTHS = [1, 3, 5, 10, 20, 50, 110]


def load_rows() -> list[dict]:
    with EFFICIENCY_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    beam_rows = [row for row in rows if row["decoder"] == "beam"]
    beam_rows.sort(key=lambda row: int(row["beam_width"]))
    exhaustive = next(row for row in rows if row["decoder"] == "exhaustive")

    widths = np.asarray([int(row["beam_width"]) for row in beam_rows], dtype=float)
    candidates = np.asarray(
        [float(row["candidates_per_expression"]) for row in beam_rows],
        dtype=float,
    )
    score_ops = np.asarray(
        [float(row["score_computations_per_expression"]) for row in beam_rows],
        dtype=float,
    )
    exhaustive_candidates = float(exhaustive["candidates_per_expression"])
    exhaustive_scores = float(exhaustive["score_computations_per_expression"])

    linear_reference = candidates[0] * widths / widths[0]

    fig, ax = plt.subplots(figsize=(8.7, 5.25))
    ax.plot(
        widths,
        candidates,
        marker="o",
        linewidth=2.1,
        color="#2454A6",
        label="Beam candidates evaluated",
    )
    ax.plot(
        widths,
        score_ops,
        marker="s",
        linewidth=2.1,
        color="#238B45",
        label="Beam score computations",
    )
    ax.plot(
        widths,
        linear_reference,
        linestyle=":",
        linewidth=1.4,
        color="#777777",
        label="Linear reference from B=1",
    )
    ax.axhline(
        exhaustive_candidates,
        color="#2454A6",
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
        label="Exhaustive candidates: 110",
    )
    ax.axhline(
        exhaustive_scores,
        color="#238B45",
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
        label="Exhaustive score ops: 550",
    )

    b10_row = next(row for row in beam_rows if row["beam_width"] == "10")
    b10_scores = float(b10_row["score_computations_per_expression"])
    score_speedup = exhaustive_scores / b10_scores
    ax.annotate(
        f"B=10 uses {b10_scores:.0f} score ops/expr\n{score_speedup:.2f}x fewer than exhaustive",
        xy=(10, b10_scores),
        xytext=(4.0, 900),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    ax.annotate(
        "Finite candidate sets cap growth;\nwide beams eventually saturate.",
        xy=(110, candidates[-1]),
        xytext=(32, 2150),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.85, 135)
    ax.set_ylim(20, 4500)
    ax.set_xticks(FINITE_WIDTHS)
    ax.set_xticklabels([str(width) for width in FINITE_WIDTHS])
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Computation per expression (log scale)")
    ax.set_title("Efficiency tradeoff: beam width vs symbolic computation")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
