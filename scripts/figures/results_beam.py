from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import FIGURES_DIR, RESULTS_DIR


RESULTS_CSV = RESULTS_DIR / "beam_exhaustive_noise_results.csv"
OUTPUT_PATH = FIGURES_DIR / "beam_width_accuracy.png"
FINITE_WIDTHS = [1, 3, 5, 10, 20, 50, 110]
EXHAUSTIVE_X = 160

COLORS = {
    0.0: "#238B45",
    0.2: "#2B6CB0",
    0.4: "#D95F02",
    0.6: "#7B3294",
}


def load_rows() -> list[dict]:
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_for(rows: list[dict], noise: float, width: int | str) -> dict:
    width_key = str(width)
    for row in rows:
        if float(row["noise_sigma"]) == noise and row["beam_width"] == width_key:
            return row
    raise KeyError((noise, width))


def main() -> None:
    rows = load_rows()
    noise_levels = sorted({float(row["noise_sigma"]) for row in rows})

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for noise in noise_levels:
        accuracies = [
            100.0 * float(row_for(rows, noise, width)["equation_accuracy"])
            for width in FINITE_WIDTHS
        ]
        exhaustive_accuracy = 100.0 * float(
            row_for(rows, noise, "infinity")["equation_accuracy"]
        )
        color = COLORS.get(noise, "#333333")
        ax.plot(
            FINITE_WIDTHS,
            accuracies,
            marker="o",
            linewidth=2.1,
            color=color,
            label=fr"$\sigma={noise:.1f}$",
        )
        ax.scatter(
            [EXHAUSTIVE_X],
            [exhaustive_accuracy],
            marker="s",
            s=46,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )
        ax.plot(
            [110, EXHAUSTIVE_X],
            [accuracies[-1], exhaustive_accuracy],
            linestyle=":",
            color=color,
            linewidth=1.0,
            alpha=0.8,
        )

    sigma04_b10 = 100.0 * float(row_for(rows, 0.4, 10)["equation_accuracy"])
    sigma04_exhaustive = 100.0 * float(
        row_for(rows, 0.4, "infinity")["equation_accuracy"]
    )
    ax.annotate(
        f"B=10: {sigma04_b10:.2f}%\nwithin {sigma04_exhaustive - sigma04_b10:.2f} points of exhaustive",
        xy=(10, sigma04_b10),
        xytext=(4.1, 75.5),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8.8,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    ax.annotate(
        "Exhaustive reference\n(beam_width = infinity)",
        xy=(EXHAUSTIVE_X, sigma04_exhaustive),
        xytext=(58, 62),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
        fontsize=8.8,
        ha="left",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    ax.set_xscale("log")
    ax.set_xlim(0.85, 190)
    ax.set_xticks([*FINITE_WIDTHS, EXHAUSTIVE_X])
    ax.set_xticklabels([str(width) for width in FINITE_WIDTHS] + ["infinity"])
    ax.set_ylim(0, 104)
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Equation accuracy (%)")
    ax.set_title("Accuracy improves with beam width and approaches exhaustive decoding")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Gaussian noise", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
