from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import FIGURES_DIR


OUTPUT_PATH = FIGURES_DIR / "search_space_reduction.png"
POSITIONS = ["d1", "op", "d2", "=", "d3", "valid"]


def beam_counts(beam_width: int) -> tuple[list[int], list[int]]:
    candidate_set_sizes = [10, 2, 10, 1, 10]
    beam = 1
    expanded = []
    retained = []
    for size in candidate_set_sizes:
        count = beam * size
        expanded.append(count)
        beam = min(beam_width, count)
        retained.append(beam)
    expanded.append(beam)
    retained.append(min(beam_width, 110))
    return expanded, retained


def exhaustive_counts() -> tuple[list[int], list[int]]:
    syntactic_prefixes = [10, 20, 200, 200, 2000, 110]
    retained = [10, 20, 200, 200, 110, 110]
    return syntactic_prefixes, retained


def draw_panel(
    ax: plt.Axes,
    title: str,
    expanded: list[int],
    retained: list[int],
    note: str,
    highlight_final: bool = False,
) -> None:
    x = np.arange(len(POSITIONS))
    width = 0.34
    bar_color = "#D8E2F3" if not highlight_final else "#E8E8E8"
    line_color = "#2454A6" if not highlight_final else "#444444"

    ax.bar(
        x - width / 2,
        expanded,
        width=width,
        color=bar_color,
        edgecolor="#61758F",
        linewidth=0.8,
        label="Expanded/evaluated",
    )
    ax.bar(
        x + width / 2,
        retained,
        width=width,
        color="#8EC9A8",
        edgecolor="#2F7D4F",
        linewidth=0.8,
        label="Retained",
    )
    ax.plot(x + width / 2, retained, color=line_color, marker="o", linewidth=1.4)

    if highlight_final:
        ax.annotate(
            "110 valid equations",
            xy=(5 + width / 2, retained[-1]),
            xytext=(2.55, 360),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#555555"},
            fontsize=8.4,
            color="#333333",
        )

    for idx, value in enumerate(expanded):
        ax.text(
            idx - width / 2,
            value * 1.12,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#3D4852",
        )
    for idx, value in enumerate(retained):
        ax.text(
            idx + width / 2,
            value * 1.12,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#1F5136",
        )

    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(POSITIONS, fontsize=8.7)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 3500)
    ax.grid(True, axis="y", which="both", alpha=0.22)
    ax.text(
        0.02,
        0.96,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color="#3D4852",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), sharey=True)

    expanded_b1, retained_b1 = beam_counts(1)
    draw_panel(
        axes[0],
        "B = 1 (greedy)",
        expanded_b1,
        retained_b1,
        "aggressive pruning\n34 candidates/expr",
    )

    expanded_b5, retained_b5 = beam_counts(5)
    draw_panel(
        axes[1],
        "B = 5",
        expanded_b5,
        retained_b5,
        "moderate breadth\n130 candidates/expr",
    )

    expanded_inf, retained_inf = exhaustive_counts()
    draw_panel(
        axes[2],
        "B = infinity (exhaustive)",
        expanded_inf,
        retained_inf,
        "no beam pruning\nall valid equations retained",
        highlight_final=True,
    )

    axes[0].set_ylabel("Symbolic objects considered (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Beam width controls how much of the structured search space is retained",
        fontsize=13.5,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.045,
        "The exhaustive reference scores the 110 valid equations directly; the unpruned syntax tree would contain 2,000 complete assignments before arithmetic filtering.",
        ha="center",
        fontsize=8.6,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
