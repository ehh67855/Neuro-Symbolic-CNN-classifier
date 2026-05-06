from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTPUT_PATH = ROOT / "pipeline_diagram.png"

POSITION_LABELS = ["0: d1", "1: op", "2: d2", "3: =", "4: d3"]
X_SPACING = 1.55
BOX_WIDTH = 1.04
BOX_HEIGHT = 0.56
BOX_HALF_WIDTH = BOX_WIDTH / 2
BOX_HALF_HEIGHT = BOX_HEIGHT / 2
TRUE_PREFIXES = {
    "3",
    "3 +",
    "3 + 4",
    "3 + 4 =",
    "3 + 4 = 7",
}
WRONG_PREFIXES = {
    "3 + 5",
    "3 + 5 =",
    "3 + 5 = 8",
}

COLORS = {
    "correct": {"face": "#D7F5E2", "edge": "#238B45"},
    "wrong": {"face": "#FFE5C2", "edge": "#D95F02"},
    "other": {"face": "#EAF1FB", "edge": "#386CB0"},
    "pruned": {"face": "#F1F1F1", "edge": "#9A9A9A"},
}


@dataclass(frozen=True)
class Node:
    label: str
    score: float
    x: float
    y: float
    status: str


def node_status(label: str, pruned: bool = False) -> str:
    if pruned:
        return "pruned"
    if label in TRUE_PREFIXES:
        return "correct"
    if label in WRONG_PREFIXES:
        return "wrong"
    return "other"


def build_nodes() -> list[list[Node]]:
    columns = [
        [
            ("3", -0.04, False),
            ("8", -1.15, False),
            ("2", -1.30, False),
            ("4", -2.10, True),
        ],
        [
            ("3 +", -0.12, False),
            ("8 -", -1.35, False),
            ("2 +", -1.45, False),
            ("3 -", -2.75, True),
        ],
        [
            ("3 + 5", -0.48, False),
            ("3 + 4", -0.70, False),
            ("8 - 1", -1.85, False),
            ("2 + 9", -2.30, True),
        ],
        [
            ("3 + 5 =", -0.50, False),
            ("3 + 4 =", -0.72, False),
            ("8 - 1 =", -1.88, False),
            ("2 + 9 =", -2.35, True),
        ],
        [
            ("3 + 5 = 8", -0.83, False),
            ("3 + 4 = 7", -0.94, False),
            ("8 - 1 = 7", -2.35, False),
            ("3 + 4 = 8", -2.60, True),
        ],
    ]
    y_positions = [2.05, 0.95, -0.15, -1.25]
    nodes: list[list[Node]] = []
    for x_index, column in enumerate(columns):
        column_nodes = []
        for y, (label, score, pruned) in zip(y_positions, column):
            column_nodes.append(
                Node(
                    label=label,
                    score=score,
                    x=float(x_index) * X_SPACING,
                    y=float(y),
                    status=node_status(label, pruned),
                )
            )
        nodes.append(column_nodes)
    return nodes


def draw_node(ax: plt.Axes, node: Node) -> None:
    style = COLORS[node.status]
    alpha = 0.52 if node.status == "pruned" else 1.0
    box = FancyBboxPatch(
        (node.x - BOX_HALF_WIDTH, node.y - BOX_HALF_HEIGHT),
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        linewidth=1.6,
        facecolor=style["face"],
        edgecolor=style["edge"],
        linestyle="--" if node.status == "pruned" else "-",
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        node.x,
        node.y + 0.08,
        node.label,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold" if node.status in {"correct", "wrong"} else "normal",
        color="#1F2933",
        zorder=4,
    )
    ax.text(
        node.x,
        node.y - 0.15,
        f"log p = {node.score:.2f}",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#4B5563",
        zorder=4,
    )


def draw_arrow(
    ax: plt.Axes,
    start: Node,
    end: Node,
    color: str,
    rad: float = 0.0,
    alpha: float = 0.85,
) -> None:
    start_x = start.x + BOX_HALF_WIDTH + 0.06
    end_x = end.x - BOX_HALF_WIDTH - 0.06
    arrow = FancyArrowPatch(
        (start_x, start.y),
        (end_x, end.y),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.85,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)


def find_node(nodes: list[list[Node]], label: str) -> Node:
    for column in nodes:
        for node in column:
            if node.label == label:
                return node
    raise KeyError(label)


def draw_paths(ax: plt.Axes, nodes: list[list[Node]]) -> None:
    correct_path = ["3", "3 +", "3 + 4", "3 + 4 =", "3 + 4 = 7"]
    wrong_path = ["3", "3 +", "3 + 5", "3 + 5 =", "3 + 5 = 8"]
    low_path = ["8", "8 -", "8 - 1", "8 - 1 =", "8 - 1 = 7"]

    for path, color, rad in [
        (low_path, "#7A869A", 0.00),
        (wrong_path, COLORS["wrong"]["edge"], -0.10),
        (correct_path, COLORS["correct"]["edge"], 0.10),
    ]:
        for before, after in zip(path, path[1:]):
            draw_arrow(ax, find_node(nodes, before), find_node(nodes, after), color, rad)


def draw_pruning_cut(ax: plt.Axes, x: float) -> None:
    ax.plot([x - 0.48, x + 0.48], [-0.50, -0.50], color="#B42318", linewidth=1.1)
    ax.text(
        x,
        -0.64,
        "prune",
        ha="center",
        va="top",
        fontsize=7.6,
        color="#B42318",
    )


def main() -> None:
    nodes = build_nodes()
    fig, ax = plt.subplots(figsize=(13.4, 5.9))
    x_positions = [column[0].x for column in nodes]
    first_x = x_positions[0]
    last_x = x_positions[-1]

    draw_paths(ax, nodes)
    for column in nodes:
        for node in column:
            draw_node(ax, node)
    for x in x_positions:
        draw_pruning_cut(ax, x)

    for x, label in zip(x_positions, POSITION_LABELS):
        ax.text(
            x,
            2.95,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#25313D",
        )

    ax.text(
        first_x - 0.62,
        3.62,
        "Beam width B = 3",
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#17202A",
    )
    ax.text(
        first_x - 0.62,
        3.34,
        "Top three partial assignments are retained; faded candidates are discarded.",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#4B5563",
    )
    ax.annotate(
        "validity filter\nranks complete candidates",
        xy=(last_x, 2.20),
        xytext=(last_x - 1.25, 3.62),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
        ha="left",
        fontsize=9,
        color="#333333",
    )

    legend_handles = [
        Patch(
            facecolor=COLORS["correct"]["face"],
            edgecolor=COLORS["correct"]["edge"],
            label="Correct equation path",
        ),
        Patch(
            facecolor=COLORS["wrong"]["face"],
            edgecolor=COLORS["wrong"]["edge"],
            label="High-probability wrong path",
        ),
        Patch(
            facecolor=COLORS["pruned"]["face"],
            edgecolor=COLORS["pruned"]["edge"],
            label="Low-probability pruned candidate",
            linestyle="--",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    ax.set_xlim(first_x - 0.80, last_x + 0.80)
    ax.set_ylim(-1.80, 3.95)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
