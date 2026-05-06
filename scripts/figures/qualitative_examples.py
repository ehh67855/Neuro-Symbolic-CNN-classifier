from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from beam_decoder import BeamSearchDecoder, is_valid_equation
from evaluation import expression_symbol_probs, load_expression_dataset, load_model
from project_paths import FIGURES_DIR, PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTPUT_PATH = FIGURES_DIR / "qualitative_decoding_examples.png"
SIGMA = 0.4
SEED = 0
BEAM_WIDTH = 10
NUM_EXAMPLES = 4

SYMBOLS = {**{index: str(index) for index in range(10)}, 10: "+", 11: "-", 12: "="}
DIGITS = set(range(10))


def assignment_text(assignment: Sequence[int] | None) -> str:
    if assignment is None:
        return "no prediction"
    return " ".join(SYMBOLS[int(label)] for label in assignment)


def equation_text(assignment: Sequence[int] | None) -> str:
    if assignment is None:
        return "no prediction"
    symbols = [SYMBOLS[int(label)] for label in assignment]
    return f"{symbols[0]} {symbols[1]} {symbols[2]} {symbols[3]} {symbols[4]}"


def is_syntactic_assignment(assignment: Sequence[int]) -> bool:
    return (
        int(assignment[0]) in DIGITS
        and int(assignment[1]) in (10, 11)
        and int(assignment[2]) in DIGITS
        and int(assignment[3]) == 12
        and int(assignment[4]) in DIGITS
    )


def local_error_label(assignment: Sequence[int]) -> str:
    if not is_syntactic_assignment(assignment):
        return "wrong: invalid slot format"
    if not is_valid_equation(assignment):
        return "wrong: violates arithmetic"
    return "wrong: valid but not target"


def decode_beam(probabilities: np.ndarray) -> list[list[int] | None]:
    decoder = BeamSearchDecoder(beam_width=BEAM_WIDTH, recover_final_if_empty=False)
    predictions = []
    for probability_rows in probabilities:
        prediction, _ = decoder.decode(probability_rows)
        predictions.append(prediction)
    return predictions


def select_examples(
    labels: np.ndarray,
    local_predictions: np.ndarray,
    beam_predictions: Sequence[Sequence[int] | None],
) -> list[int]:
    syntax_errors: list[int] = []
    arithmetic_errors: list[int] = []

    for index, (true_label, local_prediction, beam_prediction) in enumerate(
        zip(labels, local_predictions, beam_predictions)
    ):
        true = [int(value) for value in true_label]
        local = [int(value) for value in local_prediction]
        if beam_prediction != true or local == true:
            continue
        if is_syntactic_assignment(local):
            arithmetic_errors.append(index)
        else:
            syntax_errors.append(index)

    selected = syntax_errors[:2] + arithmetic_errors[:2]
    if len(selected) < NUM_EXAMPLES:
        remaining = [
            index
            for index in syntax_errors + arithmetic_errors
            if index not in set(selected)
        ]
        selected.extend(remaining[: NUM_EXAMPLES - len(selected)])
    if len(selected) < NUM_EXAMPLES:
        raise RuntimeError("could not find enough qualitative improvement examples")
    return selected[:NUM_EXAMPLES]


def draw_expression(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_prediction_box(
    ax: plt.Axes,
    header: str,
    prediction: str,
    status: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"{header}\n{prediction}\n{status}",
        ha="center",
        va="center",
        fontsize=9.2,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.08",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 1.4,
        },
    )


def main() -> None:
    images, labels = load_expression_dataset(ROOT / "synthetic_expressions.npz")
    rng = np.random.default_rng(SEED)
    noisy_images = np.clip(
        images + rng.normal(0.0, SIGMA, images.shape),
        0.0,
        1.0,
    )

    model = load_model(ROOT / "symbol_classifier.pth")
    probabilities = expression_symbol_probs(model, noisy_images)
    local_predictions = probabilities.argmax(axis=2).astype(int)
    beam_predictions = decode_beam(probabilities)
    selected_indices = select_examples(labels, local_predictions, beam_predictions)

    fig, axes = plt.subplots(
        NUM_EXAMPLES,
        4,
        figsize=(12.0, 5.8),
        gridspec_kw={"width_ratios": [2.25, 2.25, 1.75, 1.95]},
    )
    column_titles = [
        "Clean held-out image",
        fr"Noisy input ($\sigma={SIGMA}$)",
        "Local CNN argmax",
        fr"Neuro-symbolic beam ($B={BEAM_WIDTH}$)",
    ]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=10.5, pad=10)

    for row, index in enumerate(selected_indices):
        true = [int(value) for value in labels[index]]
        local = [int(value) for value in local_predictions[index]]
        beam = beam_predictions[index]

        draw_expression(axes[row, 0], images[index])
        draw_expression(axes[row, 1], noisy_images[index])
        axes[row, 0].text(
            -0.08,
            0.5,
            f"Target:\n{equation_text(true)}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9.2,
        )

        draw_prediction_box(
            axes[row, 2],
            "CNN predicts",
            assignment_text(local),
            local_error_label(local),
            facecolor="#FFE6D5",
            edgecolor="#D95F02",
        )
        draw_prediction_box(
            axes[row, 3],
            "Decoder returns",
            equation_text(beam),
            "valid and correct",
            facecolor="#DDF3E6",
            edgecolor="#238B45",
        )

    fig.tight_layout(rect=[0.03, 0.0, 1.0, 1.0], h_pad=1.2, w_pad=0.95)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
