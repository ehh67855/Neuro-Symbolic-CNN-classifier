"""Final reproducible experiment pipeline for the term-paper results.

This script is the canonical source for the reported evaluation. It loads the
saved CNN checkpoint, rebuilds the fixed-seed MNIST-test expression set, runs
the constrained beam-search decoder, and regenerates the paper metrics and
figures. Older scripts such as evaluation.py and demo.py are exploratory
one-off entry points and are not used for the final reported numbers.
"""

import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from beam_decoder import BeamSearchDecoder
from project_paths import PROJECT_ROOT


SEED = 0
NUM_EXPRESSIONS = 5000
GAUSSIAN_SIGMAS = [0.0, 0.2, 0.4, 0.6]
CORRUPTION_PROBS = [0.0, 0.1, 0.2, 0.3]
CLASS_LABELS = [str(i) for i in range(10)] + ["+", "-", "="]
ROOT = PROJECT_ROOT
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = ROOT / "symbol_classifier.pth"
METRICS_PATH = ROOT / "paper_metrics.json"
RESULTS_FIGURE_PATH = ROOT / "paper_results.png"
PIPELINE_FIGURE_PATH = ROOT / "pipeline_diagram.png"
EXAMPLE_FIGURE_PATH = ROOT / "correction_example.png"
DATASET_FIGURE_PATH = ROOT / "dataset_samples.png"
SEARCH_SPACE_FIGURE_PATH = ROOT / "search_space_reduction.png"
ERROR_MODES_FIGURE_PATH = ROOT / "error_modes.png"
SUCCESS_FAILURE_FIGURE_PATH = ROOT / "success_failure_examples.png"
ERROR_TOP_K = 3
ERROR_ANALYSIS_SIGMA = 0.4
BEAM_WIDTH = 20
PAPER_ASSET_PATHS = [
    ROOT / "paper.tex",
    METRICS_PATH,
    RESULTS_FIGURE_PATH,
    PIPELINE_FIGURE_PATH,
    EXAMPLE_FIGURE_PATH,
    DATASET_FIGURE_PATH,
    SEARCH_SPACE_FIGURE_PATH,
    ERROR_MODES_FIGURE_PATH,
    SUCCESS_FAILURE_FIGURE_PATH,
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def create_operator_image(symbol: str, size=(28, 28)) -> np.ndarray:
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    if symbol == "+":
        draw.line((14, 6, 14, 22), fill=255, width=2)
        draw.line((6, 14, 22, 14), fill=255, width=2)
    elif symbol == "-":
        draw.line((6, 14, 22, 14), fill=255, width=2)
    elif symbol == "=":
        draw.line((6, 12, 22, 12), fill=255, width=2)
        draw.line((6, 16, 22, 16), fill=255, width=2)
    return np.array(img, dtype=np.float32) / 255.0


def build_expression_dataset(num_samples: int = NUM_EXPRESSIONS):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = torchvision.datasets.MNIST(
        root=str(ROOT / "data"), train=False, download=True, transform=transform
    )
    digit_images = mnist_test.data.numpy().astype(np.float32) / 255.0
    digit_labels = mnist_test.targets.numpy()

    plus_img = create_operator_image("+")
    minus_img = create_operator_image("-")
    equals_img = create_operator_image("=")

    label_to_indices = {
        digit: np.where(digit_labels == digit)[0] for digit in range(10)
    }

    expressions = []
    labels = []
    metadata = []

    for _ in range(num_samples):
        op = random.choice(["+", "-"])
        if op == "+":
            a = random.randint(0, 4)
            b = random.randint(0, 4)
            c = a + b
            op_img = plus_img
            op_class = 10
        else:
            a = random.randint(5, 9)
            b = random.randint(0, 4)
            c = a - b
            op_img = minus_img
            op_class = 11

        a_img = digit_images[random.choice(label_to_indices[a])]
        b_img = digit_images[random.choice(label_to_indices[b])]
        c_img = digit_images[random.choice(label_to_indices[c])]

        expressions.append(
            np.concatenate([a_img, op_img, b_img, equals_img, c_img], axis=1)
        )
        labels.append([a, op_class, b, 12, c])
        metadata.append({"a": a, "op": op, "b": b, "c": c})

    return np.array(expressions), np.array(labels), metadata


class SymbolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 13)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def load_model() -> SymbolClassifier:
    model = SymbolClassifier()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_symbol_test_accuracy(model: SymbolClassifier) -> float:
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = torchvision.datasets.MNIST(
        root=str(ROOT / "data"), train=False, download=True, transform=transform
    )
    digit_images = mnist_test.data.numpy().astype(np.float32) / 255.0
    digit_labels = mnist_test.targets.numpy()

    operator_images = np.concatenate(
        [
            np.repeat(create_operator_image("+")[None, :, :], 100, axis=0),
            np.repeat(create_operator_image("-")[None, :, :], 100, axis=0),
            np.repeat(create_operator_image("=")[None, :, :], 100, axis=0),
        ],
        axis=0,
    )
    operator_labels = np.array([10] * 100 + [11] * 100 + [12] * 100)

    images = np.concatenate([digit_images, operator_images], axis=0)
    labels = np.concatenate([digit_labels, operator_labels], axis=0)

    inputs = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1).cpu().numpy()
    return float((predictions == labels).mean())


def expression_symbol_probs(model: SymbolClassifier, expressions: np.ndarray) -> np.ndarray:
    symbol_images = []
    for slot in range(5):
        start = slot * 28
        end = (slot + 1) * 28
        symbol_images.append(expressions[:, :, start:end])
    symbol_images = np.concatenate(symbol_images, axis=0)
    symbol_tensor = torch.tensor(symbol_images, dtype=torch.float32).unsqueeze(1)

    batch_probs = []
    with torch.no_grad():
        for start in range(0, len(symbol_tensor), 512):
            batch = symbol_tensor[start : start + 512]
            logits = model(batch)
            batch_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(batch_probs, axis=0)
    probs = probs.reshape(5, expressions.shape[0], 13).transpose(1, 0, 2)
    return probs


def build_valid_assignments() -> np.ndarray:
    assignments = []
    for d1 in range(10):
        for op in [10, 11]:
            for d2 in range(10):
                d3 = d1 + d2 if op == 10 else d1 - d2
                if 0 <= d3 < 10:
                    assignments.append([d1, op, d2, 12, d3])
    return np.array(assignments, dtype=np.int64)


def build_syntax_assignments() -> np.ndarray:
    assignments = []
    for d1 in range(10):
        for op in [10, 11]:
            for d2 in range(10):
                for d3 in range(10):
                    assignments.append([d1, op, d2, 12, d3])
    return np.array(assignments, dtype=np.int64)


def score_assignments(probabilities: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
    scores = np.zeros((probabilities.shape[0], len(assignments)), dtype=np.float32)
    for slot in range(5):
        scores += log_probs[:, slot, assignments[:, slot]]
    return scores


def decode_from_assignments(probabilities: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    scores = score_assignments(probabilities, assignments)
    best_indices = scores.argmax(axis=1)
    return assignments[best_indices]


def map_decode(probabilities: np.ndarray, valid_assignments: np.ndarray | None = None) -> np.ndarray:
    """Compatibility wrapper for the neuro-symbolic beam-search decoder."""
    decoder = BeamSearchDecoder(beam_width=BEAM_WIDTH)
    predictions = []
    for probability_rows in probabilities:
        assignment, _ = decoder.decode(probability_rows)
        if assignment is None:
            assignment = [-1] * probabilities.shape[1]
        predictions.append(assignment)
    return np.array(predictions, dtype=np.int64)


def minimal_edit_repair_decode(
    probabilities: np.ndarray,
    seed_predictions: np.ndarray,
    valid_assignments: np.ndarray,
) -> np.ndarray:
    scores = score_assignments(probabilities, valid_assignments)
    hamming = (seed_predictions[:, None, :] != valid_assignments[None, :, :]).sum(axis=2)
    min_distance = hamming.min(axis=1, keepdims=True)
    masked_scores = np.where(hamming == min_distance, scores, -np.inf)
    best_indices = masked_scores.argmax(axis=1)
    return valid_assignments[best_indices]


def compute_metrics(predictions: np.ndarray, labels: np.ndarray):
    return {
        "symbol_accuracy": float((predictions == labels).mean()),
        "expression_accuracy": float((predictions == labels).all(axis=1).mean()),
    }


def evaluate_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    valid_assignments: np.ndarray,
    syntax_assignments: np.ndarray,
    baseline_predictions: np.ndarray | None = None,
):
    if baseline_predictions is None:
        baseline_predictions = probabilities.argmax(axis=2)

    repair_predictions = minimal_edit_repair_decode(
        probabilities, baseline_predictions, valid_assignments
    )
    syntax_predictions = decode_from_assignments(probabilities, syntax_assignments)
    neuro_predictions = map_decode(probabilities, valid_assignments)

    metrics = {
        "baseline": compute_metrics(baseline_predictions, labels),
        "repair": compute_metrics(repair_predictions, labels),
        "syntax_only": compute_metrics(syntax_predictions, labels),
        "neuro_symbolic": compute_metrics(neuro_predictions, labels),
    }

    predictions = {
        "baseline": baseline_predictions,
        "repair": repair_predictions,
        "syntax_only": syntax_predictions,
        "neuro_symbolic": neuro_predictions,
    }
    return metrics, predictions


def analyze_error_modes(
    probabilities: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    valid_assignments: np.ndarray,
    top_k: int = ERROR_TOP_K,
):
    failures = ~(predictions == labels).all(axis=1)
    failure_indices = np.where(failures)[0]
    if len(failure_indices) == 0:
        return {
            "top_k": top_k,
            "num_failures": 0,
            "percent_of_failures": {
                "neural_failure": 0.0,
                "symbolic_ambiguity": 0.0,
                "extreme_noise_collapse": 0.0,
            },
        }

    topk_indices = np.argsort(probabilities, axis=2)[:, :, -top_k:]
    true_in_topk = np.array(
        [
            all(labels[i, slot] in topk_indices[i, slot] for slot in range(5))
            for i in range(len(labels))
        ]
    )

    max_confidence = probabilities.max(axis=2).mean(axis=1)
    valid_scores = score_assignments(probabilities, valid_assignments)
    assignment_to_index = {
        tuple(assignment.tolist()): idx for idx, assignment in enumerate(valid_assignments)
    }
    true_assignment_indices = np.array(
        [assignment_to_index[tuple(label.tolist())] for label in labels], dtype=np.int64
    )
    true_scores = valid_scores[np.arange(len(labels)), true_assignment_indices]
    best_scores = valid_scores.max(axis=1)
    score_margin = best_scores - true_scores

    counts = {
        "neural_failure": 0,
        "symbolic_ambiguity": 0,
        "extreme_noise_collapse": 0,
    }

    for index in failure_indices:
        if not true_in_topk[index]:
            counts["neural_failure"] += 1
        elif max_confidence[index] < 0.45 or score_margin[index] > 3.0:
            counts["extreme_noise_collapse"] += 1
        else:
            counts["symbolic_ambiguity"] += 1

    num_failures = len(failure_indices)
    return {
        "top_k": top_k,
        "num_failures": int(num_failures),
        "average_max_confidence_on_failures": float(max_confidence[failure_indices].mean()),
        "average_score_margin_on_failures": float(score_margin[failure_indices].mean()),
        "percent_of_failures": {
            key: float(value / num_failures) for key, value in counts.items()
        },
        "counts": counts,
        "true_equation_in_topk_rate_on_failures": float(true_in_topk[failure_indices].mean()),
    }


def evaluate_all(model: SymbolClassifier, expressions: np.ndarray, labels: np.ndarray):
    valid_assignments = build_valid_assignments()
    syntax_assignments = build_syntax_assignments()
    clean_probs = expression_symbol_probs(model, expressions)
    clean_metrics, clean_predictions = evaluate_predictions(
        clean_probs, labels, valid_assignments, syntax_assignments
    )

    results = {
        "clean": clean_metrics,
        "gaussian": {},
        "corruption": {},
    }
    artifacts = {"clean": clean_predictions, "gaussian": {}, "corruption": {}}

    for sigma in GAUSSIAN_SIGMAS:
        noisy = np.clip(expressions + np.random.normal(0.0, sigma, expressions.shape), 0.0, 1.0)
        probs = expression_symbol_probs(model, noisy)
        metrics, predictions = evaluate_predictions(
            probs, labels, valid_assignments, syntax_assignments
        )
        results["gaussian"][str(sigma)] = metrics
        artifacts["gaussian"][str(sigma)] = {
            "probabilities": probs,
            "predictions": predictions,
        }
        if sigma in {ERROR_ANALYSIS_SIGMA, 0.6}:
            artifacts["gaussian"][str(sigma)]["images"] = noisy.astype(np.float32)

    for corruption_prob in CORRUPTION_PROBS:
        corrupted_baseline = clean_predictions["baseline"].copy()
        mask = np.random.rand(*corrupted_baseline.shape) < corruption_prob
        noise = np.random.randint(0, 12, size=corrupted_baseline.shape)
        corrupted_baseline = np.where(mask, (corrupted_baseline + 1 + noise) % 13, corrupted_baseline)
        metrics, predictions = evaluate_predictions(
            clean_probs,
            labels,
            valid_assignments,
            syntax_assignments,
            baseline_predictions=corrupted_baseline,
        )
        results["corruption"][str(corruption_prob)] = metrics
        artifacts["corruption"][str(corruption_prob)] = {
            "predictions": predictions,
        }

    error_analysis = {
        f"gaussian_{ERROR_ANALYSIS_SIGMA}": analyze_error_modes(
            artifacts["gaussian"][str(ERROR_ANALYSIS_SIGMA)]["probabilities"],
            labels,
            artifacts["gaussian"][str(ERROR_ANALYSIS_SIGMA)]["predictions"]["neuro_symbolic"],
            valid_assignments,
        ),
        "gaussian_0.6": analyze_error_modes(
            artifacts["gaussian"]["0.6"]["probabilities"],
            labels,
            artifacts["gaussian"]["0.6"]["predictions"]["neuro_symbolic"],
            valid_assignments,
        ),
    }

    return results, artifacts, error_analysis


def render_results(results: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    baseline_color = "#3b6fb6"
    neuro_color = "#d95f02"

    axes[0, 0].plot(
        GAUSSIAN_SIGMAS,
        [results["gaussian"][str(s)]["baseline"]["symbol_accuracy"] for s in GAUSSIAN_SIGMAS],
        marker="o",
        linewidth=2,
        color=baseline_color,
        label="Neural baseline",
    )
    axes[0, 0].plot(
        GAUSSIAN_SIGMAS,
        [results["gaussian"][str(s)]["neuro_symbolic"]["symbol_accuracy"] for s in GAUSSIAN_SIGMAS],
        marker="o",
        linewidth=2,
        color=neuro_color,
        label="Neuro-symbolic",
    )
    axes[0, 0].set_title("Symbol Accuracy Under Gaussian Noise")
    axes[0, 0].set_xlabel("Noise standard deviation")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        GAUSSIAN_SIGMAS,
        [results["gaussian"][str(s)]["baseline"]["expression_accuracy"] for s in GAUSSIAN_SIGMAS],
        marker="o",
        linewidth=2,
        color=baseline_color,
        label="Neural baseline",
    )
    axes[0, 1].plot(
        GAUSSIAN_SIGMAS,
        [results["gaussian"][str(s)]["neuro_symbolic"]["expression_accuracy"] for s in GAUSSIAN_SIGMAS],
        marker="o",
        linewidth=2,
        color=neuro_color,
        label="Neuro-symbolic",
    )
    axes[0, 1].set_title("Expression Accuracy Under Gaussian Noise")
    axes[0, 1].set_xlabel("Noise standard deviation")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        CORRUPTION_PROBS,
        [results["corruption"][str(p)]["baseline"]["symbol_accuracy"] for p in CORRUPTION_PROBS],
        marker="o",
        linewidth=2,
        color=baseline_color,
        label="Corrupted baseline",
    )
    axes[1, 0].plot(
        CORRUPTION_PROBS,
        [results["corruption"][str(p)]["neuro_symbolic"]["symbol_accuracy"] for p in CORRUPTION_PROBS],
        marker="o",
        linewidth=2,
        color=neuro_color,
        label="Neuro-symbolic",
    )
    axes[1, 0].set_title("Symbol Accuracy Under Decision Corruption")
    axes[1, 0].set_xlabel("Corruption probability")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        CORRUPTION_PROBS,
        [results["corruption"][str(p)]["baseline"]["expression_accuracy"] for p in CORRUPTION_PROBS],
        marker="o",
        linewidth=2,
        color=baseline_color,
        label="Corrupted baseline",
    )
    axes[1, 1].plot(
        CORRUPTION_PROBS,
        [results["corruption"][str(p)]["neuro_symbolic"]["expression_accuracy"] for p in CORRUPTION_PROBS],
        marker="o",
        linewidth=2,
        color=neuro_color,
        label="Neuro-symbolic",
    )
    axes[1, 1].set_title("Expression Accuracy Under Decision Corruption")
    axes[1, 1].set_xlabel("Corruption probability")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    colors = {
        "input": "#d9e6f2",
        "cnn": "#f9e0ae",
        "probs": "#f7cad0",
        "reasoner": "#d7f2d0",
        "output": "#cde4f9",
    }

    boxes = [
        (0.02, 0.2, 0.16, 0.6, colors["input"], "Input equation image\n$28 \\times 140$ grayscale"),
        (0.23, 0.2, 0.16, 0.6, colors["cnn"], "CNN symbol classifier\n13-way softmax"),
        (0.44, 0.2, 0.16, 0.6, colors["probs"], "Per-slot probabilities\n$d_1, op, d_2, =, d_3$"),
        (0.65, 0.2, 0.16, 0.6, colors["reasoner"], "Constraint-based decoder\nmaximize valid assignment"),
        (0.86, 0.2, 0.12, 0.6, colors["output"], "Final expression\nprediction"),
    ]

    for x, y, w, h, color, label in boxes:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#333333",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrow_y = 0.5
    for start_x in [0.18, 0.39, 0.60, 0.81]:
        ax.annotate(
            "",
            xy=(start_x + 0.04, arrow_y),
            xytext=(start_x, arrow_y),
            arrowprops=dict(arrowstyle="->", lw=2, color="#555555"),
        )

    ax.text(
        0.73,
        0.1,
        "Constraint: keep only assignments where $d_1 \\pm d_2 = d_3$ and the fourth slot is '='",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.savefig(PIPELINE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_dataset_samples(expressions: np.ndarray, labels: np.ndarray) -> None:
    sample_indices = np.linspace(0, len(expressions) - 1, 8, dtype=int)
    fig, axes = plt.subplots(4, 2, figsize=(10, 6))
    fig.patch.set_facecolor("white")

    for ax, index in zip(axes.ravel(), sample_indices):
        ax.imshow(expressions[index], cmap="gray", vmin=0, vmax=1)
        ax.set_title(label_sequence(labels[index]), fontsize=10, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for slot in range(1, 5):
            ax.axvline(slot * 28 - 0.5, color="#c7c7c7", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

    fig.suptitle("Held-out MNIST-test Expression Samples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DATASET_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_search_space_reduction() -> None:
    labels = [
        "All 13-way sequences",
        "Syntax-only template",
        "Valid equations",
    ]
    counts = np.array(
        [13**5, len(build_syntax_assignments()), len(build_valid_assignments())],
        dtype=np.int64,
    )
    notes = [
        "$13^5$ class assignments",
        "$d_1, op, d_2, =, d_3$",
        "$d_1 \\pm d_2 = d_3$",
    ]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("white")
    colors = ["#6b7280", "#3b6fb6", "#d95f02"]
    y_positions = np.arange(len(labels))
    bars = ax.barh(y_positions, counts, color=colors, height=0.55)

    ax.set_xscale("log")
    ax.set_xlim(80, 1_000_000)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate outputs considered by decoder (log scale)")
    ax.set_title("Search Space Reduction from Per-Symbol Classes to Valid Equations")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, count, note in zip(bars, counts, notes):
        ax.text(
            0.74,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}\n{note}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(SEARCH_SPACE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_error_mode_chart(error_analysis: dict) -> None:
    mode_keys = ["neural_failure", "symbolic_ambiguity", "extreme_noise_collapse"]
    mode_labels = [
        "Neural failure\n(absent from top-3)",
        "Symbolic ambiguity\n(competing valid equation)",
        "Noise collapse\n(diffuse valid scores)",
    ]
    conditions = [
        ("gaussian_0.4", "$\\sigma=0.4$"),
        ("gaussian_0.6", "$\\sigma=0.6$"),
    ]
    x_positions = np.arange(len(mode_keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    colors = ["#3b6fb6", "#d95f02"]

    for offset, (key, label) in enumerate(conditions):
        values = [
            100.0 * error_analysis[key]["percent_of_failures"][mode]
            for mode in mode_keys
        ]
        count = error_analysis[key]["num_failures"]
        x = x_positions + (offset - 0.5) * width
        bars = ax.bar(x, values, width=width, color=colors[offset], label=f"{label} ({count} failures)")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 2.0, 103.0),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions, mode_labels)
    ax.set_ylabel("Percent of full-MAP failures")
    ax.set_ylim(0, 108)
    ax.set_title("Failure Mode Distribution Under Gaussian Noise")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ERROR_MODES_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_success_failure_panel(labels: np.ndarray, artifacts: dict) -> None:
    sigma_key = str(ERROR_ANALYSIS_SIGMA)
    gaussian = artifacts["gaussian"][sigma_key]
    noisy_images = gaussian["images"]
    predictions = gaussian["predictions"]
    baseline = predictions["baseline"]
    neuro = predictions["neuro_symbolic"]

    baseline_correct = (baseline == labels).all(axis=1)
    neuro_correct = (neuro == labels).all(axis=1)
    success_candidates = np.where((~baseline_correct) & neuro_correct)[0]
    failure_candidates = np.where(~neuro_correct)[0]

    success_index = int(success_candidates[0]) if len(success_candidates) else int(np.where(neuro_correct)[0][0])
    failure_index = int(failure_candidates[0]) if len(failure_candidates) else int(np.where(~baseline_correct)[0][0])

    cases = [
        ("Recovered by exact MAP", success_index),
        ("Remaining MAP failure", failure_index),
    ]

    fig = plt.figure(figsize=(11, 4.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.35, wspace=0.18)
    fig.patch.set_facecolor("white")

    for row, (title, index) in enumerate(cases):
        ax_img = fig.add_subplot(grid[row, 0])
        ax_img.imshow(noisy_images[index], cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(f"{title} ($\\sigma={ERROR_ANALYSIS_SIGMA}$)", fontsize=12)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for slot in range(1, 5):
            ax_img.axvline(slot * 28 - 0.5, color="#c7c7c7", linewidth=0.8)
        for spine in ax_img.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.8)

        outcome = "correct" if neuro_correct[index] else "incorrect"
        ax_text = fig.add_subplot(grid[row, 1])
        ax_text.axis("off")
        lines = [
            f"True:     {label_sequence(labels[index])}",
            f"Argmax:   {label_sequence(baseline[index])}",
            f"Full MAP: {label_sequence(neuro[index])}",
            "",
            f"Full MAP outcome: {outcome}",
        ]
        ax_text.text(
            0.0,
            0.95,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=11,
            family="monospace",
        )

    fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.1, hspace=0.52, wspace=0.18)
    fig.savefig(SUCCESS_FAILURE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def label_sequence(sequence: np.ndarray) -> str:
    return " ".join(CLASS_LABELS[int(item)] for item in sequence)


def render_example_figure(
    expressions: np.ndarray,
    labels: np.ndarray,
    model: SymbolClassifier,
    valid_assignments: np.ndarray,
) -> None:
    sigma = 0.4
    noisy = np.clip(expressions + np.random.normal(0.0, sigma, expressions.shape), 0.0, 1.0)
    probs = expression_symbol_probs(model, noisy)
    baseline = probs.argmax(axis=2)
    neuro = map_decode(probs, valid_assignments)

    chosen_index = None
    for index in range(len(expressions)):
        if not np.array_equal(baseline[index], labels[index]) and np.array_equal(neuro[index], labels[index]):
            chosen_index = index
            break

    if chosen_index is None:
        chosen_index = 0

    expression = noisy[chosen_index]
    true_label = labels[chosen_index]
    base_label = baseline[chosen_index]
    neuro_label = neuro[chosen_index]
    slot_diffs = np.where(base_label != true_label)[0]
    focus_slot = int(slot_diffs[0]) if len(slot_diffs) else 0

    slot_probs = probs[chosen_index, focus_slot]
    top_indices = slot_probs.argsort()[::-1][:3]

    fig = plt.figure(figsize=(12, 4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])

    ax_img = fig.add_subplot(grid[0, 0])
    ax_img.imshow(expression, cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Noisy expression example ($\\sigma={sigma}$)")
    ax_img.axis("off")
    for slot in range(1, 5):
        ax_img.axvline(slot * 28 - 0.5, color="#cccccc", linewidth=1)

    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis("off")
    lines = [
        f"True:          {label_sequence(true_label)}",
        f"Baseline:      {label_sequence(base_label)}",
        f"Neuro-symbolic:{label_sequence(neuro_label)}",
        "",
        f"Ambiguous slot: position {focus_slot + 1}",
        "Top probabilities:",
    ]
    for rank, cls_idx in enumerate(top_indices, start=1):
        lines.append(f"{rank}. {CLASS_LABELS[int(cls_idx)]}: {slot_probs[cls_idx]:.3f}")
    lines.append("")
    lines.append("Reasoning step:")
    lines.append("The baseline chooses the slot-wise argmax.")
    lines.append("The neuro-symbolic decoder selects the")
    lines.append("highest-probability assignment that still")
    lines.append("forms a valid arithmetic equation.")

    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(EXAMPLE_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics(results: dict, symbol_test_accuracy: float, error_analysis: dict) -> None:
    clean_gain = (
        results["clean"]["neuro_symbolic"]["expression_accuracy"]
        - results["clean"]["baseline"]["expression_accuracy"]
    )
    repair_gain_clean = (
        results["clean"]["repair"]["expression_accuracy"]
        - results["clean"]["baseline"]["expression_accuracy"]
    )
    syntax_gain_clean = (
        results["clean"]["syntax_only"]["expression_accuracy"]
        - results["clean"]["baseline"]["expression_accuracy"]
    )
    gaussian_gains = [
        results["gaussian"][str(s)]["neuro_symbolic"]["expression_accuracy"]
        - results["gaussian"][str(s)]["baseline"]["expression_accuracy"]
        for s in GAUSSIAN_SIGMAS
    ]
    corruption_gains = [
        results["corruption"][str(p)]["neuro_symbolic"]["expression_accuracy"]
        - results["corruption"][str(p)]["baseline"]["expression_accuracy"]
        for p in CORRUPTION_PROBS
    ]
    nontrivial_gains = gaussian_gains[1:] + corruption_gains[1:]
    parameter_count = sum(parameter.numel() for parameter in load_model().parameters())
    valid_assignment_count = int(len(build_valid_assignments()))
    syntax_assignment_count = int(len(build_syntax_assignments()))
    unconstrained_assignment_count = int(13**5)

    selected_conditions = {
        "clean": results["clean"],
        "gaussian_0.4": results["gaussian"]["0.4"],
        "gaussian_0.6": results["gaussian"]["0.6"],
        "corruption_0.2": results["corruption"]["0.2"],
    }

    payload = {
        "seed": SEED,
        "beam_width": BEAM_WIDTH,
        "num_expressions": NUM_EXPRESSIONS,
        "gaussian_sigmas": GAUSSIAN_SIGMAS,
        "corruption_probs": CORRUPTION_PROBS,
        "search_space": {
            "unconstrained_class_assignments": unconstrained_assignment_count,
            "syntax_only_assignments": syntax_assignment_count,
            "full_valid_assignments": valid_assignment_count,
        },
        "symbol_test_accuracy": symbol_test_accuracy,
        "trainable_parameters": parameter_count,
        "results": results,
        "selected_conditions": selected_conditions,
        "error_analysis": error_analysis,
        "summary": {
            "clean_expression_gain": clean_gain,
            "clean_repair_gain": repair_gain_clean,
            "clean_syntax_only_gain": syntax_gain_clean,
            "average_gaussian_expression_gain": float(np.mean(gaussian_gains)),
            "average_corruption_expression_gain": float(np.mean(corruption_gains)),
            "average_overall_expression_gain": float(np.mean(gaussian_gains + corruption_gains)),
            "average_nontrivial_expression_gain": float(np.mean(nontrivial_gains)),
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def copy_final_assets_to_output() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for path in PAPER_ASSET_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Expected generated asset is missing: {path}")
        shutil.copy2(path, OUTPUT_DIR / path.name)


def main() -> None:
    set_seed()
    expressions, labels, _ = build_expression_dataset()
    model = load_model()
    symbol_test_accuracy = evaluate_symbol_test_accuracy(model)
    valid_assignments = build_valid_assignments()
    results, artifacts, error_analysis = evaluate_all(model, expressions, labels)
    render_results(results)
    render_pipeline_diagram()
    render_dataset_samples(expressions, labels)
    render_search_space_reduction()
    render_error_mode_chart(error_analysis)
    render_success_failure_panel(labels, artifacts)
    render_example_figure(expressions, labels, model, valid_assignments)
    save_metrics(results, symbol_test_accuracy, error_analysis)
    copy_final_assets_to_output()

    print(f"Generated assets in {ROOT}")
    for path in PAPER_ASSET_PATHS:
        print(f"- {path.name}")
    print(f"Copied final assets to {OUTPUT_DIR}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
