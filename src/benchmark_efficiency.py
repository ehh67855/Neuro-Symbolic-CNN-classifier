from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from beam_decoder import BeamSearchDecoder
from evaluation import (
    EPSILON,
    build_valid_assignments,
    expression_symbol_probs,
    load_expression_dataset,
    load_model,
    set_seed,
)
from project_paths import FIGURES_DIR, RESULTS_DIR


BEAM_WIDTHS = [1, 3, 5, 10, 20, 50, 110]
RESULTS_JSON_PATH = RESULTS_DIR / "beam_efficiency_results.json"
RESULTS_CSV_PATH = RESULTS_DIR / "beam_efficiency_results.csv"
SPEEDUP_PLOT_PATH = FIGURES_DIR / "beam_speedup.png"
NUM_EXPRESSIONS = 5000


def benchmark_beam(probabilities: np.ndarray, beam_width: int) -> dict:
    decoder = BeamSearchDecoder(beam_width=beam_width, recover_final_if_empty=False)
    elapsed_seconds = 0.0
    candidates = []
    retained_candidates = []
    score_computations = []
    final_valid_candidates = []
    missing_predictions = 0

    for probability_rows in probabilities:
        prediction, _, stats = decoder.decode_with_stats(probability_rows)
        elapsed_seconds += float(stats["elapsed_seconds"])
        candidates.append(int(stats["total_candidates_evaluated"]))
        retained_candidates.append(int(stats["retained_candidates"]))
        score_computations.append(int(stats["score_computations"]))
        final_valid_candidates.append(int(stats["final_valid_candidates"]))
        missing_predictions += int(prediction is None)

    return {
        "decoder": "beam",
        "beam_width": str(beam_width),
        "num_expressions": int(len(probabilities)),
        "candidates_per_expression": float(np.mean(candidates)),
        "retained_candidates_per_expression": float(np.mean(retained_candidates)),
        "score_computations_per_expression": float(np.mean(score_computations)),
        "time_per_expression_ms": float(1000.0 * elapsed_seconds / len(probabilities)),
        "time_per_5000_expressions_ms": float(1000.0 * elapsed_seconds * NUM_EXPRESSIONS / len(probabilities)),
        "total_time_seconds": float(elapsed_seconds),
        "mean_final_valid_candidates": float(np.mean(final_valid_candidates)),
        "missing_prediction_rate": float(missing_predictions / len(probabilities)),
    }


def benchmark_exhaustive(probabilities: np.ndarray) -> dict:
    valid_assignments = build_valid_assignments()
    start_time = time.perf_counter()
    predictions = []
    for probability_rows in probabilities:
        log_probs = np.log(np.clip(probability_rows, EPSILON, 1.0))
        best_score = -np.inf
        best_assignment = None
        for assignment in valid_assignments:
            score = 0.0
            for position, label in enumerate(assignment):
                score += float(log_probs[position, label])
            if score > best_score:
                best_score = score
                best_assignment = assignment
        predictions.append(best_assignment)
    elapsed_seconds = time.perf_counter() - start_time

    score_computations = len(valid_assignments) * probabilities.shape[1]
    return {
        "decoder": "exhaustive",
        "beam_width": "infinity",
        "num_expressions": int(len(probabilities)),
        "candidates_per_expression": float(len(valid_assignments)),
        "retained_candidates_per_expression": float(len(valid_assignments)),
        "score_computations_per_expression": float(score_computations),
        "time_per_expression_ms": float(1000.0 * elapsed_seconds / len(probabilities)),
        "time_per_5000_expressions_ms": float(1000.0 * elapsed_seconds * NUM_EXPRESSIONS / len(probabilities)),
        "total_time_seconds": float(elapsed_seconds),
        "mean_final_valid_candidates": float(len(valid_assignments)),
        "missing_prediction_rate": 0.0,
    }


def add_speedup_columns(rows: Sequence[dict], exhaustive_row: dict) -> list[dict]:
    enriched = []
    exhaustive_time = float(exhaustive_row["time_per_expression_ms"])
    exhaustive_candidates = float(exhaustive_row["candidates_per_expression"])
    exhaustive_scores = float(exhaustive_row["score_computations_per_expression"])

    for row in rows:
        item = dict(row)
        item["time_speedup_vs_exhaustive"] = safe_divide(
            exhaustive_time,
            float(row["time_per_expression_ms"]),
        )
        item["candidate_reduction_vs_exhaustive"] = safe_divide(
            exhaustive_candidates,
            float(row["candidates_per_expression"]),
        )
        item["score_computation_speedup_vs_exhaustive"] = safe_divide(
            exhaustive_scores,
            float(row["score_computations_per_expression"]),
        )
        enriched.append(item)
    return enriched


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(rows: Sequence[dict], path: Path) -> None:
    fieldnames = [
        "decoder",
        "beam_width",
        "num_expressions",
        "candidates_per_expression",
        "retained_candidates_per_expression",
        "score_computations_per_expression",
        "time_per_expression_ms",
        "time_per_5000_expressions_ms",
        "time_speedup_vs_exhaustive",
        "candidate_reduction_vs_exhaustive",
        "score_computation_speedup_vs_exhaustive",
        "mean_final_valid_candidates",
        "missing_prediction_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value):
    if isinstance(value, float):
        return round(value, 6)
    return value


def plot_speedup(rows: Sequence[dict]) -> None:
    beam_rows = [row for row in rows if row["decoder"] == "beam"]
    widths = [int(row["beam_width"]) for row in beam_rows]
    time_speedups = [float(row["time_speedup_vs_exhaustive"]) for row in beam_rows]
    score_speedups = [
        float(row["score_computation_speedup_vs_exhaustive"])
        for row in beam_rows
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(widths, time_speedups, marker="o", label="Measured wall-clock speedup")
    ax.plot(widths, score_speedups, marker="s", linestyle="--", label="Score-computation speedup")
    ax.axhline(1.0, color="#777777", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels([str(width) for width in widths])
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Speedup vs exhaustive")
    ax.set_title("Beam Search Efficiency vs Exhaustive Enumeration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SPEEDUP_PLOT_PATH, dpi=200)
    plt.close(fig)


def print_table(rows: Sequence[dict]) -> None:
    print("\nComputational efficiency:")
    print(
        "beam_width | expanded+checked/expr | retained/expr | score_ops/expr | "
        "time/expr(ms) | time/5000(ms) | time_speedup | score_speedup"
    )
    print("-" * 128)
    for row in rows:
        print(
            f"{row['beam_width']:<10} | "
            f"{row['candidates_per_expression']:<21.1f} | "
            f"{row['retained_candidates_per_expression']:<13.1f} | "
            f"{row['score_computations_per_expression']:<14.1f} | "
            f"{row['time_per_expression_ms']:<13.4f} | "
            f"{row['time_per_5000_expressions_ms']:<13.2f} | "
            f"{row['time_speedup_vs_exhaustive']:<12.3f} | "
            f"{row['score_computation_speedup_vs_exhaustive']:<.3f}"
        )


def main() -> None:
    set_seed()
    model = load_model()
    images, _ = load_expression_dataset()
    images = images[:NUM_EXPRESSIONS]

    print("Extracting CNN probabilities once for the 5,000-expression benchmark...")
    probabilities = expression_symbol_probs(model, images)

    beam_rows = []
    for beam_width in BEAM_WIDTHS:
        print(f"Benchmarking beam_width={beam_width}")
        beam_rows.append(benchmark_beam(probabilities, beam_width))

    print("Benchmarking exhaustive reference")
    exhaustive_row = benchmark_exhaustive(probabilities)
    rows = add_speedup_columns([*beam_rows, exhaustive_row], exhaustive_row)

    payload = {
        "config": {
            "num_expressions": int(len(probabilities)),
            "beam_widths": BEAM_WIDTHS,
            "exhaustive_candidates_per_expression": int(build_valid_assignments().shape[0]),
            "timing_scope": "symbolic decoding only; CNN probability extraction is excluded",
            "score_computation_definition": (
                "Beam score computations count per-position log-probability additions during "
                "candidate expansion. Exhaustive score computations count 5 log-probability "
                "additions for each of the 110 valid equations."
            ),
            "interpretation_note": (
                "The equation domain is tiny, so the exhaustive reference can be faster "
                "in wall-clock time. Beam search is included to show how search cost can be "
                "controlled for domains whose valid candidate sets are too large to enumerate."
            ),
        },
        "rows": rows,
    }

    write_json(payload, RESULTS_JSON_PATH)
    write_csv(rows, RESULTS_CSV_PATH)
    plot_speedup(rows)
    print_table(rows)
    print(f"\nSaved JSON results to {RESULTS_JSON_PATH}")
    print(f"Saved CSV results to {RESULTS_CSV_PATH}")
    print(f"Saved speedup plot to {SPEEDUP_PLOT_PATH}")


if __name__ == "__main__":
    main()
