from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from project_paths import PROJECT_ROOT

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "beam_exhaustive_noise_results.json"
ANALYSIS_JSON_PATH = PROJECT_ROOT / "search_behavior_analysis.json"
PLATEAU_CSV_PATH = PROJECT_ROOT / "search_plateaus.csv"
PRUNING_CSV_PATH = PROJECT_ROOT / "search_pruning_rates.csv"
SURVIVAL_CSV_PATH = PROJECT_ROOT / "search_survival_rates.csv"
FINAL_BEAM_CSV_PATH = PROJECT_ROOT / "final_beam_composition.csv"
HYPOTHESIS_CSV_PATH = PROJECT_ROOT / "recoverability_hypothesis.csv"
ACCURACY_HEATMAP_PATH = PROJECT_ROOT / "beam_accuracy_heatmap.png"
ORACLE_VS_ACTUAL_PATH = PROJECT_ROOT / "oracle_vs_actual_accuracy.png"
RANK_BOXPLOT_PATH = PROJECT_ROOT / "correct_rank_boxplot.png"
POSITION_KEYS = ["0", "1", "2", "3", "4", "5"]


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_results(results: dict) -> dict:
    table = results["table"]
    conditions = results["conditions"]
    noise_levels = sorted(conditions.keys(), key=float)

    plateau_rows = compute_plateaus(table, noise_levels)
    pruning_rows = []
    survival_rows = []
    final_beam_rows = []
    hypothesis_rows = []

    for noise in noise_levels:
        width_keys = sorted(
            conditions[noise].keys(),
            key=lambda value: float("inf") if value == "infinity" else int(value),
        )
        finite_widths = [width for width in width_keys if width != "infinity"]

        for width in width_keys:
            records = conditions[noise][width]["per_expression"]
            pruning_rows.append(pruning_stats(noise, width, records))
            survival_rows.extend(survival_stats(noise, width, records))
            final_beam_rows.append(final_beam_stats(noise, width, records))

        for previous_width, next_width in zip(finite_widths, finite_widths[1:]):
            hypothesis_rows.append(
                recoverability_step_stats(
                    noise,
                    previous_width,
                    next_width,
                    conditions[noise][previous_width]["per_expression"],
                    conditions[noise][next_width]["per_expression"],
                )
            )

    analysis = {
        "config": {
            "source_seed": results.get("config", {}).get("seed"),
            "num_expressions": results.get("config", {}).get("num_expressions"),
            "noise_levels": [float(noise) for noise in noise_levels],
            "beam_widths": results.get("config", {}).get("beam_widths", []),
        },
        "definitions": {
            "plateau_width": (
                "The smallest finite beam width whose equation accuracy matches exhaustive "
                "accuracy for that noise level within numerical tolerance."
            ),
            "never_in_topk": (
                "The true first-symbol prefix was not retained after the first beam step. "
                "Because all first-position candidates are expanded, this means the correct "
                "prefix was outside the initial retained beam."
            ),
            "pruned_too_early": (
                "The true prefix survived at least one earlier position but was removed at "
                "positions 1-4 before the final validity filter."
            ),
            "oracle_accuracy": (
                "The fraction of examples where the true equation remains in the final valid "
                "beam. The gap between oracle and actual accuracy is a ranking failure."
            ),
        },
        "plateaus": plateau_rows,
        "pruning_rates": pruning_rows,
        "survival_rates": survival_rows,
        "final_beam_composition": final_beam_rows,
        "recoverability_hypothesis": hypothesis_rows,
    }
    return analysis


def compute_plateaus(table: Sequence[dict], noise_levels: Sequence[str]) -> list[dict]:
    rows = []
    for noise in noise_levels:
        condition_rows = [
            row for row in table if float(row["noise_sigma"]) == float(noise)
        ]
        exhaustive = next(row for row in condition_rows if row["beam_width"] == "infinity")
        finite_rows = sorted(
            [row for row in condition_rows if row["beam_width"] != "infinity"],
            key=lambda row: int(row["beam_width"]),
        )

        exhaustive_accuracy = float(exhaustive["equation_accuracy"])
        max_finite_accuracy = max(float(row["equation_accuracy"]) for row in finite_rows)
        plateau_width = first_width_at_accuracy(finite_rows, exhaustive_accuracy)
        max_finite_width = first_width_at_accuracy(finite_rows, max_finite_accuracy)
        within_one_tenth_point = first_width_within(finite_rows, exhaustive_accuracy, 0.001)

        rows.append(
            {
                "noise_sigma": float(noise),
                "exhaustive_accuracy": exhaustive_accuracy,
                "max_finite_accuracy": max_finite_accuracy,
                "plateau_width_matching_exhaustive": plateau_width,
                "first_width_at_max_finite_accuracy": max_finite_width,
                "first_width_within_0_001_of_exhaustive": within_one_tenth_point,
            }
        )
    return rows


def first_width_at_accuracy(rows: Sequence[dict], target: float, tolerance: float = 1e-12):
    for row in rows:
        if abs(float(row["equation_accuracy"]) - target) <= tolerance:
            return int(row["beam_width"])
    return None


def first_width_within(rows: Sequence[dict], target: float, tolerance: float):
    for row in rows:
        if target - float(row["equation_accuracy"]) <= tolerance:
            return int(row["beam_width"])
    return None


def pruning_stats(noise: str, width: str, records: Sequence[dict]) -> dict:
    num_examples = len(records)
    losses = [first_loss_position(record) for record in records]
    counts = {
        "survived": sum(loss is None for loss in losses),
        "never_in_topk": sum(loss == 0 for loss in losses),
        "pruned_too_early": sum(loss in {1, 2, 3, 4} for loss in losses),
        "final_beam_loss": sum(loss == 5 for loss in losses),
    }
    row = {
        "noise_sigma": float(noise),
        "beam_width": width,
        "num_examples": num_examples,
    }
    for key, value in counts.items():
        row[f"{key}_count"] = int(value)
        row[f"{key}_rate"] = safe_rate(value, num_examples)
    for position in POSITION_KEYS:
        count = sum(loss == int(position) for loss in losses)
        row[f"first_loss_pos_{position}_count"] = int(count)
        row[f"first_loss_pos_{position}_rate"] = safe_rate(count, num_examples)
    return row


def first_loss_position(record: dict) -> int | None:
    survival = record.get("prefix_survival", {})
    if not survival:
        return None if record.get("correct_in_final_beam", False) else 5
    for position in POSITION_KEYS:
        if not bool(survival.get(position, False)):
            return int(position)
    return None


def survival_stats(noise: str, width: str, records: Sequence[dict]) -> list[dict]:
    rows = []
    num_examples = len(records)
    for position in POSITION_KEYS:
        if not any(position in record.get("prefix_survival", {}) for record in records):
            continue
        survived = [
            bool(record.get("prefix_survival", {}).get(position, False))
            for record in records
        ]
        rows.append(
            {
                "noise_sigma": float(noise),
                "beam_width": width,
                "position": int(position),
                "survival_rate": float(np.mean(survived)) if survived else 0.0,
                "survived_count": int(sum(survived)),
                "num_examples": num_examples,
            }
        )
    return rows


def final_beam_stats(noise: str, width: str, records: Sequence[dict]) -> dict:
    sizes = np.asarray(
        [record.get("final_valid_candidates", 0) for record in records],
        dtype=np.float64,
    )
    ranks = [
        int(record["correct_rank_in_final_beam"])
        for record in records
        if record.get("correct_rank_in_final_beam") is not None
    ]
    return {
        "noise_sigma": float(noise),
        "beam_width": width,
        "num_examples": int(len(records)),
        "mean_final_valid_candidates": float(np.mean(sizes)) if len(sizes) else 0.0,
        "median_final_valid_candidates": float(np.median(sizes)) if len(sizes) else 0.0,
        "p90_final_valid_candidates": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
        "max_final_valid_candidates": int(np.max(sizes)) if len(sizes) else 0,
        "correct_present_count": int(len(ranks)),
        "correct_present_rate": safe_rate(len(ranks), len(records)),
        "mean_correct_rank": float(np.mean(ranks)) if ranks else None,
        "median_correct_rank": float(np.median(ranks)) if ranks else None,
        "p90_correct_rank": float(np.percentile(ranks, 90)) if ranks else None,
        "max_correct_rank": int(np.max(ranks)) if ranks else None,
    }


def recoverability_step_stats(
    noise: str,
    previous_width: str,
    next_width: str,
    previous_records: Sequence[dict],
    next_records: Sequence[dict],
) -> dict:
    recovered = []
    recovered_after_pruned = 0
    recovered_after_never_topk = 0
    recovered_after_ranking_failure = 0
    new_correct = 0

    for previous, current in zip(previous_records, next_records):
        was_correct = bool(previous.get("equation_correct", False))
        now_correct = bool(current.get("equation_correct", False))
        if not was_correct and now_correct:
            new_correct += 1
            previous_loss = first_loss_position(previous)
            if previous_loss == 0:
                recovered_after_never_topk += 1
            elif previous_loss in {1, 2, 3, 4, 5}:
                recovered_after_pruned += 1
            elif previous.get("correct_in_final_beam", False):
                recovered_after_ranking_failure += 1
            recovered.append(int(previous["index"]))

    return {
        "noise_sigma": float(noise),
        "from_beam_width": previous_width,
        "to_beam_width": next_width,
        "new_correct_count": int(new_correct),
        "new_correct_rate": safe_rate(new_correct, len(previous_records)),
        "new_correct_after_never_in_topk_count": int(recovered_after_never_topk),
        "new_correct_after_pruned_too_early_count": int(recovered_after_pruned),
        "new_correct_after_ranking_failure_count": int(recovered_after_ranking_failure),
        "sample_recovered_indices": recovered[:25],
    }


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value):
    if isinstance(value, float):
        return round(value, 6)
    if value is None:
        return ""
    return value


def generate_plots(results: dict) -> None:
    table = results["table"]
    conditions = results["conditions"]
    plot_accuracy_heatmap(table)
    plot_oracle_vs_actual(table)
    plot_rank_boxplot(conditions)


def plot_accuracy_heatmap(table: Sequence[dict]) -> None:
    noise_levels = sorted({float(row["noise_sigma"]) for row in table})
    width_labels = sorted(
        {row["beam_width"] for row in table},
        key=lambda value: float("inf") if value == "infinity" else int(value),
    )
    matrix = np.zeros((len(noise_levels), len(width_labels)))
    for row in table:
        y = noise_levels.index(float(row["noise_sigma"]))
        x = width_labels.index(row["beam_width"])
        matrix[y, x] = float(row["equation_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(width_labels)))
    ax.set_xticklabels(["inf" if label == "infinity" else label for label in width_labels])
    ax.set_yticks(np.arange(len(noise_levels)))
    ax.set_yticklabels([str(noise) for noise in noise_levels])
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Gaussian noise sigma")
    ax.set_title("Equation Accuracy by Beam Width and Noise")

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            color = "white" if matrix[y, x] < 0.55 else "black"
            ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(image, ax=ax, label="Equation accuracy")
    fig.tight_layout()
    fig.savefig(ACCURACY_HEATMAP_PATH, dpi=200)
    plt.close(fig)


def plot_oracle_vs_actual(table: Sequence[dict]) -> None:
    noise_levels = sorted({float(row["noise_sigma"]) for row in table})
    width_labels = sorted(
        {row["beam_width"] for row in table},
        key=lambda value: float("inf") if value == "infinity" else int(value),
    )
    x_positions = np.arange(len(width_labels))
    display_labels = ["inf" if label == "infinity" else label for label in width_labels]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis, noise in zip(axes, noise_levels):
        rows = [
            next(
                row
                for row in table
                if float(row["noise_sigma"]) == noise and row["beam_width"] == width
            )
            for width in width_labels
        ]
        actual = [float(row["equation_accuracy"]) for row in rows]
        oracle = [float(row["oracle_accuracy"]) for row in rows]
        axis.plot(x_positions, actual, marker="o", label="Actual")
        axis.plot(x_positions, oracle, marker="s", linestyle="--", label="Oracle")
        axis.set_title(f"sigma={noise}")
        axis.grid(True, alpha=0.3)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(display_labels)

    axes[0].legend()
    for axis in axes[2:]:
        axis.set_xlabel("Beam width")
    for axis in axes[::2]:
        axis.set_ylabel("Accuracy")
    fig.suptitle("Oracle vs Actual Beam Search Accuracy")
    fig.tight_layout()
    fig.savefig(ORACLE_VS_ACTUAL_PATH, dpi=200)
    plt.close(fig)


def plot_rank_boxplot(conditions: dict) -> None:
    noise_levels = sorted(conditions.keys(), key=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.ravel()

    for axis, noise in zip(axes, noise_levels):
        width_keys = sorted(
            conditions[noise].keys(),
            key=lambda value: float("inf") if value == "infinity" else int(value),
        )
        ranks_by_width = []
        labels = []
        for width in width_keys:
            ranks = [
                int(record["correct_rank_in_final_beam"])
                for record in conditions[noise][width]["per_expression"]
                if record.get("correct_rank_in_final_beam") is not None
            ]
            if ranks:
                ranks_by_width.append(ranks)
                labels.append("inf" if width == "infinity" else width)

        axis.boxplot(ranks_by_width, tick_labels=labels, showfliers=False)
        axis.set_title(f"sigma={float(noise)}")
        axis.grid(True, axis="y", alpha=0.3)
        axis.set_xlabel("Beam width")
        axis.set_ylabel("Correct equation rank")

    fig.suptitle("Distribution of Correct Equation Rank in Final Beam")
    fig.tight_layout()
    fig.savefig(RANK_BOXPLOT_PATH, dpi=200)
    plt.close(fig)


def print_summary(analysis: dict) -> None:
    print("\nPlateau widths:")
    for row in analysis["plateaus"]:
        print(
            f"sigma={row['noise_sigma']:.1f}: "
            f"plateau={row['plateau_width_matching_exhaustive']}, "
            f"exhaustive_acc={row['exhaustive_accuracy']:.4f}"
        )

    print("\nRecoverability summary:")
    for row in analysis["recoverability_hypothesis"]:
        if row["new_correct_count"] == 0:
            continue
        print(
            f"sigma={row['noise_sigma']:.1f}, "
            f"{row['from_beam_width']} -> {row['to_beam_width']}: "
            f"+{row['new_correct_count']} correct "
            f"(never_topk={row['new_correct_after_never_in_topk_count']}, "
            f"pruned={row['new_correct_after_pruned_too_early_count']}, "
            f"ranking={row['new_correct_after_ranking_failure_count']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze beam-search behavior from saved results JSON.")
    parser.add_argument(
        "results_json",
        nargs="?",
        default=DEFAULT_RESULTS_PATH,
        type=Path,
        help="Path to beam_exhaustive_noise_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results_json)
    analysis = analyze_results(results)

    write_json(analysis, ANALYSIS_JSON_PATH)
    write_csv(analysis["plateaus"], PLATEAU_CSV_PATH)
    write_csv(analysis["pruning_rates"], PRUNING_CSV_PATH)
    write_csv(analysis["survival_rates"], SURVIVAL_CSV_PATH)
    write_csv(analysis["final_beam_composition"], FINAL_BEAM_CSV_PATH)
    write_csv(analysis["recoverability_hypothesis"], HYPOTHESIS_CSV_PATH)
    generate_plots(results)
    print_summary(analysis)

    print(f"\nSaved analysis JSON to {ANALYSIS_JSON_PATH}")
    print(f"Saved plateau table to {PLATEAU_CSV_PATH}")
    print(f"Saved pruning table to {PRUNING_CSV_PATH}")
    print(f"Saved survival table to {SURVIVAL_CSV_PATH}")
    print(f"Saved final-beam table to {FINAL_BEAM_CSV_PATH}")
    print(f"Saved recoverability table to {HYPOTHESIS_CSV_PATH}")
    print(f"Saved plots to {ACCURACY_HEATMAP_PATH}, {ORACLE_VS_ACTUAL_PATH}, {RANK_BOXPLOT_PATH}")


if __name__ == "__main__":
    main()
