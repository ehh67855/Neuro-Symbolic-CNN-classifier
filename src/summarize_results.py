from __future__ import annotations

import csv
import json
from pathlib import Path

from project_paths import RESULTS_DIR


ROOT = RESULTS_DIR
BEAM_RESULTS_CSV = ROOT / "beam_exhaustive_noise_results.csv"
BEAM_RESULTS_JSON = ROOT / "beam_exhaustive_noise_results.json"
EFFICIENCY_CSV = ROOT / "beam_efficiency_results.csv"
FINAL_BEAM_CSV = ROOT / "final_beam_composition.csv"
PRUNING_CSV = ROOT / "search_pruning_rates.csv"
PLATEAU_CSV = ROOT / "search_plateaus.csv"

MAIN_TABLE_CSV = ROOT / "main_results_table.csv"
EFFICIENCY_TABLE_CSV = ROOT / "efficiency_table.csv"
RECOVERABILITY_TABLE_CSV = ROOT / "recoverability_table_sigma_0.4.csv"
SUMMARY_MD = ROOT / "results_summary.md"

FINITE_WIDTHS = ["1", "3", "5", "10", "20", "50", "110"]
ALL_WIDTHS = FINITE_WIDTHS + ["infinity"]
RECOVERABILITY_SIGMA = 0.4


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def row_for(rows: list[dict], noise: float, width: str) -> dict:
    for row in rows:
        if float(row["noise_sigma"]) == float(noise) and row["beam_width"] == width:
            return row
    raise KeyError((noise, width))


def metric_for(rows: list[dict], noise: float, width: str, metric: str) -> float:
    return float(row_for(rows, noise, width)[metric])


def make_main_table(rows: list[dict]) -> list[dict]:
    noise_levels = sorted({float(row["noise_sigma"]) for row in rows})
    table = []
    for noise in noise_levels:
        item = {"noise_sigma": f"{noise:.1f}"}
        for width in ALL_WIDTHS:
            item[f"beam_width_{width}"] = fmt_pct(
                metric_for(rows, noise, width, "equation_accuracy")
            )
        table.append(item)
    return table


def make_efficiency_table(rows: list[dict]) -> list[dict]:
    table = []
    for row in rows:
        label = "exhaustive" if row["beam_width"] == "infinity" else row["beam_width"]
        table.append(
            {
                "beam_width": label,
                "candidates_per_expression": fmt_float(
                    float(row["candidates_per_expression"]), 1
                ),
                "score_ops_per_expression": fmt_float(
                    float(row["score_computations_per_expression"]), 1
                ),
                "time_per_expression_ms": fmt_float(
                    float(row["time_per_expression_ms"]), 4
                ),
                "wall_clock_speedup_vs_exhaustive": fmt_float(
                    float(row["time_speedup_vs_exhaustive"]), 3
                ),
                "score_op_speedup_vs_exhaustive": fmt_float(
                    float(row["score_computation_speedup_vs_exhaustive"]), 3
                ),
            }
        )
    return table


def make_recoverability_table(
    beam_rows: list[dict],
    final_rows: list[dict],
    pruning_rows: list[dict],
    sigma: float = RECOVERABILITY_SIGMA,
) -> list[dict]:
    table = []
    for width in FINITE_WIDTHS:
        final = row_for(final_rows, sigma, width)
        pruning = row_for(pruning_rows, sigma, width)
        beam = row_for(beam_rows, sigma, width)
        table.append(
            {
                "noise_sigma": f"{sigma:.1f}",
                "beam_width": width,
                "correct_in_final_beam_pct": fmt_pct(
                    float(final["correct_present_rate"])
                ),
                "mean_correct_rank": fmt_float(float(final["mean_correct_rank"]), 3),
                "median_correct_rank": fmt_float(float(final["median_correct_rank"]), 1),
                "pruned_before_final_pct": fmt_pct(
                    float(pruning["never_in_topk_rate"])
                    + float(pruning["pruned_too_early_rate"])
                ),
                "equation_accuracy_pct": fmt_pct(float(beam["equation_accuracy"])),
            }
        )
    return table


def validate_against_exhaustive(rows: list[dict]) -> list[dict]:
    validations = []
    for noise in sorted({float(row["noise_sigma"]) for row in rows}):
        b110 = metric_for(rows, noise, "110", "equation_accuracy")
        exhaustive = metric_for(rows, noise, "infinity", "equation_accuracy")
        validations.append(
            {
                "noise_sigma": f"{noise:.1f}",
                "beam_110_accuracy_pct": fmt_pct(b110),
                "exhaustive_accuracy_pct": fmt_pct(exhaustive),
                "matches_accuracy": str(abs(b110 - exhaustive) < 1e-12),
                "absolute_gap_pct_points": f"{100.0 * abs(b110 - exhaustive):.4f}",
            }
        )
    return validations


def prediction_match_summary() -> dict:
    with BEAM_RESULTS_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    comparisons = payload.get("beam_width_comparisons", {})
    summary = {}
    for noise, condition in comparisons.items():
        width_summary = condition.get("summary_by_width", {})
        if "110" in width_summary:
            summary[noise] = {
                "beam_110_match_exhaustive_rate": width_summary["110"][
                    "match_reference_rate"
                ],
                "beam_110_prediction_differences": width_summary["110"][
                    "num_differences_from_reference"
                ],
            }
    return summary


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return "\n".join(lines)


def write_summary(
    main_table: list[dict],
    efficiency_table: list[dict],
    recoverability_table: list[dict],
    validation_table: list[dict],
    plateau_rows: list[dict],
    prediction_matches: dict,
) -> None:
    b10_sigma04 = next(
        row for row in recoverability_table if row["beam_width"] == "10"
    )
    b50_sigma04 = next(
        row for row in main_table if row["noise_sigma"] == "0.4"
    )["beam_width_50"]
    exhaustive_sigma04 = next(
        row for row in main_table if row["noise_sigma"] == "0.4"
    )["beam_width_infinity"]
    b10_efficiency = next(row for row in efficiency_table if row["beam_width"] == "10")

    lines = [
        "# Beam Search Results Summary",
        "",
        "Generated by `summarize_results.py` from the latest experiment artifacts.",
        "",
        "## Key Numbers",
        "",
        f"- At sigma=0.4, beam_width=10 equation accuracy is {b10_sigma04['equation_accuracy_pct']}%.",
        f"- At sigma=0.4, beam_width=10 keeps the correct equation in the final beam for {b10_sigma04['correct_in_final_beam_pct']}% of examples.",
        f"- At sigma=0.4, beam_width=50 reaches {b50_sigma04}% equation accuracy, matching exhaustive at {exhaustive_sigma04}%.",
        f"- Beam_width=10 uses {b10_efficiency['score_ops_per_expression']} score operations/expression versus 550.0 for exhaustive.",
        f"- Beam_width=10 measured wall-clock speedup is {b10_efficiency['wall_clock_speedup_vs_exhaustive']}x in this Python implementation.",
        "",
        "Decision corruption is not part of the current beam-centered evaluation pipeline or paper tables, so it was not regenerated here.",
        "",
        "## Main Results: Equation Accuracy (%)",
        "",
        markdown_table(main_table),
        "",
        "## Efficiency",
        "",
        markdown_table(efficiency_table),
        "",
        "## Recoverability at sigma=0.4",
        "",
        markdown_table(recoverability_table),
        "",
        "## Plateau Widths",
        "",
        markdown_table(plateau_rows),
        "",
        "## Validation: beam_width=110 vs Exhaustive",
        "",
        markdown_table(validation_table),
        "",
        "Prediction-level match rates for beam_width=110 against exhaustive:",
        "",
        "```json",
        json.dumps(prediction_matches, indent=2, sort_keys=True),
        "```",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    beam_rows = read_csv(BEAM_RESULTS_CSV)
    efficiency_rows = read_csv(EFFICIENCY_CSV)
    final_rows = read_csv(FINAL_BEAM_CSV)
    pruning_rows = read_csv(PRUNING_CSV)
    plateau_rows = read_csv(PLATEAU_CSV)

    main_table = make_main_table(beam_rows)
    efficiency_table = make_efficiency_table(efficiency_rows)
    recoverability_table = make_recoverability_table(
        beam_rows, final_rows, pruning_rows
    )
    validation_table = validate_against_exhaustive(beam_rows)
    prediction_matches = prediction_match_summary()

    write_csv(main_table, MAIN_TABLE_CSV)
    write_csv(efficiency_table, EFFICIENCY_TABLE_CSV)
    write_csv(recoverability_table, RECOVERABILITY_TABLE_CSV)
    write_summary(
        main_table,
        efficiency_table,
        recoverability_table,
        validation_table,
        plateau_rows,
        prediction_matches,
    )

    print(f"Saved {MAIN_TABLE_CSV}")
    print(f"Saved {EFFICIENCY_TABLE_CSV}")
    print(f"Saved {RECOVERABILITY_TABLE_CSV}")
    print(f"Saved {SUMMARY_MD}")


if __name__ == "__main__":
    main()
