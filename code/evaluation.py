from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from beam_decoder import BeamSearchDecoder
from project_paths import PROJECT_ROOT


SEED = 0
MODEL_PATH = PROJECT_ROOT / "symbol_classifier.pth"
DATASET_PATH = PROJECT_ROOT / "synthetic_expressions.npz"
JSON_RESULTS_PATH = PROJECT_ROOT / "beam_exhaustive_noise_results.json"
CSV_RESULTS_PATH = PROJECT_ROOT / "beam_exhaustive_noise_results.csv"
PLOT_PATH = PROJECT_ROOT / "beam_width_accuracy.png"
GAUSSIAN_SIGMAS = [0.0, 0.2, 0.4, 0.6]
BEAM_WIDTHS = [1, 3, 5, 10, 20, 50, 110]
TRACE_SAMPLE_SIZE = 3
TRACE_SAMPLE_WIDTHS = [1, 10, 110]
BATCH_SIZE = 512
EPSILON = 1e-12


class SymbolClassifier(nn.Module):
    def __init__(self):
        super(SymbolClassifier, self).__init__()
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
        x = self.fc2(x)
        return x


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(checkpoint_path: str | Path = MODEL_PATH) -> SymbolClassifier:
    model = SymbolClassifier()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_expression_dataset(dataset_path: str | Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(dataset_path)
    return _normalize_images(data["images"]), np.asarray(data["labels"], dtype=np.int64)


def get_probs(model: SymbolClassifier, expression_img: np.ndarray) -> np.ndarray:
    probs = []
    for slot in range(5):
        symbol_img = expression_img[:, slot * 28 : (slot + 1) * 28]
        symbol_tensor = torch.tensor(symbol_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(symbol_tensor)
            prob = torch.softmax(output, dim=1).squeeze(0).cpu().numpy()
        probs.append(prob)
    return np.asarray(probs, dtype=np.float64)


def expression_symbol_probs(
    model: SymbolClassifier,
    expressions: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Return probabilities with shape (num_expressions, 5, 13)."""
    symbol_images = []
    for slot in range(5):
        symbol_images.append(expressions[:, :, slot * 28 : (slot + 1) * 28])
    symbol_images = np.concatenate(symbol_images, axis=0)
    symbol_tensor = torch.tensor(symbol_images, dtype=torch.float32).unsqueeze(1)

    batch_probs = []
    with torch.no_grad():
        for start in range(0, len(symbol_tensor), batch_size):
            batch = symbol_tensor[start : start + batch_size]
            logits = model(batch)
            batch_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(batch_probs, axis=0)
    return probs.reshape(5, expressions.shape[0], 13).transpose(1, 0, 2)


def build_valid_assignments() -> np.ndarray:
    assignments = []
    for d1 in range(10):
        for op in [10, 11]:
            for d2 in range(10):
                d3 = d1 + d2 if op == 10 else d1 - d2
                if 0 <= d3 < 10:
                    assignments.append([d1, op, d2, 12, d3])
    return np.asarray(assignments, dtype=np.int64)


def score_assignments(probabilities: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    log_probs = np.log(np.clip(probabilities, EPSILON, 1.0))
    scores = np.zeros((probabilities.shape[0], len(assignments)), dtype=np.float64)
    for slot in range(5):
        scores += log_probs[:, slot, assignments[:, slot]]
    return scores


def exhaustive_decode(probabilities: np.ndarray, labels: np.ndarray) -> tuple[list[dict], dict]:
    valid_assignments = build_valid_assignments()
    scores = score_assignments(probabilities, valid_assignments)
    best_indices = scores.argmax(axis=1)
    predictions = valid_assignments[best_indices]

    assignment_to_index = {
        tuple(assignment.tolist()): index
        for index, assignment in enumerate(valid_assignments)
    }

    records = []
    ranks = []
    present_flags = []
    for index, true_label in enumerate(labels):
        true = tuple(int(value) for value in true_label)
        true_assignment_index = assignment_to_index.get(true)
        if true_assignment_index is None:
            true_rank = None
            correct_in_final_beam = False
        else:
            true_score = scores[index, true_assignment_index]
            true_rank = int(1 + np.sum(scores[index] > true_score))
            correct_in_final_beam = True
            ranks.append(true_rank)

        present_flags.append(correct_in_final_beam)
        prediction = predictions[index].astype(int).tolist()
        true_list = [int(value) for value in true_label]
        records.append(
            {
                "index": int(index),
                "prediction": prediction,
                "score": float(scores[index, best_indices[index]]),
                "equation_correct": prediction == true_list,
                "symbol_accuracy": _symbol_accuracy(prediction, true_list),
                "correct_in_final_beam": bool(correct_in_final_beam),
                "oracle_hit": bool(correct_in_final_beam),
                "correct_rank_in_final_beam": true_rank,
                "candidates_evaluated": int(len(valid_assignments)),
                "expanded_candidates": 0,
                "validity_checked_candidates": int(len(valid_assignments)),
                "final_valid_candidates": int(len(valid_assignments)),
                "missing_prediction": False,
            }
        )

    metrics = _summarize_records(records, ranks, present_flags)
    return records, metrics


def evaluate_beam_search(
    model: SymbolClassifier,
    dataset,
    beam_widths: Sequence[int] = BEAM_WIDTHS,
    noise_levels: Sequence[float] = GAUSSIAN_SIGMAS,
    seed: int = SEED,
    json_path: str | Path = JSON_RESULTS_PATH,
    csv_path: str | Path = CSV_RESULTS_PATH,
    plot_path: str | Path = PLOT_PATH,
) -> dict:
    images, labels = _unpack_dataset(dataset)
    set_seed(seed)

    table_rows = []
    per_condition = {}
    sample_traces = []

    for sigma in noise_levels:
        print(f"Evaluating Gaussian noise sigma={sigma}")
        if float(sigma) == 0.0:
            noisy_images = images.copy()
        else:
            noisy_images = np.clip(
                images + np.random.normal(0.0, float(sigma), images.shape),
                0.0,
                1.0,
            )

        probabilities = expression_symbol_probs(model, noisy_images)
        sigma_key = str(float(sigma))
        per_condition[sigma_key] = {}

        for beam_width in beam_widths:
            records, traces = _decode_with_beam_width(
                probabilities,
                labels,
                int(beam_width),
                collect_trace_count=TRACE_SAMPLE_SIZE,
                collect_traces=int(beam_width) in TRACE_SAMPLE_WIDTHS,
            )
            metrics = _metrics_from_records(records)
            row = _make_table_row(
                sigma=float(sigma),
                beam_width_label=str(int(beam_width)),
                decoder="beam",
                metrics=metrics,
            )
            table_rows.append(row)
            per_condition[sigma_key][str(int(beam_width))] = {
                "metrics": metrics,
                "per_expression": records,
            }
            sample_traces.extend(
                {
                    "noise_sigma": float(sigma),
                    "beam_width": int(beam_width),
                    **trace_record,
                }
                for trace_record in traces
            )

        exhaustive_records, exhaustive_metrics = exhaustive_decode(probabilities, labels)
        exhaustive_row = _make_table_row(
            sigma=float(sigma),
            beam_width_label="infinity",
            decoder="exhaustive",
            metrics=exhaustive_metrics,
        )
        table_rows.append(exhaustive_row)
        per_condition[sigma_key]["infinity"] = {
            "metrics": exhaustive_metrics,
            "per_expression": exhaustive_records,
        }

    results = {
        "config": {
            "seed": seed,
            "num_expressions": int(len(labels)),
            "noise_levels": [float(value) for value in noise_levels],
            "beam_widths": [int(value) for value in beam_widths],
            "exhaustive_valid_assignments": int(len(build_valid_assignments())),
            "model_path": str(MODEL_PATH),
            "dataset_path": str(DATASET_PATH),
        },
        "metric_definitions": {
            "equation_accuracy": "Fraction of five-symbol equations predicted exactly.",
            "symbol_accuracy": "Mean per-symbol accuracy across all positions.",
            "oracle_accuracy": (
                "Fraction of examples where the true equation appears in the final valid beam; "
                "this measures the accuracy an oracle reranker could achieve from the retained "
                "beam candidates."
            ),
            "prefix_oracle_accuracy_by_position": (
                "For beam rows, the fraction of examples where the true prefix remained in the "
                "kept beam after each position expansion, with position 5 corresponding to the "
                "final validity-filtered beam."
            ),
            "candidates_evaluated_per_expression": (
                "Beam search counts expanded partial candidates plus final validity checks; "
                "exhaustive counts the 110 valid equations scored."
            ),
            "correct_in_final_beam_pct": (
                "Percentage of examples where the true equation appears in the final retained "
                "valid beam. Exhaustive uses all valid equations as the final set."
            ),
            "mean_correct_rank_in_final_beam": (
                "Mean one-indexed score rank of the true equation among final valid candidates, "
                "computed only when the true equation is present."
            ),
        },
        "table": table_rows,
        "conditions": per_condition,
        "beam_width_comparisons": _compare_beam_width_choices(per_condition),
        "sample_traces": sample_traces,
    }

    _write_json(results, json_path)
    _write_csv(table_rows, csv_path)
    _plot_accuracy(table_rows, plot_path)
    _print_table(table_rows)

    print(f"\nSaved JSON results to {json_path}")
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved accuracy plot to {plot_path}")
    return results


def _decode_with_beam_width(
    probabilities: np.ndarray,
    labels: np.ndarray,
    beam_width: int,
    collect_trace_count: int = 0,
    collect_traces: bool = False,
) -> tuple[list[dict], list[dict]]:
    decoder = BeamSearchDecoder(beam_width=beam_width, recover_final_if_empty=False)
    records = []
    traces = []

    for index, (probability_rows, true_label) in enumerate(zip(probabilities, labels)):
        prediction, trace = decoder.decode(probability_rows)
        true = [int(value) for value in true_label]
        final_candidates = _final_valid_beam(trace)
        rank = _rank_in_candidates(true, final_candidates)
        prediction_list = _as_int_list(prediction)
        efficiency = _trace_efficiency(trace)
        prefix_survival = _prefix_survival(trace, true)

        records.append(
            {
                "index": int(index),
                "prediction": prediction_list,
                "score": _score_assignment(probability_rows, prediction_list),
                "equation_correct": prediction_list == true,
                "symbol_accuracy": _symbol_accuracy(prediction_list, true),
                "correct_in_final_beam": rank is not None,
                "oracle_hit": rank is not None,
                "correct_rank_in_final_beam": rank,
                "prefix_survival": prefix_survival,
                "candidates_evaluated": int(efficiency["total_candidates_evaluated"]),
                "expanded_candidates": int(efficiency["expanded_candidates"]),
                "validity_checked_candidates": int(efficiency["validity_checked_candidates"]),
                "final_valid_candidates": int(len(final_candidates)),
                "missing_prediction": prediction_list is None,
            }
        )

        if collect_traces and index < collect_trace_count:
            traces.append(
                {
                    "index": int(index),
                    "true": true,
                    "prediction": prediction_list,
                    "correct_rank_in_final_beam": rank,
                    "prefix_survival": prefix_survival,
                    "trace": trace,
                }
            )

    return records, traces


def _metrics_from_records(records: Sequence[dict]) -> dict:
    ranks = [
        int(record["correct_rank_in_final_beam"])
        for record in records
        if record["correct_rank_in_final_beam"] is not None
    ]
    present_flags = [bool(record["correct_in_final_beam"]) for record in records]
    return _summarize_records(records, ranks, present_flags)


def _summarize_records(
    records: Sequence[dict],
    ranks: Sequence[int],
    present_flags: Sequence[bool],
) -> dict:
    if not records:
        return {
            "equation_accuracy": 0.0,
            "symbol_accuracy": 0.0,
            "oracle_accuracy": 0.0,
            "oracle_accuracy_pct": 0.0,
            "candidates_evaluated_per_expression": 0.0,
            "correct_in_final_beam_pct": 0.0,
            "mean_correct_rank_in_final_beam": None,
            "median_correct_rank_in_final_beam": None,
            "ranked_examples": 0,
            "missing_prediction_rate": 0.0,
            "search_efficiency": {},
            "prefix_oracle_accuracy_by_position": {},
        }

    return {
        "equation_accuracy": float(np.mean([record["equation_correct"] for record in records])),
        "symbol_accuracy": float(np.mean([record["symbol_accuracy"] for record in records])),
        "oracle_accuracy": float(np.mean(present_flags)),
        "oracle_accuracy_pct": float(100.0 * np.mean(present_flags)),
        "candidates_evaluated_per_expression": float(
            np.mean([record["candidates_evaluated"] for record in records])
        ),
        "correct_in_final_beam_pct": float(100.0 * np.mean(present_flags)),
        "mean_correct_rank_in_final_beam": float(np.mean(ranks)) if ranks else None,
        "median_correct_rank_in_final_beam": float(np.median(ranks)) if ranks else None,
        "ranked_examples": int(len(ranks)),
        "missing_prediction_rate": float(
            np.mean([record.get("missing_prediction", False) for record in records])
        ),
        "search_efficiency": {
            "expanded_candidates_per_expression": float(
                np.mean([record.get("expanded_candidates", 0) for record in records])
            ),
            "validity_checked_candidates_per_expression": float(
                np.mean([record.get("validity_checked_candidates", 0) for record in records])
            ),
            "candidates_evaluated_per_expression": float(
                np.mean([record["candidates_evaluated"] for record in records])
            ),
            "final_valid_candidates_per_expression": float(
                np.mean([record.get("final_valid_candidates", 0) for record in records])
            ),
        },
        "prefix_oracle_accuracy_by_position": _prefix_oracle_summary(records),
    }


def _make_table_row(sigma: float, beam_width_label: str, decoder: str, metrics: dict) -> dict:
    row = {
        "noise_sigma": sigma,
        "beam_width": beam_width_label,
        "decoder": decoder,
    }
    row.update(metrics)
    return row


def _unpack_dataset(dataset) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(dataset, (str, Path)):
        return load_expression_dataset(dataset)

    if isinstance(dataset, tuple) and len(dataset) >= 2:
        images, labels = dataset[0], dataset[1]
        return _normalize_images(images), np.asarray(labels, dtype=np.int64)

    if isinstance(dataset, Mapping) or hasattr(dataset, "__getitem__"):
        images = dataset["images"]
        labels = dataset["labels"]
        return _normalize_images(images), np.asarray(labels, dtype=np.int64)

    raise TypeError("dataset must be a path, mapping with images/labels, or (images, labels)")


def _normalize_images(images) -> np.ndarray:
    images = np.asarray(images, dtype=np.float32)
    if images.size and images.max() > 1.0:
        images = images / 255.0
    return images


def _score_assignment(probabilities: np.ndarray, assignment: Sequence[int] | None) -> float | None:
    if assignment is None:
        return None
    log_probs = np.log(np.clip(probabilities, EPSILON, 1.0))
    score = 0.0
    for position, label in enumerate(assignment):
        score += float(log_probs[position, int(label)])
    return score


def _as_int_list(assignment: Sequence[int] | None) -> list[int] | None:
    if assignment is None:
        return None
    return [int(value) for value in assignment]


def _symbol_accuracy(prediction: Sequence[int] | None, true: Sequence[int]) -> float:
    if prediction is None:
        return 0.0
    return float(np.mean([int(pred) == int(label) for pred, label in zip(prediction, true)]))


def _final_valid_beam(trace: Sequence[dict]) -> list[dict]:
    if not trace:
        return []
    final_step = trace[-1]
    if final_step.get("action") != "validity_filter":
        return []
    return list(final_step.get("kept", []))


def _rank_in_candidates(assignment: Sequence[int], candidates: Sequence[dict]) -> int | None:
    assignment = [int(value) for value in assignment]
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.get("assignment") == assignment:
            return int(rank)
    return None


def _prefix_survival(trace: Sequence[dict], true: Sequence[int]) -> dict:
    true = [int(value) for value in true]
    survival = {}
    for step in trace:
        position = int(step["position"])
        if step.get("action") == "expand":
            prefix = true[: position + 1]
            survival[str(position)] = any(
                candidate.get("assignment") == prefix
                for candidate in step.get("kept", [])
            )
        elif step.get("action") == "validity_filter":
            survival[str(position)] = any(
                candidate.get("assignment") == true
                for candidate in step.get("kept", [])
            )
    return survival


def _prefix_oracle_summary(records: Sequence[dict]) -> dict:
    positions = sorted(
        {
            position
            for record in records
            for position in record.get("prefix_survival", {})
        },
        key=lambda value: int(value),
    )
    return {
        position: float(
            np.mean(
                [
                    bool(record.get("prefix_survival", {}).get(position, False))
                    for record in records
                ]
            )
        )
        for position in positions
    }


def _trace_efficiency(trace: Sequence[dict]) -> dict:
    expanded = sum(
        int(step.get("expanded_count", 0))
        for step in trace
        if step.get("action") == "expand"
    )
    validity_checked = sum(
        int(step.get("expanded_count", 0))
        for step in trace
        if step.get("action") == "validity_filter"
    )
    return {
        "expanded_candidates": expanded,
        "validity_checked_candidates": validity_checked,
        "total_candidates_evaluated": expanded + validity_checked,
    }


def _compare_beam_width_choices(per_condition: dict) -> dict:
    comparisons = {}
    for sigma_key, condition in per_condition.items():
        width_keys = sorted(
            condition.keys(),
            key=lambda value: float("inf") if value == "infinity" else int(value),
        )
        reference_key = "infinity" if "infinity" in condition else width_keys[-1]
        reference_records = condition[reference_key]["per_expression"]
        num_examples = len(reference_records)

        summary_by_width = {}
        for width_key in width_keys:
            records = condition[width_key]["per_expression"]
            matches = [
                records[index]["prediction"] == reference_records[index]["prediction"]
                for index in range(num_examples)
            ]
            summary_by_width[width_key] = {
                "match_reference_rate": float(np.mean(matches)) if matches else 0.0,
                "num_differences_from_reference": int(len(matches) - sum(matches)),
            }

        per_expression = []
        for index in range(num_examples):
            predictions = {
                width_key: condition[width_key]["per_expression"][index]["prediction"]
                for width_key in width_keys
            }
            scores = {
                width_key: condition[width_key]["per_expression"][index]["score"]
                for width_key in width_keys
            }
            unique_predictions = []
            for prediction in predictions.values():
                if prediction not in unique_predictions:
                    unique_predictions.append(prediction)
            per_expression.append(
                {
                    "index": int(index),
                    "prediction_by_beam_width": predictions,
                    "score_by_beam_width": scores,
                    "num_unique_predictions": int(len(unique_predictions)),
                    "all_widths_match_reference": all(
                        prediction == predictions[reference_key]
                        for prediction in predictions.values()
                    ),
                }
            )

        comparisons[sigma_key] = {
            "reference_width": reference_key,
            "summary_by_width": summary_by_width,
            "per_expression": per_expression,
        }
    return comparisons


def _write_json(results: dict, path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def _write_csv(table_rows: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    fieldnames = [
        "noise_sigma",
        "beam_width",
        "decoder",
        "equation_accuracy",
        "symbol_accuracy",
        "oracle_accuracy",
        "oracle_accuracy_pct",
        "candidates_evaluated_per_expression",
        "correct_in_final_beam_pct",
        "mean_correct_rank_in_final_beam",
        "median_correct_rank_in_final_beam",
        "ranked_examples",
        "missing_prediction_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _csv_value(value):
    if isinstance(value, float):
        return round(value, 6)
    if value is None:
        return ""
    return value


def _plot_accuracy(table_rows: Sequence[dict], path: str | Path) -> None:
    numeric_labels = sorted(
        {
            row["beam_width"]
            for row in table_rows
            if row["beam_width"] != "infinity"
        },
        key=lambda value: int(value),
    )
    labels = numeric_labels + (["infinity"] if any(row["beam_width"] == "infinity" for row in table_rows) else [])
    display_labels = ["inf" if label == "infinity" else label for label in labels]
    x_positions = np.arange(len(labels))
    sigmas = sorted({float(row["noise_sigma"]) for row in table_rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in sigmas:
        y_values = []
        for label in labels:
            row = next(
                item
                for item in table_rows
                if item["noise_sigma"] == float(sigma) and item["beam_width"] == label
            )
            y_values.append(row["equation_accuracy"])
        ax.plot(x_positions, y_values, marker="o", label=f"sigma={sigma}")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels)
    ax.set_xlabel("Beam width")
    ax.set_ylabel("Equation accuracy")
    ax.set_title("Beam Search Accuracy Approaches Exhaustive Decoding")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _print_table(table_rows: Sequence[dict]) -> None:
    print("\nComprehensive beam vs exhaustive results:")
    header = (
        "sigma | beam_width | equation_acc | symbol_acc | candidates/expression | "
        "oracle_acc | mean_rank"
    )
    print(header)
    print("-" * len(header))
    for row in table_rows:
        rank = row["mean_correct_rank_in_final_beam"]
        rank_text = "NA" if rank is None else f"{rank:.2f}"
        print(
            f"{row['noise_sigma']:<5.1f} | "
            f"{row['beam_width']:<10} | "
            f"{row['equation_accuracy']:.4f}       | "
            f"{row['symbol_accuracy']:.4f}     | "
            f"{row['candidates_evaluated_per_expression']:<21.1f} | "
            f"{row['oracle_accuracy']:<10.4f} | "
            f"{rank_text}"
        )


def main() -> None:
    model = load_model()
    dataset = load_expression_dataset()
    evaluate_beam_search(model, dataset)


if __name__ == "__main__":
    main()
