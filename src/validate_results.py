"""Validate beam-search artifacts used by the paper.

This script checks reproducibility-critical invariants across the generated
JSON/CSV results, figures, and LaTeX source. It is intentionally lightweight:
it validates existing artifacts rather than rerunning the full experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from project_paths import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR


ROOT = PROJECT_ROOT
EXPECTED_SEED = 0
EXPECTED_NUM_EXPRESSIONS = 5000
EXPECTED_NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6]
EXPECTED_BEAM_WIDTHS = [1, 3, 5, 10, 20, 50, 110]
PAPER_FIGURES = [
    FIGURES_DIR / "qualitative_decoding_examples.png",
    FIGURES_DIR / "search_space_reduction.png",
    FIGURES_DIR / "beam_width_accuracy.png",
    FIGURES_DIR / "efficiency_tradeoff.png",
    FIGURES_DIR / "recoverability_analysis.png",
]
MIN_FIGURE_PIXELS = 1200


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_close(actual: float, expected: float, message: str, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        fail(f"{message}: actual={actual}, expected={expected}")


def validate_config(results: dict) -> None:
    config = results.get("config", {})
    if config.get("seed") != EXPECTED_SEED:
        fail(f"unexpected seed: {config.get('seed')}")
    if config.get("num_expressions") != EXPECTED_NUM_EXPRESSIONS:
        fail(f"unexpected num_expressions: {config.get('num_expressions')}")
    if [float(x) for x in config.get("noise_levels", [])] != EXPECTED_NOISE_LEVELS:
        fail(f"unexpected noise levels: {config.get('noise_levels')}")
    if [int(x) for x in config.get("beam_widths", [])] != EXPECTED_BEAM_WIDTHS:
        fail(f"unexpected beam widths: {config.get('beam_widths')}")


def validate_table_config(rows: list[dict]) -> None:
    noise_levels = sorted({float(row["noise_sigma"]) for row in rows})
    if noise_levels != EXPECTED_NOISE_LEVELS:
        fail(f"unexpected noise levels: {noise_levels}")

    finite_widths = sorted({
        int(row["beam_width"])
        for row in rows
        if row["beam_width"] != "infinity"
    })
    if finite_widths != EXPECTED_BEAM_WIDTHS:
        fail(f"unexpected beam widths: {finite_widths}")


def validate_dataset_size() -> None:
    data_path = ROOT / "synthetic_expressions.npz"
    if not data_path.exists():
        return
    with np.load(data_path) as data:
        count = int(data["images"].shape[0])
    if count != EXPECTED_NUM_EXPRESSIONS:
        fail(f"unexpected num_expressions: {count}")


def validate_accuracy_table(rows: list[dict]) -> None:
    for noise in EXPECTED_NOISE_LEVELS:
        condition = [
            row
            for row in rows
            if float(row["noise_sigma"]) == noise
        ]
        finite_rows = sorted(
            [row for row in condition if row["beam_width"] != "infinity"],
            key=lambda row: int(row["beam_width"]),
        )
        exhaustive = next(row for row in condition if row["beam_width"] == "infinity")
        previous_accuracy = -1.0
        for row in finite_rows:
            accuracy = float(row["equation_accuracy"])
            if accuracy + 1e-12 < previous_accuracy:
                fail(f"accuracy is not monotonic for sigma={noise}")
            previous_accuracy = accuracy
            oracle = float(row["oracle_accuracy"])
            if oracle + 1e-12 < accuracy:
                fail(f"oracle below actual for sigma={noise}, B={row['beam_width']}")

        b110 = next(row for row in finite_rows if row["beam_width"] == "110")
        assert_close(
            float(b110["equation_accuracy"]),
            float(exhaustive["equation_accuracy"]),
            f"B=110 accuracy does not match exhaustive for sigma={noise}",
        )


def validate_prediction_matches(results: dict) -> list[str]:
    warnings = []
    comparisons = results.get("beam_width_comparisons", {})
    for noise, condition in sorted(comparisons.items(), key=lambda item: float(item[0])):
        width_summary = condition.get("summary_by_width", {})
        if "110" not in width_summary:
            continue
        differences = int(width_summary["110"]["num_differences_from_reference"])
        if differences:
            warnings.append(
                f"B=110 has {differences} prediction-level differences from exhaustive "
                f"at sigma={float(noise):.1f}; equation accuracy still matches."
            )
    return warnings


def validate_recoverability(recoverability_rows: list[dict]) -> None:
    previous_in_beam = -1.0
    previous_pruned = 101.0
    for row in recoverability_rows:
        in_beam = float(row["correct_in_final_beam_pct"])
        pruned = float(row["pruned_before_final_pct"])
        accuracy = float(row["equation_accuracy_pct"])
        if in_beam + 1e-12 < accuracy:
            fail(f"recoverability below accuracy for B={row['beam_width']}")
        if in_beam + 1e-12 < previous_in_beam:
            fail("correct-in-final-beam is not monotonic")
        if pruned - 1e-12 > previous_pruned:
            fail("pruned-before-final is not decreasing")
        previous_in_beam = in_beam
        previous_pruned = pruned


def validate_dataset_split() -> list[str]:
    """Check whether the first expression digit is drawn from MNIST test."""
    warnings = []
    try:
        import torchvision
        import torchvision.transforms as transforms
    except Exception as exc:  # pragma: no cover - optional environment check
        return [f"could not import torchvision for dataset split check: {exc}"]

    data = np.load(ROOT / "synthetic_expressions.npz")
    first_digit = data["images"][0, :, :28]
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(
        root=str(ROOT / "data"),
        train=True,
        download=False,
        transform=transform,
    )
    test = torchvision.datasets.MNIST(
        root=str(ROOT / "data"),
        train=False,
        download=False,
        transform=transform,
    )
    train_matches = int(((train.data.numpy() == first_digit).all(axis=(1, 2))).sum())
    test_matches = int(((test.data.numpy() == first_digit).all(axis=(1, 2))).sum())
    if test_matches == 0:
        fail("synthetic_expressions.npz does not appear to use MNIST test digits")
    if train_matches:
        warnings.append("first expression digit also matches MNIST train; duplicate digit image")
    return warnings


def validate_figures() -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency issue
        fail(f"could not import PIL for figure validation: {exc}")

    for path in PAPER_FIGURES:
        if not path.exists():
            fail(f"missing figure: {path.relative_to(ROOT)}")
        with Image.open(path) as image:
            width, height = image.size
        if width < MIN_FIGURE_PIXELS or height < MIN_FIGURE_PIXELS:
            fail(f"figure resolution too small: {path.relative_to(ROOT)} is {width}x{height}")


def validate_paper_text() -> None:
    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    banned_patterns = [
        "OLD:",
        "Exact MAP decoding by enumerating 110 valid equations",
        "9x faster",
        "9-10x faster",
        "exhaustive enumeration",
    ]
    for pattern in banned_patterns:
        if pattern in paper:
            fail(f"old or unsupported paper framing remains: {pattern}")

    for figure_label in [
        "fig:qualitative",
        "fig:searchspace",
        "fig:beamwidth",
        "fig:efficiency",
        "fig:recoverability",
    ]:
        if figure_label not in paper:
            fail(f"paper missing figure label {figure_label}")

    for table_label in ["tab:results", "tab:efficiency", "tab:recoverability"]:
        if table_label not in paper:
            fail(f"paper missing table label {table_label}")


def validate_latex_log() -> list[str]:
    log_path = ROOT / "paper.log"
    if not log_path.exists():
        return ["paper.log is absent; skipped LaTeX log checks"]
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    bad_patterns = [
        "undefined references",
        "Citation",
        "Reference",
        "Overfull",
        "Underfull",
        "! LaTeX Error",
    ]
    for pattern in bad_patterns:
        if re.search(pattern, log, flags=re.IGNORECASE):
            fail(f"LaTeX log contains: {pattern}")
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated beam-search artifacts.")
    parser.add_argument(
        "--skip-dataset-split",
        action="store_true",
        help="Skip the MNIST test-vs-train split check.",
    )
    parser.add_argument(
        "--skip-latex-log",
        action="store_true",
        help="Skip paper.log checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = RESULTS_DIR / "beam_exhaustive_noise_results.json"
    results = load_json(results_path) if results_path.exists() else None
    table_rows = read_csv(RESULTS_DIR / "beam_exhaustive_noise_results.csv")
    recoverability_rows = read_csv(RESULTS_DIR / "recoverability_table_sigma_0.4.csv")

    if results is not None:
        validate_config(results)
    else:
        validate_table_config(table_rows)
    validate_dataset_size()
    validate_accuracy_table(table_rows)
    validate_recoverability(recoverability_rows)
    validate_figures()
    validate_paper_text()
    if results is not None:
        warnings = validate_prediction_matches(results)
    else:
        warnings = ["full per-expression JSON is absent; skipped prediction-level match checks"]
    if not args.skip_latex_log:
        warnings.extend(validate_latex_log())
    if not args.skip_dataset_split:
        warnings.extend(validate_dataset_split())

    print("Validation passed.")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
