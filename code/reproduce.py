"""Run the reproducible beam-search experiment pipeline.

Default behavior regenerates all experiment results, analysis tables, figures,
paper summary artifacts, and the PDF from the existing fixed dataset and saved
CNN checkpoint. Use ``--from-scratch`` to regenerate the dataset and retrain the
classifier first.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from project_paths import CODE_DIR, PROJECT_ROOT


ROOT = PROJECT_ROOT
PYTHON = sys.executable


def script_path(name: str) -> str:
    return str(CODE_DIR / name)


def run_step(command: list[str], description: str) -> None:
    """Run one pipeline command and fail immediately on errors."""
    print(f"\n==> {description}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce beam-search results and paper assets.")
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Regenerate synthetic_expressions.npz and retrain symbol_classifier.pth before evaluation.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation.py and reuse existing beam_exhaustive_noise_results.* artifacts.",
    )
    parser.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="Skip benchmark_efficiency.py and reuse existing beam_efficiency_results.* artifacts.",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="Skip latexmk PDF generation.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validate_results.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_scratch:
        run_step([PYTHON, script_path("dataset.py")], "Generate fixed-seed held-out expression dataset")
        run_step([PYTHON, script_path("classifier.py")], "Train fixed-seed symbol classifier")

    if not args.skip_evaluation:
        run_step([PYTHON, script_path("evaluation.py")], "Run Gaussian-noise beam-width sweep")

    run_step(
        [PYTHON, script_path("analyze_search.py"), str(ROOT / "beam_exhaustive_noise_results.json")],
        "Analyze pruning, oracle accuracy, and rank behavior",
    )

    if not args.skip_efficiency:
        run_step([PYTHON, script_path("benchmark_efficiency.py")], "Benchmark symbolic search efficiency")

    figure_steps = [
        ("qualitative_examples.py", "Generate qualitative dataset/noise/decoding examples"),
        ("search_space.py", "Generate search-space reduction diagram"),
        ("results_beam.py", "Generate accuracy-vs-beam-width plot"),
        ("efficiency.py", "Generate efficiency tradeoff plot"),
        ("recoverability.py", "Generate recoverability breakdown plot"),
    ]
    for script, description in figure_steps:
        run_step([PYTHON, script_path(script)], description)

    run_step([PYTHON, script_path("summarize_results.py")], "Generate CSV tables and results summary")

    if not args.skip_paper:
        if shutil.which("latexmk") is None:
            raise RuntimeError("latexmk is required to rebuild paper.pdf")
        run_step(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
            "Compile paper.pdf",
        )

    if not args.skip_validation:
        run_step([PYTHON, script_path("validate_results.py")], "Validate generated artifacts")

    print("\nReproduction pipeline complete.")


if __name__ == "__main__":
    main()
