# Neuro-Symbolic CNN Classifier

This project evaluates a small neuro-symbolic equation recognizer. A CNN predicts
one symbol at a time from fixed-width image crops. A symbolic decoder then chooses
a complete equation that has high neural probability and satisfies the arithmetic
constraint.

The experiment is intentionally small enough to compare beam search against an
exact exhaustive reference. That makes it possible to separate search errors from
ranking errors.

## Repository layout

- `code/`: Python source files for data generation, CNN training, decoding,
  evaluation, analysis, plotting, and validation.
- `paper.tex`: LaTeX source for the paper.
- `paper.pdf`: Compiled paper.
- `requirements.txt`: Python package versions used for the final run.
- `synthetic_expressions.npz`: Fixed held-out equation dataset.
- `symbol_classifier.pth`: Saved CNN checkpoint.
- `*.csv`, `*.png`, and `results_summary.md`: Reproducible result tables and
  paper figures.

The full per-expression JSON file is not tracked because it is larger than
GitHub's normal file limit. The summary CSV files and figures are tracked, and
the JSON can be regenerated with `python code/evaluation.py`.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To compile the paper, install a LaTeX distribution that provides `latexmk`.

## Common commands

Run the complete pipeline from the saved dataset and checkpoint:

```bash
python code/reproduce.py
```

Regenerate the dataset and retrain the classifier first:

```bash
python code/reproduce.py --from-scratch
```

Run the validator only:

```bash
python code/validate_results.py
```

Run a small demonstration on one noisy expression:

```bash
python code/demo.py
```

Compile the paper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

## Code guide

- `beam_decoder.py`: Task-independent beam search decoder and arithmetic
  validity check.
- `dataset.py`: Builds the fixed held-out equation dataset from MNIST test
  digits and rendered operators.
- `classifier.py`: Trains the 13-class symbol CNN.
- `evaluation.py`: Applies Gaussian noise, runs beam widths and the exhaustive
  reference, and writes the main result files.
- `analyze_search.py`: Computes pruning, oracle accuracy, final-beam, and rank
  analyses.
- `benchmark_efficiency.py`: Measures symbolic decoding cost.
- `qualitative_examples.py`: Produces clean/noisy example panels.
- `search_space.py`: Produces the search-space reduction figure.
- `results_beam.py`, `efficiency.py`, `recoverability.py`: Produce the paper
  result figures.
- `summarize_results.py`: Writes compact paper tables and `results_summary.md`.
- `validate_results.py`: Checks result consistency, figures, paper labels, and
  LaTeX log health.
- `project_paths.py`: Central project-root path helper used by the scripts.

Compatibility wrappers are kept for older script names:

- `pipeline_diagram_beam.py`
- `search_space_beam.py`
- `error_analysis_beam.py`

## Key settings

- Random seed: `0`
- Held-out equations: `5,000`
- Layout: `d1 op d2 = d3`
- Symbol classes: digits `0-9`, plus, minus, equals
- Gaussian noise levels: `0.0`, `0.2`, `0.4`, `0.6`
- Beam widths: `1`, `3`, `5`, `10`, `20`, `50`, `110`
- Exhaustive reference: all 110 valid equations

## Main reported numbers

- At `sigma=0.4`, `B=10` reaches `92.38%` equation accuracy.
- At `sigma=0.4`, `B=10` keeps the correct equation in the final beam for
  `96.18%` of examples.
- At `sigma=0.4`, `B=50` reaches `93.80%`, matching exhaustive decoding.
- `B=10` uses `240` score operations per expression versus `550` for exhaustive
  decoding.

## Validation scope

`code/validate_results.py` checks the result tables, expected noise levels and
beam widths, monotonicity properties, recoverability trends, required figures,
paper labels, and LaTeX log health. If the full per-expression JSON is present,
it also checks prediction-level agreement between `B=110` and exhaustive
decoding.
