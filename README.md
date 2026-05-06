# Neuro-Symbolic CNN Classifier

This repository contains a small neuro-symbolic equation recognizer. A CNN reads
fixed-width symbol crops from an equation image. A decoder then searches for a
complete equation that is both likely under the CNN and valid under the
arithmetic rule.

The experiment is deliberately small. Exhaustive decoding is still possible, so
beam search can be compared against an exact reference and the errors can be
split into pruning failures and ranking failures.

## Layout

- `src/`: Core project code.
- `scripts/figures/`: Scripts that only generate figures.
- `figures/`: Paper and analysis figures.
- `results/`: CSV tables, compact JSON summaries, and `results_summary.md`.
- `paper.tex` and `paper.pdf`: Paper source and compiled paper.
- `synthetic_expressions.npz`: Fixed held-out equation dataset.
- `symbol_classifier.pth`: Saved CNN checkpoint.

The full per-example evaluation JSON is not tracked because it is about 153 MB.
It can be regenerated with `python src/evaluation.py`. The smaller CSV and JSON
summaries used by the paper are tracked in `results/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install a LaTeX distribution with `latexmk` if you want to rebuild the PDF.

## Reproduce the Results

Run the full pipeline from the checked-in dataset and checkpoint:

```bash
python src/reproduce.py
```

Regenerate the dataset and retrain the classifier first:

```bash
python src/reproduce.py --from-scratch
```

Run validation only:

```bash
python src/validate_results.py
```

Run a one-example demo:

```bash
python src/demo.py
```

Compile the paper:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

## Core Code

- `src/beam_decoder.py`: Beam search decoder and arithmetic validity check.
- `src/dataset.py`: Builds the fixed held-out dataset from MNIST test digits.
- `src/classifier.py`: Trains the 13-class CNN on MNIST digits and rendered
  operators.
- `src/evaluation.py`: Runs Gaussian-noise evaluation for all beam widths and
  the exhaustive reference.
- `src/analyze_search.py`: Computes pruning, oracle, rank, and final-beam
  analyses.
- `src/benchmark_efficiency.py`: Measures symbolic decoding cost.
- `src/summarize_results.py`: Writes paper-ready tables and `results_summary.md`.
- `src/validate_results.py`: Checks result consistency, figure availability,
  paper labels, and LaTeX log health.
- `src/reproduce.py`: Orchestrates the full pipeline.

## Figure Scripts

- `scripts/figures/qualitative_examples.py`
- `scripts/figures/search_space.py`
- `scripts/figures/results_beam.py`
- `scripts/figures/efficiency.py`
- `scripts/figures/recoverability.py`

These scripts read from `results/` and write to `figures/`.

## Key Settings

- Random seed: `0`
- Held-out equations: `5,000`
- Layout: `d1 op d2 = d3`
- Symbol classes: digits `0-9`, plus, minus, equals
- Gaussian noise levels: `0.0`, `0.2`, `0.4`, `0.6`
- Beam widths: `1`, `3`, `5`, `10`, `20`, `50`, `110`
- Exhaustive reference: all 110 valid equations

## Main Reported Numbers

- At `sigma=0.4`, `B=10` reaches `92.38%` equation accuracy.
- At `sigma=0.4`, `B=10` keeps the correct equation in the final beam for
  `96.18%` of examples.
- At `sigma=0.4`, `B=50` reaches `93.80%`, matching exhaustive decoding.
- `B=10` uses `240` score operations per expression versus `550` for exhaustive
  decoding.
