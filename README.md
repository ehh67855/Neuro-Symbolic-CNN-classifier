# Neuro-Symbolic CNN Classifier

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


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
