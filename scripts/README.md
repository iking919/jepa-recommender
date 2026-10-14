# Scripts

This directory contains command-line entry points for running project workflows that use the reusable modules under `src/`.

| Script | Purpose |
|---|---|
| `download_ml1m.py` | Download GroupLens Dataset |
| `preprocess_ml1m.py` | Preprocess Dataset |
| `run_bert4rec_evaluation.py` | Evaluate a BERT4Rec checkpoint |
| `run_jepa_evaluation.py` | Evaluate a JEPA checkpoint |
| `run_bert4rec_hyperparameter_search.py` | Run BERT4Rec hyperparameter search |
| `run_jepa_hyperparameter_search.py` | Run JEPA Causal hyperparameter search |


## Dataset Preparation

Run the preprocessing entry points available in the repository to generate the intermediate and processed datasets. If the repository includes `scripts/download_ml1m.py` and `scripts/preprocess_ml1m.py`, the commands are:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

## BERT4Rec Evaluation

Baseline:

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_baseline.yaml \
    --checkpoint checkpoints/bert4rec_baseline/bert4rec_baseline_best.pt
```

Tuned:

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_tuned.yaml \
    --checkpoint checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
```

## JEPA Evaluation

Baseline:

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_baseline.yaml \
    --checkpoint checkpoints/jepa_baseline/jepa_baseline_best.pt
```

Tuned:

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_tuned.yaml \
    --checkpoint checkpoints/jepa_tuned/jepa_tuned_best.pt
```

## Hyperparameter Search

BERT4Rec:

```bash
python scripts/run_bert4rec_hyperparameter_search.py \
    --config configs/bert4rec_baseline.yaml
```

JEPA:

```bash
python scripts/run_jepa_hyperparameter_search.py \
    --config configs/jepa_baseline.yaml
```

## General Execution

Run scripts using Python's module execution syntax from the repository root:

```bash
python -m scripts.<module_name> --help
```

This supports consistent package imports when scripts depend on modules under `src/`.