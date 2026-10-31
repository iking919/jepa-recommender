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
| `run_cross_seed_comparison.py` | Train one config over several seeds |
| `migrate_results_layout.py` | One-time move of old `results/` and `checkpoints/` into `<dataset>/` folders |


## Dataset Preparation

Run the preprocessing entry points available in the repository to generate the intermediate and processed datasets. If the repository includes `scripts/download_ml1m.py` and `scripts/preprocess_ml1m.py`, the commands are:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

## Evaluation

All entry points take `--config`, plus optional `--dataset`, `--set`, and `--name`. `--checkpoint` defaults to `checkpoints/<dataset>/<experiment>/<experiment>_best.pt`.

```bash
python -m scripts.run_bert4rec_evaluation --config configs/ml-1m/bert4rec_tuned.yaml
python -m scripts.run_jepa_evaluation --config configs/ml-1m/jepa_tuned.yaml
python -m scripts.run_jepa_evaluation --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty
```

## Hyperparameter Search

BERT4Rec:

```bash
python -m scripts.run_bert4rec_hyperparameter_search \
    --config configs/ml-1m/bert4rec_baseline.yaml --dataset amazon_beauty
```

JEPA:

```bash
python -m scripts.run_jepa_hyperparameter_search \
    --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty

# reports go to results/<dataset>/<model>_grid_search/ unless --output-dir is given
```

## General Execution

Run scripts using Python's module execution syntax from the repository root:

```bash
python -m scripts.<module_name> --help
```

This supports consistent package imports when scripts depend on modules under `src/`.