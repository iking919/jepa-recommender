# Scripts

Command-line entry points for the project workflows. They are thin wrappers
over the reusable code in [`src/`](../src/README.md).

Run every script **from the repository root, as a module**:

```bash
python -m scripts.<name> --help
```

Running a file directly (`python scripts/<name>.py`) does not put the
repository on the import path.

| Script | Purpose |
|---|---|
| `download_ml1m.py` | Download MovieLens-1M from GroupLens |
| `download_amazon_beauty.py` | Download the Amazon Beauty 5-core data from SNAP |
| `preprocess_ml1m.py` | Build model-ready MovieLens-1M datasets |
| `preprocess_amazon_beauty.py` | Build model-ready Amazon Beauty datasets |
| `run_bert4rec_evaluation.py` | Evaluate a BERT4Rec checkpoint on the test set |
| `run_jepa_evaluation.py` | Evaluate a JEPA checkpoint on the test set |
| `run_bert4rec_hyperparameter_search.py` | BERT4Rec grid search |
| `run_jepa_hyperparameter_search.py` | Causal JEPA grid search |
| `run_cross_seed_comparison.py` | Train one configuration over several seeds |
| `summarize_runs.py` | Mean and standard deviation over groups of finished runs |

## Dataset preparation

```bash
python -m scripts.download_ml1m
python -m scripts.preprocess_ml1m

python -m scripts.download_amazon_beauty
python -m scripts.preprocess_amazon_beauty
```

Both preprocessing scripts only differ in how they read and order the raw
data; the remaining steps are shared (`src/data/preprocessing.py`). The
downloaders accept `--raw-dir` and `--force`.

## Evaluation

The evaluation scripts take `--config`, plus optional `--dataset`, `--set` and
`--name`. `--checkpoint` defaults to
`checkpoints/<dataset>/<experiment>/<experiment>_best.pt`.

```bash
python -m scripts.run_bert4rec_evaluation --config configs/ml-1m/bert4rec_tuned.yaml
python -m scripts.run_jepa_evaluation --config configs/ml-1m/jepa_tuned.yaml
python -m scripts.run_jepa_evaluation --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty
```

## Hyperparameter search

```bash
python -m scripts.run_bert4rec_hyperparameter_search \
    --config configs/ml-1m/bert4rec_tuned.yaml --dataset amazon_beauty

python -m scripts.run_jepa_hyperparameter_search \
    --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty
```

Reports go to `results/<dataset>/<model>_grid_search/` unless `--output-dir`
is given.

## Cross-seed comparison and summaries

```bash
python -m scripts.run_cross_seed_comparison \
    --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty

python -m scripts.summarize_runs --runs "results/ml-1m/causal_const_s*"
```

Cross-seed reports go to `results/<dataset>/<model>_seed_search/`.
`summarize_runs` takes one glob pattern per group of runs.
