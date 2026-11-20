# Causal JEPA for Sequential Recommendation

A research project investigating **Joint-Embedding Predictive Architectures
(JEPA)** for sequential recommendation, with a focus on causal, one-pass
training and its comparison against **BERT4Rec**.

The project evaluates recommendation quality, training efficiency and
computational requirements on MovieLens-1M and Amazon Beauty.

## Contents

- [Overview](#overview)
- [Models](#models)
- [Datasets](#datasets)
- [Evaluation protocol](#evaluation-protocol)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Experiment configuration](#experiment-configuration)
- [Training](#training)
- [Hyperparameter search](#hyperparameter-search)
- [Cross-seed comparison](#cross-seed-comparison)
- [Test-set evaluation](#test-set-evaluation)
- [Results and experiment tracking](#results-and-experiment-tracking)
- [Tests](#tests)
- [Reproducibility](#reproducibility)
- [Citation and acknowledgements](#citation-and-acknowledgements)

## Overview

Sequential recommendation models learn from a user's interaction history to
predict the items they will interact with next. This project investigates
whether a causal JEPA can learn useful sequence representations by predicting
the *embeddings* of future items, rather than their identities directly.

The primary comparison is between:

- **BERT4Rec**, a bidirectional Transformer trained with masked item
  prediction.
- **Causal JEPA**, a causal Transformer that predicts future item
  representations from encoded interaction contexts.

A central question is whether causal, one-pass JEPA training can remove
redundant computation while keeping recommendation quality competitive.

The project aims to:

1. Implement a modular causal JEPA model for sequential recommendation.
2. Establish BERT4Rec as a comparative baseline.
3. Compare both models under identical splits and evaluation protocols.
4. Measure the computational efficiency of causal JEPA training.
5. Examine the effect of hyperparameter selection on quality and efficiency.
6. Compare training time, convergence, parameter counts and GPU memory.
7. Repeat the comparison across datasets and random seeds.

## Models

### BERT4Rec

The non-JEPA baseline. A bidirectional Transformer is trained to reconstruct
masked items from the surrounding sequence. At evaluation time the final
position is masked and the model ranks candidate items for it.

Implementation: [`src/models/bert4rec.py`](src/models/bert4rec.py).

### Causal JEPA

The primary research model. It consists of:

- **Context encoder:** encodes the interaction history with causal attention.
- **Target encoder:** produces the target representations that serve as the
  prediction objective.
- **Predictor:** predicts target representations from context
  representations.
- **EMA update:** the target encoder tracks an exponential moving average of
  the context encoder.

Causal attention guarantees that a position only sees earlier interactions.
This allows *one-pass* training: each user's history is cut into fixed-size
windows, and every position of a window yields a prediction in a single
forward pass, instead of re-encoding a separate prefix for every target.

Implementation: [`src/models/jepa.py`](src/models/jepa.py) and
[`src/data/sequence_windows.py`](src/data/sequence_windows.py). A standard
(non-causal) JEPA, which encodes one prefix per target, is kept as an
additional reference point.

## Datasets

### MovieLens-1M

Published by the GroupLens Research Project at the University of Minnesota.

| Statistic | Count |
|---|---:|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,706 |

### Amazon Beauty

The Stanford SNAP Amazon Beauty **5-core** subset (every user and item has at
least five reviews). After preprocessing the model vocabulary has 12,103
entries, including the special tokens.

### Preprocessing

Both datasets go through the same pipeline
([`src/data/preprocessing.py`](src/data/preprocessing.py)):

1. Load the interactions and sort them chronologically per user.
2. Remove items with fewer than five interactions.
3. Build one interaction sequence per user, keeping the most recent 200
   interactions.
4. Split chronologically, leaving the last two interactions out: the final
   item is the test target, the one before it the validation target, and the
   rest is training data. Users with fewer than three interactions are
   dropped.
5. Map item identifiers to contiguous model IDs.
6. Write model-specific training files for BERT4Rec and JEPA.

Special token IDs:

| Token | ID |
|---|---:|
| Padding (`PAD`) | 0 |
| Mask (`MASK`) | 1 |
| Items | 2 and above |

The two models use different training file formats, but both are derived from
the same chronological split. See [`data/README.md`](data/README.md) for the
file layout.

## Evaluation protocol

Models are evaluated on the held-out interaction of every user. All candidate
items are ranked, items the user has already interacted with are removed from
the ranking, and the position of the target item is scored.

| Metric | Description |
|---|---|
| HR@5 | Whether the target appears in the top 5 recommendations |
| HR@10 | Whether the target appears in the top 10 recommendations |
| NDCG@5 | Rank-discounted gain, cut off at rank 5 |
| NDCG@10 | Rank-discounted gain, cut off at rank 10 |

Validation NDCG@10 drives checkpoint selection, early stopping and
hyperparameter selection. The test set is only used for the final report.

Besides quality, every run records training and validation time per epoch,
parameter counts, peak GPU memory and convergence behaviour.

## Repository structure

```text
jepa-recommender/
├── configs/
│   ├── datasets/               # Per-dataset facts (vocabulary size, ...)
│   ├── ml-1m/                  # MovieLens-1M experiments
│   └── amazon_beauty/          # Amazon Beauty experiments
├── data/                       # raw/, interim/ and processed/ datasets
├── checkpoints/                # Trained model weights
├── results/                    # Histories, summaries and search reports
├── notebooks/                  # Exploratory analysis
├── docs/                       # Written results summary
├── scripts/                    # Command-line entry points
│   ├── download_*.py           #   fetch raw datasets
│   ├── preprocess_*.py         #   build model-ready datasets
│   ├── run_*_evaluation.py     #   test-set evaluation
│   ├── run_*_search.py         #   hyperparameter search
│   ├── run_cross_seed_comparison.py
│   └── summarize_runs.py       #   mean/std over groups of runs
├── src/
│   ├── data/                   # Datasets, loaders, splits, preprocessing
│   ├── models/                 # BERT4Rec, JEPA and the shared Transformer
│   ├── training/               # Training pipelines and shared helpers
│   ├── evaluation/             # Metrics, candidate filtering, evaluation
│   └── utils/                  # Config, paths, logging, checkpoints, seeding
└── tests/                      # Unit tests
```

Folders with generated or non-obvious content have their own README:
[`src/`](src/README.md), [`scripts/`](scripts/README.md),
[`configs/`](configs/README.md), [`data/`](data/README.md),
[`results/`](results/README.md), [`checkpoints/`](checkpoints/README.md) and
[`notebooks/`](notebooks/README.md).

## Installation

Requirements: Python 3.10 or later, PyTorch 2.2 or later, and (recommended for
training) a CUDA-capable NVIDIA GPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The right PyTorch build depends on your operating system and CUDA version;
follow the [official instructions](https://pytorch.org/get-started/locally/)
if the default wheel does not suit your machine. Then verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Optional extras:

| File | Adds |
|---|---|
| `requirements-dev.txt` | `pytest`, for the test suite |
| `requirements-notebooks.txt` | `matplotlib`, `seaborn`, Jupyter |

> **Running scripts.** Run everything from the repository root, and start
> scripts as modules (`python -m scripts.<name>`). Running a file directly
> (`python scripts/<name>.py`) does not put the repository on the import path.

## Data preparation

Download and preprocess a dataset with its two scripts:

```bash
# MovieLens-1M
python -m scripts.download_ml1m
python -m scripts.preprocess_ml1m

# Amazon Beauty (SNAP 5-core)
python -m scripts.download_amazon_beauty
python -m scripts.preprocess_amazon_beauty
```

Downloads go to `data/raw/<dataset>/`. Preprocessing writes
`data/interim/<dataset>/` and `data/processed/<dataset>/{bert4rec,jepa}/`.
If a download is blocked on your network, place the original files in
`data/raw/<dataset>/` by hand; the expected file names are listed in
[`data/README.md`](data/README.md).

## Experiment configuration

Experiments are YAML files under `configs/<dataset>/`:

| Configuration | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec |
| `bert4rec_tuned.yaml` | BERT4Rec with tuned hyperparameters |
| `jepa_causal_baseline.yaml` | Baseline causal JEPA |
| `jepa_causal_tuned.yaml` | Causal JEPA with tuned hyperparameters |
| `jepa_baseline.yaml`, `jepa_tuned.yaml` | Standard (non-causal) JEPA, MovieLens-1M only |
| `*_search.yaml` | Starting points for hyperparameter search (Amazon Beauty) |

Dataset facts (vocabulary size, maximum sequence length) live in
`configs/datasets/<dataset>.yaml` and take precedence over the experiment
file. The dataset is taken from the folder the experiment YAML sits in, or
chosen explicitly with `--dataset`, so one experiment file can be reused:

```bash
# MovieLens-1M (inferred from the folder)
python -m src.training.train_jepa_causal \
  --config configs/ml-1m/jepa_causal_tuned.yaml

# The same configuration on Amazon Beauty
python -m src.training.train_jepa_causal \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty
```

Paths in a config use the `{dataset}` placeholder, which the loader expands
and validates:

```yaml
data:
  train_path: "data/processed/{dataset}/jepa/train_context_target.parquet"
```

Command-line options shared by the trainers and evaluation scripts:

| Option | Effect |
|---|---|
| `--config PATH` | Experiment YAML |
| `--dataset NAME` | Dataset to run on |
| `--name NAME` | Rename the experiment; all output paths follow |
| `--set key=value` | Override one value, e.g. `--set training.learning_rate=1e-3` |

### Output organisation

```text
results/<dataset>/<experiment>/
├── config.yaml              # fully resolved configuration
├── training_history.csv     # one row per epoch
├── run_summary.json         # metrics, timing, hardware, files
└── test_results.json        # written by the evaluation scripts

checkpoints/<dataset>/<experiment>/<experiment>_best.pt
```

`--name` writes a run to a separate folder, which preserves earlier
experiments while an updated comparison runs.

## Training

Run from the repository root. Every trainer saves the best checkpoint (by
validation NDCG@10), the epoch history, the run summary and the resolved
configuration.

```bash
# BERT4Rec
python -m src.training.train_bert4rec \
  --config configs/amazon_beauty/bert4rec_tuned.yaml

# Causal JEPA
python -m src.training.train_jepa_causal \
  --config configs/amazon_beauty/jepa_causal_tuned.yaml

# Standard (non-causal) JEPA
python -m src.training.train_jepa \
  --config configs/ml-1m/jepa_tuned.yaml
```

Checkpoint saving can be disabled with `--set training.save_checkpoint=false`.

## Hyperparameter search

Both searches evaluate a grid of configurations on the validation set and
write aggregate reports to `results/<dataset>/<model>_grid_search/`
(`bert4rec_grid_search`, `jepa_grid_search`):

```text
search_summary.csv        # one row per trial
config_summary.csv        # trials aggregated by configuration
grid_search_report.json   # machine-readable report, used for resuming
```

```bash
python -m scripts.run_bert4rec_hyperparameter_search \
  --config configs/ml-1m/bert4rec_tuned.yaml \
  --dataset amazon_beauty

python -m scripts.run_jepa_hyperparameter_search \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing
```

Searches never write into a real experiment folder. The JEPA search keeps
per-trial files in a reused `_scratch/` folder inside the search directory;
the BERT4Rec search keeps no per-trial files at all. Interrupted searches
resume from their JSON report (`--restart` or `--no-resume`, depending on the
script, starts over).

Selected hyperparameters are copied into the tuned YAML files by hand. The
baseline files stay unchanged so the initial setup is preserved, and test
metrics are only computed once selection is finished.

## Cross-seed comparison

Repeat a configuration across random seeds to judge run-to-run noise:

```bash
python -m scripts.run_cross_seed_comparison \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing

python -m scripts.run_cross_seed_comparison \
  --config configs/ml-1m/bert4rec_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing
```

Reports go to `results/<dataset>/<model>_seed_search/` (`bert4rec_seed_search`,
`jepa_seed_search`):

| File | Content |
|---|---|
| `seed_results.csv` | One row per seed: metrics, best epoch, training time |
| `metric_summary.csv` | Mean and spread across successful seeds |
| `cross_seed_report.json` | Metadata, per-seed results and run status |

The default seeds are `42, 123, 456, 789, 2026`; use `--seeds` to change them.
Cross-seed comparisons use validation metrics. To compare groups of finished
runs, including test metrics, use
[`scripts/summarize_runs.py`](scripts/summarize_runs.py).

## Test-set evaluation

Evaluate a trained checkpoint on the held-out test set:

```bash
python -m scripts.run_bert4rec_evaluation \
  --config configs/amazon_beauty/bert4rec_tuned.yaml

python -m scripts.run_jepa_evaluation \
  --config configs/amazon_beauty/jepa_causal_tuned.yaml
```

The checkpoint defaults to the one the same configuration produced during
training; pass `--checkpoint PATH` to use another. Metrics are written to
`results/<dataset>/<experiment>/test_results.json`.

## Results and experiment tracking

```text
results/
├── ml-1m/
│   ├── bert4rec_baseline/        bert4rec_tuned/
│   ├── jepa_baseline/            jepa_tuned/
│   ├── jepa_causal_baseline/     jepa_causal_tuned/
│   ├── bert4rec_grid_search/     jepa_grid_search/
│   └── bert4rec_seed_search/     jepa_seed_search/
└── amazon_beauty/
    ├── bert4rec_baseline/        bert4rec_tuned/
    ├── jepa_causal_baseline/     jepa_causal_tuned/
    ├── bert4rec_grid_search/     jepa_grid_search/
    └── bert4rec_seed_search/     jepa_seed_search/
```

A written summary of the findings is in
[`docs/results_summary.md`](docs/results_summary.md). See
[`results/README.md`](results/README.md) and
[`checkpoints/README.md`](checkpoints/README.md) for the file formats.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover configuration loading and overrides, learning-rate schedules,
window construction and the shared preprocessing steps. They need neither a
GPU nor any dataset.

## Reproducibility

- Seeds are set for Python, NumPy and PyTorch, and cuDNN runs in
  deterministic mode.
- Every run stores its fully resolved configuration next to its results.
- Splits are chronological and shared between both models.
- Models are selected on validation data only; the test set is evaluated once
  per final configuration.
- Use several seeds to judge stability (see
  [Cross-seed comparison](#cross-seed-comparison)).

Exact numbers can still vary with hardware, library versions and
non-deterministic GPU operations.

## Citation and acknowledgements

If you use MovieLens-1M, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. *ACM Transactions on Interactive Intelligent Systems*,
> 5(4), Article 19. https://doi.org/10.1145/2827872

MovieLens-1M is provided by the GroupLens Research Project at the University
of Minnesota and must not be redistributed without permission. The Amazon
Beauty data come from the Stanford SNAP Amazon product reviews collection.
Both datasets are used for academic research; their inclusion does not imply
endorsement by the providers.
