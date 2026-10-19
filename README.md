# JEPA for Sequential Recommendation

A research project investigating **Joint-Embedding Predictive Architectures (JEPA)** for sequential recommendation and comparing a JEPA-based approach with **BERT4Rec** on the MovieLens-1M dataset.

## Overview

Sequential recommendation models learn from a user's interaction history to predict items they may interact with next. This project investigates whether a JEPA-style learning framework can support sequential recommendation by predicting future item representations rather than directly predicting item identities during representation learning.

The project compares two approaches:

- **BERT4Rec:** A bidirectional Transformer-based sequential recommendation model trained using a masked item prediction objective.
- **JEPA:** A Joint-Embedding Predictive Architecture that learns to predict target representations from encoded user interaction contexts.

The repository also investigates a causal, one-pass JEPA training approach designed to reduce redundant computation across user interaction sequences.

The models are evaluated on MovieLens-1M using chronological recommendation tasks and ranking-based evaluation metrics.

### Research Objectives

The primary objectives are to:

1. Implement a modular JEPA-based sequential recommendation model.
2. Establish BERT4Rec as a sequential recommendation baseline.
3. Compare recommendation performance under a shared dataset and evaluation protocol.
4. Examine the impact of model architecture and training configuration on recommendation performance.
5. Investigate the effect of causal, one-pass training on JEPA training efficiency.
6. Use hyperparameter search to identify promising optimization settings.
7. Record training time, validation performance, and resource usage to support analysis of computational efficiency.

## Models

### BERT4Rec

BERT4Rec is used as a sequential recommendation baseline. It applies a bidirectional Transformer to user interaction sequences and learns through masked item prediction.

The model is trained to predict masked items from surrounding interaction context. During evaluation, it produces item-ranking scores for the held-out target item.

The repository includes baseline and tuned configurations:

- `bert4rec_baseline.yaml`
- `bert4rec_tuned.yaml`

### JEPA

The JEPA-based model learns representations of user interaction histories and predicts target item representations in an embedding space.

The architecture consists of:

- **Context encoder:** Encodes a user's interaction history.
- **Target encoder:** Produces target item representations.
- **Predictor:** Predicts target representations from context representations.
- **Exponential moving average (EMA):** Updates the target encoder using the context encoder's parameters.

The target encoder provides target representations during training. At inference time, the model uses learned representations to rank candidate items.

The standard JEPA implementation includes baseline and tuned configurations:

- `jepa_baseline.yaml`
- `jepa_tuned.yaml`

### Causal JEPA

The causal JEPA implementation uses a causal Transformer encoder and a one-pass training strategy. Rather than repeatedly encoding a separate prefix for each prediction target, the training pipeline processes user interaction sequences through a smaller number of windows and derives multiple supervised targets from those windows.

This approach is intended to reduce redundant forward passes and training time while retaining a sequential prediction objective.

The repository includes the following configurations:

- `jepa_causal_baseline.yaml`
- `jepa_causal_tuned.yaml`

The causal baseline uses starting optimization settings before hyperparameter tuning. The tuned configuration uses the selected optimization settings from the search process.

The causal implementation is evaluated separately from the standard JEPA implementation because the training strategies and model configurations differ. Comparisons should document these differences when interpreting performance and computational efficiency.

## Dataset

The primary dataset is **MovieLens-1M**, published by the GroupLens Research Project at the University of Minnesota.

The original dataset contains:

| Statistic | Count |
|---|---:|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,706 |

The project uses timestamped ratings to construct chronological user interaction sequences.

### Preprocessing

The experimental preprocessing pipeline includes:

1. Loading the original MovieLens-1M ratings.
2. Sorting user interactions chronologically.
3. Filtering movies with fewer than five interactions.
4. Constructing user interaction sequences.
5. Mapping movie IDs to contiguous model item IDs.
6. Limiting sequence length to a maximum of 200.
7. Creating chronological training, validation, and test splits.
8. Preparing model-specific training examples for BERT4Rec and JEPA.

The current processed dataset contains:

| Statistic | Value |
|---|---:|
| Users | 6,040 |
| Mapped movies | 3,416 |
| Vocabulary entries, including special tokens | 3,418 |
| Maximum sequence length | 200 |
| Training interactions | 648,759 |

Special token IDs are defined as follows:

| Token | ID |
|---|---:|
| Padding (`PAD`) | 0 |
| Mask (`MASK`) | 1 |
| Movie items | 2 and above |

BERT4Rec and JEPA use model-specific training examples derived from the shared chronological data. Refer to [`data/README.md`](data/README.md) for details.

## Evaluation

The models are evaluated on held-out test interactions using the following ranking metrics:

- **HR@5:** Whether the held-out target item appears in the top 5 recommendations.
- **HR@10:** Whether the held-out target item appears in the top 10 recommendations.
- **NDCG@5:** Normalized discounted cumulative gain at rank 5.
- **NDCG@10:** Normalized discounted cumulative gain at rank 10.

The evaluation pipeline ranks candidate movies and excludes items the user has already interacted with, where configured.

Validation performance is used for model selection, including hyperparameter selection and checkpoint selection. The held-out test set is reserved for final evaluation.

For fair comparisons, use consistent data splits, candidate-item handling, ranking metrics, and evaluation settings. Document differences between the standard and causal JEPA implementations where relevant.

## Repository Structure

```text
jepa-recommender/
├── configs/                       # Experiment configurations
├── data/
│   ├── raw/                       # Original dataset
│   ├── interim/                   # Intermediate preprocessing artifacts
│   └── processed/                 # Model-ready datasets
├── checkpoints/                   # Trained model weights
├── results/                       # Metrics, histories, and summaries
│   └── jepa_grid_search/          # Causal JEPA hyperparameter-search results
├── notebooks/                     # Exploratory analysis and experiments
├── scripts/                       # Dataset, search, and evaluation entry points
└── src/
    ├── data/                      # Dataset loading and preprocessing
    ├── models/                    # BERT4Rec and JEPA architectures
    ├── training/                  # Training loops and EMA
    ├── evaluation/                # Metrics and evaluation logic
    └── utils/                     # Configuration, logging, and utilities
```

## Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.2 or later
- CUDA-compatible GPU recommended for training

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows (Git Bash):

```bash
source .venv/Scripts/activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch and CUDA:** The appropriate PyTorch build depends on your operating system, GPU, and CUDA support. If you intend to train on an NVIDIA GPU, follow the official PyTorch installation instructions to install a compatible CUDA-enabled build.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data Preparation

Download the official MovieLens-1M dataset from GroupLens and extract the dataset files into:

```text
data/raw/ml-1m/
```

The expected files are:

```text
data/raw/ml-1m/
├── README
├── movies.dat
├── ratings.dat
└── users.dat
```

Do not modify the original dataset files.

Run the preprocessing entry points available in the repository to generate the intermediate and processed datasets. If the repository includes `scripts/download_ml1m.py` and `scripts/preprocess_ml1m.py`, the commands are:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

Refer to `data/README.md` and the available scripts for the exact data preparation workflow.

## Experiment Configurations

Experiment parameters are stored in YAML files under `configs/`.

| Configuration | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec experiment |
| `bert4rec_tuned.yaml` | Tuned BERT4Rec experiment |
| `jepa_baseline.yaml` | Baseline standard JEPA experiment |
| `jepa_tuned.yaml` | Tuned standard JEPA experiment |
| `jepa_causal_baseline.yaml` | Baseline causal, one-pass JEPA experiment |
| `jepa_causal_tuned.yaml` | Tuned causal, one-pass JEPA experiment |

Each configuration specifies dataset paths, model architecture, training parameters, and output locations.

The causal JEPA configurations use the causal training implementation and should be treated as separate experiments from the standard JEPA configurations.

## Training

Run commands from the repository root. Ensure the required processed datasets are available before starting training.

### BERT4Rec Baseline

```bash
python -m src.training.train_bert4rec \
  --config configs/bert4rec_baseline.yaml
```

### BERT4Rec Tuned

```bash
python -m src.training.train_bert4rec \
  --config configs/bert4rec_tuned.yaml
```

### Standard JEPA Baseline

```bash
python -m src.training.train_jepa \
  --config configs/jepa_baseline.yaml
```

### Standard JEPA Tuned

```bash
python -m src.training.train_jepa \
  --config configs/jepa_tuned.yaml
```

### Causal JEPA Baseline

```bash
python -m src.training.train_jepa_causal \
  --config configs/jepa_causal_baseline.yaml
```

### Causal JEPA Tuned

```bash
python -m src.training.train_jepa_causal \
  --config configs/jepa_causal_tuned.yaml
```

Training saves model checkpoints under `checkpoints/` and experiment outputs under the corresponding `results/` directory, as specified by the configuration and training pipeline.

## Hyperparameter Search

Hyperparameter search is used to investigate optimization settings for the BERT4Rec and causal, one-pass JEPA implementation. The search evaluates the parameter combinations defined in hyperparamter search script respectively.

The search is launched using each baseline configuration:

BERT4Rec:
```bash
python -m scripts.run_bert4rec_hyperparameter_search \
    --config configs/bert4rec_tuned.yaml \
    --output-dir results/bert4rec_grid_search
```

JEPA Causal:

```bash
python -m scripts.run_jepa_hyperparameter_search \
  --config configs/jepa_causal_tuned.yaml \
  --output-dir results/jepa_grid_search \
  --allow-existing
```

The baseline configuration provides the starting architecture and training setup. The search script generates trial-specific configurations, applies the candidate optimization settings, and records the resulting validation performance.

### Search Outputs

Search outputs are organized separately from ordinary training results:

```text
results/
└── <model>_grid_search/
    ├── search_summary.csv
    ├── grid_search_report.json
    ├── <trial_name_1>/
    │   ├── config.yaml
    │   ├── training_history.csv
    │   └── run_summary.json
    └── <trial_name_2>/
        ├── config.yaml
        ├── training_history.csv
        └── run_summary.json
```

The exact number and names of trials depend on the parameter grid defined in the search script.

- `search_summary.csv` contains the aggregate results for individual trials.
- `grid_search_report.json` contains the machine-readable search report.
- Each trial directory records the configuration used, training history, and run summary.

The search is intended to avoid saving a model checkpoint for every trial, while preserving the information needed to compare configurations. Verify that checkpoint saving is disabled in the search script and that the trainer honors that setting.

### Selecting the Tuned Configuration

Use validation performance, such as validation NDCG@10, to compare candidate configurations and select the optimization settings for the tuned causal JEPA experiment.

The selected settings should then be recorded in `configs/bert4rec_tuned.yaml` and `configs/jepa_causal_tuned.yaml` respectively. Keep the baseline configuration unchanged so the original settings remain available for comparison.

For reproducibility, retain the aggregate search results and the configuration for each trial. Report the final held-out test metrics only after selecting the configuration using validation data.

## Evaluation

Evaluation uses a trained checkpoint and the corresponding experiment configuration. Run the commands from the repository root so that project imports and relative paths resolve correctly.

### BERT4Rec Baseline

```bash
python -m scripts.run_bert4rec_evaluation \
  --config configs/bert4rec_baseline.yaml \
  --checkpoint checkpoints/bert4rec_baseline/bert4rec_baseline_best.pt
```

### BERT4Rec Tuned

```bash
python -m scripts.run_bert4rec_evaluation \
  --config configs/bert4rec_tuned.yaml \
  --checkpoint checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
```

### Standard JEPA Baseline

```bash
python -m scripts.run_jepa_evaluation \
  --config configs/jepa_baseline.yaml \
  --checkpoint checkpoints/jepa_baseline/jepa_baseline_best.pt
```

### Standard JEPA Tuned

```bash
python -m scripts.run_jepa_evaluation \
  --config configs/jepa_tuned.yaml \
  --checkpoint checkpoints/jepa_tuned/jepa_tuned_best.pt
```

### Causal JEPA Baseline

```bash
python -m scripts.run_jepa_evaluation \
  --config configs/jepa_causal_baseline.yaml \
  --checkpoint checkpoints/jepa_causal_baseline/jepa_causal_baseline_best.pt
```

### Causal JEPA Tuned

```bash
python -m scripts.run_jepa_evaluation \
  --config configs/jepa_causal_tuned.yaml \
  --checkpoint checkpoints/jepa_causal_tuned/jepa_causal_tuned_best.pt
```

## Results and Experiment Tracking

Each experiment has a dedicated results directory. For example:

```text
results/
├── bert4rec_baseline/
├── bert4rec_tuned/
├── jepa_baseline/
├── jepa_tuned/
├── jepa_causal_baseline/
└── jepa_causal_tuned/
```

Depending on the experiment pipeline, results may include:

- `config.yaml`: The configuration used for the run.
- `training_history.csv`: Epoch-level loss, validation metrics, and timing.
- `run_summary.json`: Configuration and run-level summary.
- `test_results.json`: Held-out test metrics, if produced by the evaluation pipeline.

The hyperparameter search directory contains aggregate trial results and trial-specific output folders.

Training time, validation time, epoch duration, and peak GPU memory are useful for assessing computational cost alongside recommendation metrics. Preserve the hardware and software environment details when reporting performance.

Model checkpoints are stored separately under `checkpoints/`. See `results/README.md` and `checkpoints/README.md` for additional details.

## Reproducibility

The project uses configuration files to define experiment settings and utility functions to support consistent random seeding and checkpoint management.

For reproducible experiments:

1. Use the same dataset and preprocessing settings.
2. Use the corresponding YAML configuration for each run.
3. Record the software and hardware environment.
4. Preserve training histories, trial configurations, search reports, and evaluation outputs.
5. Select hyperparameters and checkpoints using validation performance.
6. Report final results on the held-out test set.
7. Document the search space and selection criterion when reporting tuned results.

Exact results may vary with hardware, software versions, nondeterministic GPU operations, and training conditions.

## Current Results

The following are the current held-out test results for the standard BERT4Rec and JEPA configurations. These are experimental results and may be updated as the project continues.

| Model | Configuration | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| BERT4Rec | Baseline | 0.1444 | 0.2265 | 0.0956 | 0.1220 |
| BERT4Rec | Tuned | 0.1955 | 0.2884 | 0.12956 | 0.1593 |
| JEPA | Baseline | 0.0728 | 0.1065 | 0.0491 | 0.0598 |
| JEPA | Tuned | 0.1273 | 0.1801 | 0.0928 | 0.1098 |
| JEPA Causal | Baseline | 0.1545 | 0.1935 | 0.1108 | 0.1234 |
| JEPA Causal | Tuned | 0.1591 | 0.2065 | 0.1140 | 0.1292 |


## Research Status

This repository contains baseline and tuned experiment pipelines for BERT4Rec and standard JEPA, alongside a causal, one-pass JEPA implementation and a hyperparameter-search workflow.

Ongoing experiments investigate the relationship between recommendation quality, optimization settings, training time, and computational efficiency. Final comparative conclusions will be updated as the causal experiments and additional analyses are completed and validated.

## Citation

If you use MovieLens-1M, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872

Please also follow the MovieLens dataset's usage terms. The dataset must not be redistributed without permission.

## Acknowledgements

MovieLens-1M is provided by the GroupLens Research Project at the University of Minnesota. The dataset is used here for academic research and does not imply endorsement by GroupLens or the University of Minnesota.