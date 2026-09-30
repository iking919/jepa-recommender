# JEPA for Sequential Recommendation

A research project investigating **Joint-Embedding Predictive Architectures (JEPA)** for sequential recommendation and comparing a JEPA-based approach with **BERT4Rec** on the MovieLens-1M dataset.

## Overview

Sequential recommendation models learn from a user's interaction history to predict items they may interact with next. This project investigates whether a JEPA-style learning framework can support sequential recommendation by predicting future item representations rather than directly predicting item identities during representation learning.

The project compares two approaches:

- **BERT4Rec:** A bidirectional Transformer-based sequential recommendation model trained using a masked item prediction objective.
- **JEPA:** A Joint-Embedding Predictive Architecture that learns to predict target representations from encoded user interaction contexts.

Both approaches are evaluated on MovieLens-1M using a chronological recommendation task and ranking-based evaluation metrics.

### Research Objectives

The primary objectives are to:

1. Implement a modular JEPA-based sequential recommendation model.
2. Establish BERT4Rec as a sequential recommendation baseline.
3. Compare the recommendation performance of both approaches under a shared dataset and evaluation protocol.
4. Examine the impact of model architecture and training configuration on recommendation performance.
5. Record training time, validation performance, and resource usage to support analysis of computational efficiency.

## Models

### BERT4Rec

BERT4Rec is used as the sequential recommendation baseline. It applies a bidirectional Transformer to user interaction sequences and learns through masked item prediction.

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

The target encoder is used during training to provide target representations. Recommendation at inference time uses the context representation and prediction to rank candidate item embeddings.

The repository includes baseline and tuned configurations:

- `jepa_baseline.yaml`
- `jepa_tuned.yaml`

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

Special token IDs are defined as:

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

Final test metrics should be reported separately from validation metrics. Validation performance is used for model selection, while the held-out test set is used for final evaluation.

## Repository Structure

```text
jepa-recommender/
├── configs/                 # Experiment configurations
├── data/
│   ├── raw/                  # Original dataset
│   ├── interim/              # Intermediate preprocessing artifacts
│   └── processed/            # Model-ready datasets
├── checkpoints/              # Trained model weights
├── results/                  # Metrics, training history, and summaries
├── notebooks/                # Exploratory analysis and experiments
├── scripts/                  # Dataset and evaluation entry points
└── src/
    ├── data/                 # Dataset loading and preprocessing
    ├── models/               # BERT4Rec and JEPA architectures
    ├── training/             # Training loops and EMA
    ├── evaluation/           # Metrics and evaluation logic
    └── utils/                # Configuration, logging, and utilities
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

**PyTorch and CUDA:** The appropriate PyTorch build depends on your operating system, GPU, and CUDA support. If you intend to train on an NVIDIA GPU, follow the official PyTorch installation instructions to install a compatible CUDA-enabled build. Verify availability with:

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

Run the preprocessing entry point available in the repository to generate the intermediate and processed datasets. If the repository includes `scripts/download_ml1m.py` and `scripts/preprocess_ml1m.py`, the commands are:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

Refer to `data/README.md` and the available scripts for the exact data preparation workflow.

## Experiment Configurations

Experiment parameters are stored in YAML files under `configs/`:

| Configuration | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec experiment |
| `bert4rec_tuned.yaml` | Tuned BERT4Rec experiment |
| `jepa_baseline.yaml` | Baseline JEPA experiment |
| `jepa_tuned.yaml` | Tuned JEPA experiment |

Each configuration specifies the relevant dataset paths, model architecture, training parameters, and output locations.

## Training

Run commands from the repository root.

### BERT4Rec Baseline

```bash
python -m src.training.train_bert4rec --config configs/bert4rec_baseline.yaml
```

### BERT4Rec Tuned

```bash
python -m src.training.train_bert4rec --config configs/bert4rec_tuned.yaml
```

### JEPA Baseline

```bash
python -m src.training.train_jepa --config configs/jepa_baseline.yaml
```

### JEPA Tuned

```bash
python -m src.training.train_jepa --config configs/jepa_tuned.yaml
```

Training saves model checkpoints under `checkpoints/` and experiment outputs under the corresponding `results/` directory, as specified by the training pipeline.

## Evaluation

Evaluation uses a trained checkpoint and the corresponding experiment configuration.

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

### JEPA Baseline

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_baseline.yaml \
    --checkpoint checkpoints/jepa_baseline/jepa_baseline_best.pt
```

### JEPA Tuned

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_tuned.yaml \
    --checkpoint checkpoints/jepa_tuned/jepa_tuned_best.pt
```

These commands assume the corresponding scripts and checkpoints are present. Run the commands from the repository root so the project imports and relative paths resolve correctly.

## Results and Experiment Tracking

Each experiment has a dedicated results directory:

```text
results/
├── bert4rec_baseline/
├── bert4rec_tuned/
├── jepa_baseline/
└── jepa_tuned/
```

Depending on the experiment pipeline, results may include:

- `training_history.csv`: Epoch-level loss, validation metrics, and timing.
- `run_summary.json`: Configuration and run-level summary.
- `test_results.json`: Held-out test metrics.

Training time, validation time, epoch duration, and peak GPU memory are useful for assessing computational cost alongside recommendation metrics.

Model checkpoints are stored separately under `checkpoints/`. See `results/README.md` and `checkpoints/README.md`.

## Reproducibility

The project uses configuration files to define experiment settings and utility functions to support consistent random seeding and checkpoint management.

For reproducible experiments:

1. Use the same dataset and preprocessing settings.
2. Use the corresponding YAML configuration for each run.
3. Record the software and hardware environment.
4. Preserve training history and evaluation outputs.
5. Select checkpoints using validation performance and report final results on the held-out test set.

Exact results may vary with hardware, software versions, nondeterministic GPU operations, and training conditions.

## Current Results

The following are the current held-out test results for the four experimental configurations. These results are considered **current experimental results** and may be updated as the project continues.

| Model | Configuration | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| BERT4Rec | Baseline | 0.0159 | 0.0356 | 0.0100 | 0.0163 |
| BERT4Rec | Tuned | 0.1416 | 0.2182 | 0.0893 | 0.1138 |
| JEPA | Baseline | 0.0247 | 0.0328 | 0.0184 | 0.0209 |
| JEPA | Tuned | 0.1285 | 0.1820 | 0.0917 | 0.1089 |

Results are reported on the held-out test set using the same evaluation metrics across experiments. Model checkpoints are selected using validation performance; the test set is reserved for final evaluation.

These results are subject to change as the experimental pipeline is further validated and additional analyses are performed.

## Research Status

This repository contains baseline and tuned experiment pipelines for BERT4Rec and JEPA. Final comparative results and conclusions will be updated as experiments are completed and validated.

## Citation

If you use MovieLens-1M, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872

Please also follow the MovieLens dataset's usage terms. The dataset must not be redistributed without permission.

## Acknowledgements

MovieLens-1M is provided by the GroupLens Research Project at the University of Minnesota. The dataset is used here for academic research and does not imply endorsement by GroupLens or the University of Minnesota.