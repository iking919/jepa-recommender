# Causal JEPA for Sequential Recommendation

A research project investigating **Joint-Embedding Predictive Architectures (JEPA)** for sequential recommendation, with a focus on causal, one-pass training and its comparison against **BERT4Rec**.

The project evaluates recommendation quality, training efficiency, and computational requirements across sequential recommendation datasets, including MovieLens-1M and Amazon Beauty.

## Overview

Sequential recommendation models learn from a user's interaction history to predict items they may interact with next.

This project investigates whether a causal JEPA framework can effectively learn sequential user representations by predicting future item representations in an embedding space, rather than directly predicting item identities during representation learning.

The primary comparison is between two approaches:

- **BERT4Rec:** A bidirectional Transformer-based sequential recommendation model trained using masked item prediction.
- **Causal JEPA:** A causal Joint-Embedding Predictive Architecture that predicts future item representations from encoded interaction contexts.

A central research objective is to investigate whether causal, one-pass JEPA training can reduce redundant computation while maintaining competitive recommendation performance.

### Research Objectives

The project aims to:

1. Implement a modular causal JEPA model for sequential recommendation.
2. Establish BERT4Rec as a comparative sequential recommendation baseline.
3. Compare recommendation performance under consistent dataset splits and evaluation protocols.
4. Investigate the computational efficiency of causal JEPA training.
5. Examine the effect of hyperparameter selection on recommendation quality and training efficiency.
6. Compare training time, convergence behavior, parameter counts, and GPU memory usage.
7. Evaluate model performance across different sequential recommendation datasets.

## Models

### BERT4Rec

BERT4Rec serves as the non-JEPA sequential recommendation baseline.

It applies a bidirectional Transformer to user interaction sequences and learns through masked item prediction. During training, items are masked and the model learns to reconstruct their identities using surrounding sequence context.

During evaluation, the model produces ranking scores for candidate items to predict a user's held-out interaction.

### Causal JEPA

Causal JEPA is the primary research model.

Rather than directly predicting item identities, the model learns representations of user interaction histories and predicts future target representations in an embedding space.

The architecture consists of:

- **Context encoder:** Encodes a user's interaction history using causal attention.
- **Target encoder:** Produces target representations used as the prediction objective during training.
- **Predictor:** Predicts future target representations from encoded context representations.
- **Exponential moving average (EMA):** Updates the target encoder using an exponential moving average of the context encoder's parameters.

The causal attention mechanism ensures that representations only incorporate information available up to the corresponding position in the sequence.

The training pipeline uses a one-pass windowing strategy to derive multiple prediction targets from sequence representations, reducing redundant forward passes compared with repeatedly encoding separate prefixes.

The project investigates whether this approach provides a favorable balance between recommendation quality and computational efficiency.

## Datasets

The project supports multiple sequential recommendation datasets.

### MovieLens-1M

MovieLens-1M, published by the GroupLens Research Project at the University of Minnesota, contains timestamped user ratings.

The original dataset contains:

| Statistic | Count |
|---|---:|
| Ratings | 1,000,209 |
| Users | 6,040 |
| Movies | 3,706 |

The project uses timestamps to construct chronological user interaction sequences.

### Amazon Beauty

Amazon Beauty is derived from the Amazon Reviews dataset and contains user-product interaction histories.

The project uses the Core 5 dataset, retaining products and users that satisfy the minimum interaction requirements of the preprocessing pipeline.

The processed dataset contains a model vocabulary of 12,103 entries, including special tokens.

### Preprocessing

The preprocessing pipeline generally includes:

1. Loading the original interaction data.
2. Sorting user interactions chronologically.
3. Filtering users and items according to the dataset's minimum interaction requirements.
4. Constructing user interaction sequences.
5. Mapping item identifiers to contiguous model item IDs.
6. Limiting sequence length to the configured maximum.
7. Creating chronological training, validation, and test splits.
8. Preparing model-specific training examples for BERT4Rec and causal JEPA.

The default maximum sequence length is 200.

Special token IDs are defined as follows:

| Token | ID |
|---|---:|
| Padding (`PAD`) | 0 |
| Mask (`MASK`) | 1 |
| Item IDs | 2 and above |

BERT4Rec and causal JEPA use different training data formats, but their examples are derived from the same underlying chronological interaction data.

Refer to [`data/README.md`](data/README.md) for additional preprocessing details.

## Evaluation

Models are evaluated on held-out interactions using ranking-based recommendation metrics.

| Metric | Description |
|---|---|
| HR@5 | Whether the held-out target appears in the top 5 recommendations |
| HR@10 | Whether the held-out target appears in the top 10 recommendations |
| NDCG@5 | Ranking quality with logarithmic discounting at rank 5 |
| NDCG@10 | Ranking quality with logarithmic discounting at rank 10 |

The evaluation pipeline ranks candidate items and excludes previously interacted items where configured.

### Model Selection

Validation performance is used for hyperparameter selection and checkpoint selection.

The held-out test set is reserved for final evaluation and should not influence model selection.

### Computational Evaluation

In addition to recommendation quality, the project records computational measurements such as:

- Total training time.
- Epoch duration.
- Training throughput.
- Number of model parameters.
- Peak GPU memory usage.
- Validation performance and convergence behavior.

These measurements help evaluate whether causal JEPA offers computational advantages over BERT4Rec.

## Repository Structure

```text
jepa-recommender/
├── configs/
│   ├── datasets/                  # Dataset-specific configuration
│   ├── ml-1m/                     # MovieLens-1M experiments
│   └── amazon_beauty/             # Amazon Beauty experiments
├── data/
│   ├── raw/                       # Original datasets
│   ├── interim/                   # Intermediate preprocessing artifacts
│   └── processed/                 # Model-ready datasets
├── checkpoints/                   # Trained model weights
├── results/                       # Experiment results and reports
├── notebooks/                     # Exploratory analysis
├── scripts/                       # Preprocessing, search, and evaluation
└── src/
    ├── data/                      # Dataset loading and preprocessing
    ├── models/                    # Model architectures
    ├── training/                  # Training pipelines and shared utilities
    ├── evaluation/                # Evaluation metrics and logic
    └── utils/                     # Configuration and utility functions
```

## Installation

### Requirements

- Python 3.10 or later
- PyTorch 2.2 or later
- CUDA-compatible NVIDIA GPU recommended for training

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

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The appropriate PyTorch installation depends on the operating system, GPU, and CUDA support. Follow the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install a compatible build.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data Preparation

Download the required dataset and place its original files in the corresponding directory under:

```text
data/raw/<dataset>/
```

For MovieLens-1M, the expected structure is:

```text
data/raw/ml-1m/
├── README
├── movies.dat
├── ratings.dat
└── users.dat
```

If the repository's download and preprocessing scripts are available, MovieLens-1M can be prepared using:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

Refer to `data/README.md` and the available preprocessing scripts for the exact workflow for each dataset.

Processed data is stored under:

```text
data/processed/<dataset>/
├── bert4rec/
└── jepa/
```

The two model pipelines use separate training formats derived from the same underlying data splits.

## Experiment Configurations

Experiment parameters are defined in YAML files under `configs/<dataset>/`.

The primary configurations include:

| Configuration | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec experiment |
| `bert4rec_tuned.yaml` | Tuned BERT4Rec experiment |
| `jepa_causal_baseline.yaml` | Baseline causal JEPA experiment |
| `jepa_causal_tuned.yaml` | Tuned causal JEPA experiment |

Additional search configurations may be provided for hyperparameter optimization.

### Dataset Configuration

Dataset-specific facts, including vocabulary size and maximum sequence length, are stored in:

```text
configs/datasets/<dataset>.yaml
```

For example:

```text
configs/datasets/amazon_beauty.yaml
configs/datasets/ml-1m.yaml
```

The configuration loader combines the experiment configuration with the selected dataset configuration.

A dataset can be inferred from the experiment configuration's directory, or explicitly selected using `--dataset`.

For example:

```bash
# Run on MovieLens-1M
python -m src.training.train_jepa_causal \
  --config configs/ml-1m/jepa_causal_tuned.yaml

# Run the same configuration on Amazon Beauty
python -m src.training.train_jepa_causal \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty
```

Data paths use the `{dataset}` placeholder:

```yaml
data:
  train_path: "data/processed/{dataset}/jepa/train_context_target.parquet"
  validation_path: "data/processed/{dataset}/jepa/validation.parquet"
  test_path: "data/processed/{dataset}/jepa/test.parquet"
```

The configuration loader expands these paths and checks that they belong to the selected dataset.

### Output Organization

Experiment outputs are separated by dataset and experiment name.

```text
results/
└── <dataset>/
    └── <experiment>/
        ├── config.yaml
        ├── training_history.csv
        ├── run_summary.json
        └── test_results.json

checkpoints/
└── <dataset>/
    └── <experiment>/
        └── <experiment>_best.pt
```

The resolved configuration is saved alongside the experiment results.

Use `--name` to assign a different experiment name without modifying the YAML file.

Use `--set` to override individual configuration values.

For example:

```bash
python -m src.training.train_jepa_causal \
  --config configs/amazon_beauty/jepa_causal_tuned.yaml \
  --name jepa_causal_comparison
```

This creates a separate experiment directory:

```text
results/amazon_beauty/jepa_causal_comparison/
checkpoints/amazon_beauty/jepa_causal_comparison/
```

This is useful for preserving previous experiments while running updated comparisons.

## Training

Run commands from the repository root.

### BERT4Rec

Baseline:

```bash
python -m src.training.train_bert4rec \
  --config configs/amazon_beauty/bert4rec_baseline.yaml
```

Tuned:

```bash
python -m src.training.train_bert4rec \
  --config configs/amazon_beauty/bert4rec_tuned.yaml
```

### Causal JEPA

Baseline:

```bash
python -m src.training.train_jepa_causal \
  --config configs/amazon_beauty/jepa_causal_baseline.yaml
```

Tuned:

```bash
python -m src.training.train_jepa_causal \
  --config configs/amazon_beauty/jepa_causal_tuned.yaml
```

Training saves the best checkpoint, training history, run summary, and resolved configuration in their respective output directories.

## Hyperparameter Search

Hyperparameter search investigates optimization settings for BERT4Rec and causal JEPA.

The search process evaluates candidate configurations using validation performance and records the results for analysis.

Example BERT4Rec search:

```bash
python -m scripts.run_bert4rec_hyperparameter_search \
  --config configs/ml-1m/bert4rec_tuned.yaml \
  --dataset amazon_beauty
```

Example causal JEPA search:

```bash
python -m scripts.run_jepa_hyperparameter_search \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing
```

Search reports are stored separately from ordinary training results.

Typical outputs include:

```text
results/<dataset>/<model>_grid_search/
├── search_summary.csv
├── config_summary.csv
├── grid_search_report.json
└── _scratch/
```

The search summary contains trial-level results, while the JSON report contains machine-readable experiment metadata.

The `_scratch/` directory is used for temporary trial artifacts and should not be confused with final experiment outputs.

### Selecting Tuned Configurations

Validation performance, particularly NDCG@10, is used to select promising configurations.

Selected hyperparameters are recorded in the corresponding tuned YAML configuration.

The original baseline configuration should remain unchanged to preserve the initial experimental setup.

Final test metrics should only be calculated after hyperparameter selection is complete.

## Cross-Seed Comparison

Cross-seed experiments investigate the stability of model performance under different random seeds while keeping the remaining configuration fixed.

The repository includes:

```text
scripts/run_cross_seed_comparison.py
```

Example causal JEPA experiment:

```bash
python -m scripts.run_cross_seed_comparison \
  --config configs/ml-1m/jepa_causal_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing
```

Example BERT4Rec experiment:

```bash
python -m scripts.run_cross_seed_comparison \
  --config configs/ml-1m/bert4rec_tuned.yaml \
  --dataset amazon_beauty \
  --allow-existing
```

Typical reports include:

- `seed_results.csv`: Individual run results, including metrics, best epoch, and training time.
- `seed_summary.csv`: Aggregate statistics across successful seeds.
- `cross_seed_report.json`: Experiment metadata, individual results, and run status.

The default seed set is:

```text
42, 123, 456, 789, 2026
```

Use `--seeds` to specify a different set.

Cross-seed comparisons use validation performance for model selection. Test performance should be reported separately.

## Evaluation

Evaluation uses a trained checkpoint and its corresponding experiment configuration.

For example:

```bash
# BERT4Rec
python -m scripts.run_bert4rec_evaluation \
  --config configs/amazon_beauty/bert4rec_tuned.yaml

# Causal JEPA
python -m scripts.run_jepa_evaluation \
  --config configs/amazon_beauty/jepa_causal_tuned.yaml
```

The evaluation scripts support the configuration options used during training. An explicit checkpoint can also be supplied using `--checkpoint`.

Test metrics are written to:

```text
results/<dataset>/<experiment>/test_results.json
```

## Results and Experiment Tracking

Each experiment has a dedicated results directory.

For example:

```text
results/
├── ml-1m/
│   ├── bert4rec_baseline/
│   ├── bert4rec_tuned/
│   ├── jepa_causal_baseline/
│   ├── jepa_causal_tuned/
│   ├── bert4rec_grid_search/
│   └── jepa_causal_grid_search/
└── amazon_beauty/
    ├── bert4rec_baseline/
    ├── bert4rec_tuned/
    ├── jepa_causal_baseline/
    ├── jepa_causal_tuned/
    ├── bert4rec_comparison/
    └── jepa_causal_comparison/
```

Depending on the experiment pipeline, results may include:

| File | Description |
|---|---|
| `config.yaml` | Fully resolved experiment configuration |
| `training_history.csv` | Epoch-level training and validation history |
| `run_summary.json` | Run-level metrics and experiment metadata |
| `test_results.json` | Final held-out test metrics |

Model checkpoints are stored separately under `checkpoints/`.

See `results/README.md` and `checkpoints/README.md` for additional details.

## Reproducibility

The project uses configuration files and shared utility functions to support consistent experiment execution and random seeding.

For reproducible experiments:

1. Use consistent dataset preprocessing and chronological splits.
2. Preserve the configuration used for each experiment.
3. Record hardware and software environment details.
4. Retain training histories and hyperparameter search reports.
5. Select models using validation performance.
6. Evaluate final performance on the held-out test set.
7. Use multiple random seeds when assessing experimental stability.
8. Document any differences in model architecture or training procedure.

Exact results may vary due to hardware, software versions, nondeterministic GPU operations, and training conditions.

## Research Status

The current research focuses on comparing BERT4Rec and causal JEPA for sequential recommendation.

The primary experimental questions are:

- Can causal JEPA achieve competitive recommendation performance relative to BERT4Rec?
- Does one-pass causal training reduce training time or computational requirements?
- How do the approaches differ in convergence behavior and resource usage?
- Are observed performance differences consistent across datasets and random seeds?

Updated comparisons are being conducted on Amazon Beauty using the modularized training pipelines. Final conclusions will be based on validated held-out test results and computational measurements.

## Citation

If you use MovieLens-1M, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872

Please follow the MovieLens dataset usage terms. The dataset must not be redistributed without permission.

## Acknowledgements

MovieLens-1M is provided by the GroupLens Research Project at the University of Minnesota.

The Amazon Reviews dataset is used as an additional source of sequential interaction data.

These datasets are used for academic research and their inclusion does not imply endorsement by their respective providers.