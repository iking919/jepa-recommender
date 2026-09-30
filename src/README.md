# Source Code

This directory contains the reusable implementation of the sequential recommendation experiments.

The code is organized into modular packages for data processing, model architectures, training, evaluation, and shared utilities.

## Directory Structure

```text
src/
├── data/
├── models/
├── training/
├── evaluation/
└── utils/
```

## Modules

### `data/`

Contains dataset loading, preprocessing, sequence construction, and dataset classes for BERT4Rec and JEPA.

### `models/`

Contains the model architectures, including:

- BERT4Rec
- JEPA
- Shared Transformer components
- JEPA prediction components

### `training/`

Contains model training pipelines and reusable training components, including the EMA update mechanism used by JEPA.

### `evaluation/`

Contains recommendation evaluation logic, ranking metrics, and recommendation utilities used to assess trained models.

### `utils/`

Contains shared utilities for configuration loading, logging, checkpoint management, and reproducibility.

## Design Principles

The source code is organized to:

- Keep model architecture separate from training logic.
- Reuse common components where appropriate.
- Keep experiment-specific parameters in YAML configuration files.
- Separate validation and held-out test evaluation.
- Keep checkpoints separate from experiment result artifacts.

## Running Training

From the repository root:

```bash
python -m src.training.train_bert4rec --config configs/bert4rec_baseline.yaml
python -m src.training.train_jepa --config configs/jepa_baseline.yaml
```

Use the corresponding tuned configuration files to run tuned experiments.