# Experiment Configurations

This directory contains the YAML configuration files used to define model architecture, training parameters, dataset paths, and experiment outputs.

## Configuration Files

| File | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec configuration |
| `bert4rec_tuned.yaml` | Tuned BERT4Rec configuration |
| `jepa_baseline.yaml` | Baseline JEPA configuration |
| `jepa_tuned.yaml` | Tuned JEPA configuration |

## Configuration Contents

Configurations may specify:

- Random seed and device settings
- Dataset paths
- Model architecture parameters
- Training hyperparameters
- Validation and early stopping settings
- Checkpoint directories
- Results directories
- Evaluation settings

The exact available parameters are defined by each YAML file and the corresponding training or evaluation implementation.

## Running Experiments

Run training commands from the repository root.

```bash
python -m src.training.train_bert4rec --config configs/bert4rec_baseline.yaml
python -m src.training.train_bert4rec --config configs/bert4rec_tuned.yaml

python -m src.training.train_jepa --config configs/jepa_baseline.yaml
python -m src.training.train_jepa --config configs/jepa_tuned.yaml
```

## Reproducibility

Treat each configuration as the recorded specification of an experiment. When changing hyperparameters, preserve the original configuration or create a separately named configuration so that previous runs remain identifiable.

A configuration alone may not fully guarantee bit-for-bit reproducibility. Record the dataset version, software environment, and hardware used alongside experimental results.