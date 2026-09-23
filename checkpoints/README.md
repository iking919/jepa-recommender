# Model Checkpoints

This directory stores trained model checkpoints generated during experiments.

## Organization

Checkpoints are organized by model and experiment:

```text
checkpoints/
├── README.md
├── bert4rec_baseline/
│   └── bert4rec_baseline_best.pt
├── bert4rec_tuned/
│   └── bert4rec_tuned_best.pt
├── jepa_baseline/
│   └── jepa_baseline_best.pt
└── jepa_tuned/
    └── jepa_tuned_best.pt
```

## Checkpoint Selection

For experiments with validation data, the best checkpoint is selected using validation **NDCG@10**.

The checkpoint contains the trained model parameters and the information needed to reproduce the associated experiment, including the training configuration and relevant training/validation history.

## Reproducibility

Each checkpoint should correspond to a configuration file in `configs/`.

For example:

```text
configs/bert4rec_tuned.yaml
        ↓
checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
        ↓
results/bert4rec/
```

The configuration defines the model architecture, training parameters, dataset paths, and checkpoint settings. Final evaluation metrics are stored separately under `results/`.

## Git Tracking

Model checkpoint files (`.pt`) are excluded from version control because trained model weights can be large.

The repository retains the configuration files and documentation needed to reproduce the experiments. Checkpoints may be generated locally by the training scripts.