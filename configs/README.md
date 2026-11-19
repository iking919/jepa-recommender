# Experiment Configurations

This directory contains the YAML configuration files used to define model architecture and training parameters. Dataset facts and output locations are filled in from the chosen dataset.

## Configuration Files

Experiment configs live in one folder per dataset: `configs/<dataset>/<experiment>.yaml`.

| File (in `configs/ml-1m/`) | Description |
|---|---|
| `bert4rec_baseline.yaml` | Baseline BERT4Rec configuration |
| `bert4rec_tuned.yaml` | Tuned BERT4Rec configuration |
| `jepa_baseline.yaml` | Baseline JEPA configuration |
| `jepa_tuned.yaml` | Tuned JEPA configuration |
| `jepa_causal_baseline.yaml` | Baseline JEPA causal configuration |
| `jepa_causal_tuned.yaml` | Tuned JEPA causal configuration |
| `datasets/<name>.yaml` | Per-dataset facts: `vocab_size`, `max_seq_len` |

## Configuration Contents

Configurations may specify:

- Random seed and device settings
- Nothing about which dataset: that comes from the folder the file sits in (or `--dataset`). A `dataset:` key is still accepted and must match the folder.
- Data paths, written with a `{dataset}` placeholder
- Model architecture parameters
- Training hyperparameters
- Validation and early stopping settings
- Evaluation settings

The exact available parameters are defined by each YAML file and the corresponding training or evaluation implementation.

## Datasets

`configs/datasets/<name>.yaml` holds what is true of a preprocessed dataset, e.g.:

```yaml
name: amazon_beauty        # must match the file name and data/processed/<name>/
model:
  vocab_size: 8609         # largest mapped item id + 1 (0 = PAD, 1 = MASK)
  max_seq_len: 200
```

A run is built in layers, later layers winning: experiment YAML, then the dataset YAML, then `--set key=value`, then `--name`. Because dataset values override the experiment YAML, do not put `vocab_size` in experiment configs. Use `--set model.max_seq_len=100` for a one-off change.

Outputs are not configured here. They are derived from the dataset and `output.experiment_name` (which defaults to the YAML file name):

```text
results/<dataset>/<experiment_name>/
checkpoints/<dataset>/<experiment_name>/<experiment_name>_best.pt
```

To add a dataset, copy `datasets/ml-1m.yaml` to `datasets/<name>.yaml` and edit it. To give it its own tuned experiments, copy the ones you want from `ml-1m/` into a new `configs/<name>/` folder:

```text
configs/
├── datasets/            # facts about each dataset (vocab_size, max_seq_len)
│   ├── ml-1m.yaml
│   └── amazon_beauty.yaml
├── ml-1m/               # experiments run on ml-1m
│   └── jepa_causal_tuned.yaml
└── amazon_beauty/       # experiments run on amazon_beauty
    └── jepa_causal_tuned.yaml
```

The dataset is taken from, in order: `--dataset`, a `dataset:` key in the YAML, the folder name. So `--config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty` runs the ml-1m hyperparameters on Amazon, while `--config configs/amazon_beauty/jepa_causal_tuned.yaml` uses the Amazon ones. `output.experiment_name` defaults to the file name, so both write to `.../<dataset>/jepa_causal_tuned/` without clashing.

## Running Experiments

Run training commands from the repository root.

```bash
python -m src.training.train_bert4rec --config configs/ml-1m/bert4rec_baseline.yaml
python -m src.training.train_bert4rec --config configs/ml-1m/bert4rec_tuned.yaml

python -m src.training.train_jepa --config configs/ml-1m/jepa_baseline.yaml
python -m src.training.train_jepa --config configs/ml-1m/jepa_tuned.yaml

python -m src.training.train_jepa_causal --config configs/ml-1m/jepa_causal_baseline.yaml
python -m src.training.train_jepa_causal --config configs/ml-1m/jepa_causal_tuned.yaml

# same config, different dataset
python -m src.training.train_jepa_causal --config configs/ml-1m/jepa_causal_tuned.yaml --dataset amazon_beauty
```

## Reproducibility

Treat each configuration as the recorded specification of an experiment. When changing hyperparameters, preserve the original configuration or create a separately named configuration so that previous runs remain identifiable.

A configuration alone may not fully guarantee bit-for-bit reproducibility. Record the dataset version, software environment, and hardware used alongside experimental results.