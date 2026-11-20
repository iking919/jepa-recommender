# Source Code

Reusable implementation of the sequential recommendation experiments. The
command-line entry points in [`scripts/`](../scripts/README.md) and the
trainers below are thin layers over these packages.

```text
src/
├── data/          datasets, loaders, splits and preprocessing
├── models/        BERT4Rec, JEPA and the shared Transformer encoder
├── training/      training pipelines and shared training helpers
├── evaluation/    ranking metrics, candidate filtering, evaluation
└── utils/         configuration, paths, logging, checkpoints, seeding
```

## `data/`

| Module | Purpose |
|---|---|
| `tokens.py` | Special token IDs (`PAD`, `MASK`, first item ID) |
| `preprocessing.py` | Dataset-agnostic preprocessing steps shared by both datasets |
| `splits.py` | Leave-last-two-out chronological split |
| `datasets.py` | PyTorch datasets for BERT4Rec and JEPA |
| `gpu_loader.py` | GPU-resident batch loaders for JEPA |
| `sequence_windows.py` | Window construction for one-pass causal training |
| `download.py` | HTTP download helpers used by the download scripts |

## `models/`

| Module | Purpose |
|---|---|
| `transformer.py` | Transformer encoder shared by the models |
| `bert4rec.py` | BERT4Rec |
| `jepa.py` | JEPA (context encoder, target encoder, predictor); causal or not |
| `factory.py` | `build_bert4rec` / `build_jepa`: models from a config's `model` section |

## `training/`

| Module | Purpose |
|---|---|
| `train_bert4rec.py` | BERT4Rec trainer |
| `train_jepa_causal.py` | Causal one-pass JEPA trainer (primary model) |
| `train_jepa.py` | Standard JEPA trainer (one prefix per target) |
| `common.py` | Device and GPU helpers, mixed precision, output paths, run-summary sections |
| `ema.py` | Exponential moving average of the JEPA target encoder |

Every trainer exposes a function (`train_bert4rec`, `train`) and a command-line
entry point.

## `evaluation/`

| Module | Purpose |
|---|---|
| `metrics.py` | HR@k and NDCG@k |
| `filtering.py` | Removes previously seen items from ranked candidates |
| `recommendation.py` | Candidate embeddings and batched JEPA recommendation |
| `evaluate.py` | `evaluate_jepa` and `evaluate_bert4rec` |

Both models share the same filtering protocol, so their metrics are directly
comparable.

## `utils/`

| Module | Purpose |
|---|---|
| `config.py` | Layered config loading (`--config`, `--dataset`, `--set`, `--name`) |
| `overrides.py` | Parsing of `--set` and `--name` |
| `paths.py` | Single source of truth for data, results and checkpoint locations |
| `checkpointing.py` | Saving and loading checkpoints |
| `schedules.py` | Learning-rate schedule and model-selection helpers |
| `serialization.py` | JSON helpers for the search and reporting scripts |
| `reproducibility.py` | Random seeding |
| `logging_utils.py` | Console logging setup |

## Design principles

- Model architecture is separate from training logic.
- Experiment parameters live in YAML files, not in code.
- Paths are built in one place (`utils/paths.py`).
- Validation and held-out test evaluation are separate steps.
- Checkpoints are kept apart from result artifacts.

## Running a trainer

From the repository root:

```bash
python -m src.training.train_bert4rec     --config configs/ml-1m/bert4rec_tuned.yaml
python -m src.training.train_jepa_causal  --config configs/ml-1m/jepa_causal_tuned.yaml
python -m src.training.train_jepa         --config configs/ml-1m/jepa_tuned.yaml
```
