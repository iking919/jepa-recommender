# Scripts

This directory contains command-line entry points for running project workflows that use the reusable modules under `src/`.

## Evaluation Scripts

The repository includes evaluation entry points for the trained recommendation models.

| Script | Purpose |
|---|---|
| `run_bert4rec_evaluation.py` | Evaluate a BERT4Rec checkpoint |
| `run_jepa_evaluation.py` | Evaluate a JEPA checkpoint |

## BERT4Rec Evaluation

Baseline:

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_baseline.yaml \
    --checkpoint checkpoints/bert4rec_baseline/bert4rec_baseline_best.pt
```

Tuned:

```bash
python -m scripts.run_bert4rec_evaluation \
    --config configs/bert4rec_tuned.yaml \
    --checkpoint checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
```

## JEPA Evaluation

Baseline:

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_baseline.yaml \
    --checkpoint checkpoints/jepa_baseline/jepa_baseline_best.pt
```

Tuned:

```bash
python -m scripts.run_jepa_evaluation \
    --config configs/jepa_tuned.yaml \
    --checkpoint checkpoints/jepa_tuned/jepa_tuned_best.pt
```

## Dataset Preparation

If dataset download or preprocessing scripts are included in this directory, consult their command-line options and use them from the repository root.

Do not assume that a preparation script exists unless it is present in the current repository.

## Execution

Run scripts using Python's module execution syntax from the repository root:

```bash
python -m scripts.<module_name> --help
```

This supports consistent package imports when scripts depend on modules under `src/`.