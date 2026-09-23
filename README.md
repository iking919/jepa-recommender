# JEPA for Sequential Recommendation

Research project investigating Joint-Embedding Predictive Architectures (JEPA) for sequential recommendation and comparing a JEPA-based approach against BERT4Rec.

## Overview

This project investigates whether JEPA-style predictive learning can be used for sequential recommendation without directly predicting item identities at every training step.

The primary experimental comparison uses:

* **JEPA-based sequential recommendation**
* **BERT4Rec**
* **MovieLens-1M**

Models are evaluated using:

* Hit Rate @ 5 (HR@5)
* Hit Rate @ 10 (HR@10)
* NDCG @ 5 (NDCG@5)
* NDCG @ 10 (NDCG@10)

## Dataset

The primary dataset is **MovieLens-1M**.

Raw dataset statistics:

* 1,000,209 ratings
* 6,040 users
* 3,706 movies

The experimental preprocessing pipeline:

* sorts interactions chronologically by user
* filters movies with fewer than 5 interactions
* constructs chronological user interaction sequences
* limits sequences to a maximum length of 200
* maps movie IDs to a contiguous model vocabulary
* uses a chronological train/validation/test split

After preprocessing:

* 6,040 users
* 3,416 movies
* 648,759 training interactions
* 3,418 vocabulary entries including special tokens

The same processed dataset and evaluation protocol are used for the BERT4Rec and JEPA experiments to support a controlled comparison.

## Models

### BERT4Rec

BERT4Rec is used as the sequential recommendation baseline.

The final tuned configuration uses:

* Hidden dimension: 128
* Transformer layers: 2
* Attention heads: 4
* Feed-forward dimension: 512
* Dropout: 0.1
* Masking probability: 0.15
* Learning rate: 0.001
* Weight decay: 0.01
* Training epochs: 50
* Cosine annealing learning-rate schedule

The best checkpoint is selected using validation NDCG@10 and evaluated on the held-out test set.

### JEPA

The JEPA model uses a context encoder, target encoder, and predictor to learn representations of future interactions.

The target encoder is updated using an exponential moving average (EMA) of the context encoder and is not used during inference.

JEPA experiments use the same MovieLens-1M preprocessing and recommendation evaluation protocol as BERT4Rec.

Final JEPA configuration and results will be documented after the JEPA experiments are completed.

## Results

### Final Held-Out Test Results

| Model    |   HR@5 |  HR@10 | NDCG@5 | NDCG@10 |
| -------- | -----: | -----: | -----: | ------: |
| BERT4Rec | 0.1416 | 0.2182 | 0.0893 |  0.1138 |
| JEPA     |      — |      — |      — |       — |

The BERT4Rec results are frozen and will serve as the reference baseline for the JEPA experiments.

Final comparative analysis will be added after the JEPA experiments are complete.

## Reproducibility

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

Download and preprocess MovieLens-1M using:

```bash
python scripts/download_ml1m.py
python scripts/preprocess_ml1m.py
```

### Configuration

Experiment configurations are stored in `configs/`.

Current configurations include:

* `bert4rec_baseline.yaml`
* `bert4rec_tuned.yaml`
* `jepa_baseline.yaml`
* `jepa_tuned.yaml`

### Training

BERT4Rec can be trained using:

```bash
python -m src.training.train_bert4rec --config configs/bert4rec_tuned.yaml
```

JEPA training will use the corresponding modular training pipeline.

### Evaluation

Evaluation is performed using:

```bash
python -m src.evaluation.run_evaluation --config configs/bert4rec_tuned.yaml --checkpoint checkpoints/bert4rec_tuned/bert4rec_tuned_best.pt
```

The evaluation pipeline uses the held-out test set and reports HR@5, HR@10, NDCG@5, and NDCG@10.

## Project Structure

```text
jepa-recommender/
├── configs/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── scripts/
├── results/
└── checkpoints/
```

## Research Status

The BERT4Rec baseline and tuning experiments have been completed and frozen.

The next stage is to implement and evaluate the JEPA-based sequential recommendation model under the same experimental protocol. Final model comparisons, analysis, and conclusions will be added after both model families have been evaluated.
