# JEPA for Sequential Recommendation

Research project investigating Joint-Embedding Predictive Architectures
(JEPA) for sequential recommendation and comparing them against BERT4Rec.

## Overview

This project investigates whether JEPA-style predictive learning can be
used for sequential recommendation without directly predicting item
identities at every training step.

The primary experiments compare:

- JEPA-based sequential recommendation
- BERT4Rec
- MovieLens-1M

Models are evaluated using:

- Hit Rate @ 5
- Hit Rate @ 10
- NDCG @ 5
- NDCG @ 10

## Dataset

MovieLens-1M

- 1,000,209 ratings
- 6,040 users
- 3,706 movies

The experimental preprocessing applies the following filtering and
sequence construction procedures:

...

## Models

### BERT4Rec

...

### JEPA

...

## Results

| Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|------|------|-------|--------|---------|
| BERT4Rec | ... | ... | ... | ... |
| JEPA | ... | ... | ... | ... |

## Reproducibility

### Installation

```bash
pip install -r requirements.txt