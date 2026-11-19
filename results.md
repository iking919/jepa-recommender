# Experimental Results

## 1. Overview

This document summarizes the current experimental comparison between BERT4Rec and Causal JEPA on MovieLens-1M and Amazon Beauty.

The experiments evaluate recommendation accuracy, training time, convergence behavior, and GPU memory consumption.

**Experimental protocol:**
- Five random seeds: 42, 123, 456, 789, and 2026.
- Models are evaluated using HR@5, HR@10, NDCG@5, and NDCG@10.
- Validation metrics, best epoch, epoch time, and total runtime are reported as five-seed averages.
- GPU memory measurements are individual-run peak measurements, not five-seed averages.
- Cross-seed validation results are reported separately from held-out test evaluations.

## 2. Recommendation Performance and Training Efficiency

All metrics below represent five-seed averages.

| Dataset | Model | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Avg. Best Epoch | Avg. Epoch Time | Avg. Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MovieLens-1M | BERT4Rec | 0.1979 | 0.2929 | 0.1317 | 0.1621 | 124.0 | 1.69s | 4.22 min |
| MovieLens-1M | Causal JEPA | 0.1707 | 0.2260 | 0.1221 | 0.1399 | 29.0 | 3.61s | 2.95 min |
| Amazon Beauty | BERT4Rec | 0.0595 | 0.0868 | 0.0406 | 0.0494 | 40.6 | 11.04s | 27.59 min |
| Amazon Beauty | Causal JEPA | 0.0396 | 0.0470 | 0.0311 | 0.0335 | 94.8 | 14.59s | 33.24 min |

## 3. GPU Memory Consumption

The following values represent peak GPU memory consumption from individual reported runs. They should not be interpreted as five-seed averages.

| Dataset | Model | Peak GPU Memory | Reported Seed |
|---|---|---:|---:|
| MovieLens-1M | BERT4Rec | 1.49 GB | 42 |
| MovieLens-1M | Causal JEPA | 1.00 GB | 123 |
| Amazon Beauty | BERT4Rec | 4.83 GB | 123 |
| Amazon Beauty | Causal JEPA | 5.56 GB | 2026 |

## 4. Preliminary Findings

### 4.1 Recommendation Accuracy

BERT4Rec achieves higher average validation performance across all four metrics on both datasets.

The difference in NDCG@10 is particularly notable:

- **MovieLens-1M:** Causal JEPA is approximately 13.7% below BERT4Rec.
- **Amazon Beauty:** Causal JEPA is approximately 32.2% below BERT4Rec.

This suggests that BERT4Rec currently has a consistent recommendation-accuracy advantage, although the magnitude of that advantage varies across datasets.

### 4.2 Training Efficiency

The computational comparison varies by dataset.

**MovieLens-1M:**
- Causal JEPA has a shorter average total runtime (2.95 vs. 4.22 minutes).
- BERT4Rec has a shorter average epoch time (1.69 vs. 3.61 seconds).
- Causal JEPA reaches its best validation performance substantially earlier.

**Amazon Beauty:**
- BERT4Rec has a shorter average epoch time (11.04 vs. 14.59 seconds).
- BERT4Rec also has a shorter average total runtime (27.59 vs. 33.24 minutes).
- Causal JEPA reaches its best validation performance later.

### 4.3 GPU Memory

On MovieLens-1M, the reported Causal JEPA run uses less peak GPU memory than BERT4Rec (1.00 vs. 1.49 GB).

On Amazon Beauty, the reported Causal JEPA run uses more peak GPU memory than BERT4Rec (5.56 vs. 4.83 GB).

This suggests that the memory advantage is not consistent across datasets. Additional per-seed measurements would be useful before drawing broader conclusions.

## 5. Research Interpretation

The current results suggest that model performance and computational efficiency depend on the dataset being evaluated.

BERT4Rec consistently achieves stronger recommendation accuracy, while Causal JEPA's computational advantages are less consistent across datasets.

One potential explanation is that differences in user-history length, interaction sparsity, and item distributions affect how effectively each architecture learns sequential representations.

A particularly interesting next step is to investigate whether Causal JEPA becomes more competitive when users have substantially longer interaction histories.

## 6. Planned Experiments

The next dataset under consideration is the Music Listening Histories Dataset (MLHD), potentially using MLHD+.

The goal is to investigate whether longer user interaction sequences change the relative performance of BERT4Rec and Causal JEPA.

Proposed analyses include:

- Comparing performance at sequence lengths of 200, 500, and 1,000.
- Measuring recommendation accuracy across user-history-length buckets.
- Comparing training time and peak GPU memory.
- Examining the effects of interaction sparsity and repeated listening events.
- Maintaining consistent evaluation protocols across models.

The broader research question is:

**How do user-history characteristics, particularly sequence length and interaction sparsity, influence the recommendation accuracy and computational efficiency of BERT4Rec and Causal JEPA?**

## 7. Reporting Notes

- Validation results are five-seed averages, not held-out test results.
- GPU memory measurements are individual-run observations.
- Held-out test results should be documented separately after model selection.
- Dataset-specific preprocessing and configuration details should accompany each experiment.
- Results should only be compared when the underlying experiment configurations and evaluation protocols are verified.