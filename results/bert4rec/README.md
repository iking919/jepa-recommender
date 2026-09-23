# BERT4Rec Results

Final BERT4Rec results on MovieLens-1M.

The tuned model was selected using validation NDCG@10 and evaluated once
on the held-out test set after tuning was complete.

## Tuned Test Results

| Metric | Score |
|---|---:|
| HR@5 | 0.1416 |
| HR@10 | 0.2182 |
| NDCG@5 | 0.0893 |
| NDCG@10 | 0.1138 |

Best checkpoint: epoch 50

## Evaluation Protocol

- Dataset: MovieLens-1M
- Users: 6,040
- Items: 3,416
- Maximum sequence length: 200
- Mask probability: 0.15
- Chronological train/validation/test split
- Seen-item filtering enabled
- Candidate items: all valid movie IDs
- Model selected using validation NDCG@10
- Final metrics reported on the held-out test set