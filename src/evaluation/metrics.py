import numpy as np
import math

def hit_rate_at_k(ranked_list, target_item, k):
    return 1 if target_item in ranked_list[:k] else 0

def ndcg_at_k(ranked_list, target_item, k):
    if target_item not in ranked_list[:k]:
        return 0.0
    rank = ranked_list.index(target_item) + 1
    return 1.0 / math.log2(rank + 1)

def calculate_metrics(ranked_items, target, top_ks=[5, 10]):
    metrics = {}
    for k in top_ks:
        metrics[f'hr_{k}'] = hit_rate_at_k(ranked_items, target, k)
        metrics[f'ndcg_{k}'] = ndcg_at_k(ranked_items, target, k)
    return metrics