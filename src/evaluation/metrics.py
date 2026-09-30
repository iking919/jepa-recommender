"""
Ranking metrics for sequential recommendation.

The project uses one canonical metric naming convention for both models:
    HR@5
    HR@10
    NDCG@5
    NDCG@10
"""

from __future__ import annotations

import math

import torch


def hit_rate_at_k(
    ranked_list,
    target_item,
    k: int,
) -> float:
    """Return 1.0 when the target appears in the top-k recommendations."""

    return float(target_item in ranked_list[:k])


def ndcg_at_k(
    ranked_list,
    target_item,
    k: int,
) -> float:
    """Compute single-target NDCG@K for one ranked recommendation list."""

    if target_item not in ranked_list[:k]:
        return 0.0

    rank = ranked_list.index(target_item) + 1
    return 1.0 / math.log2(rank + 1)


def calculate_metrics(
    ranked_items,
    target,
    top_ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Calculate canonical ranking metrics for one user."""

    metrics = {}

    for k in top_ks:
        metrics[f"HR@{k}"] = hit_rate_at_k(
            ranked_items,
            target,
            k,
        )
        metrics[f"NDCG@{k}"] = ndcg_at_k(
            ranked_items,
            target,
            k,
        )

    return metrics


def compute_ranking_metrics(
    recommendations: torch.Tensor,
    targets: torch.Tensor,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Compute canonical ranking metrics for a batch of recommendations."""

    metrics = {}

    for k in k_values:
        metrics[f"HR@{k}"] = _batch_hit_rate_at_k(
            recommendations,
            targets,
            k,
        )
        metrics[f"NDCG@{k}"] = _batch_ndcg_at_k(
            recommendations,
            targets,
            k,
        )

    return metrics


def _batch_hit_rate_at_k(
    recommendations: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """Compute batch Hit Rate@K."""

    top_k = recommendations[:, :k]
    hits = top_k.eq(targets.unsqueeze(1)).any(dim=1)

    return hits.float().mean().item()


def _batch_ndcg_at_k(
    recommendations: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    """Compute batch NDCG@K for one ground-truth item per sample."""

    top_k = recommendations[:, :k]
    matches = top_k.eq(targets.unsqueeze(1))

    hit_rows, hit_positions = matches.nonzero(as_tuple=True)

    scores = torch.zeros(
        recommendations.shape[0],
        dtype=torch.float32,
        device=recommendations.device,
    )

    if hit_rows.numel() > 0:
        ranks = hit_positions.float() + 1.0
        scores[hit_rows] = 1.0 / torch.log2(ranks + 1.0)

    return scores.mean().item()
