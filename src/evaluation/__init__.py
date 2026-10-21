"""
Evaluation and recommendation utilities for the JEPA Recommender project.
"""

from .metrics import (
    hit_rate_at_k,
    ndcg_at_k,
)

__all__ = [
    "hit_rate_at_k",
    "ndcg_at_k",
]