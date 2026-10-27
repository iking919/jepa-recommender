"""
Recommendation utilities for JEPA evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.jepa import JEPA


@torch.no_grad()
def build_candidate_embeddings(
    model: JEPA,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build normalized embeddings for all valid recommendation items.

    IDs 0 and 1 are reserved for PAD and MASK.
    """

    candidate_ids = torch.arange(
        2,
        num_items,
        dtype=torch.long,
        device=device,
    )

    candidate_embeddings = model.target_encoder.item_embedding(
        candidate_ids
    )
    candidate_embeddings = F.normalize(
        candidate_embeddings,
        p=2,
        dim=-1,
    )

    return candidate_ids, candidate_embeddings


@torch.no_grad()
def recommend_batch(
    model: JEPA,
    context: torch.Tensor,
    actual_lens: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate top-k JEPA recommendations for a batch of contexts."""

    z_pred = model(
        context_sequence=context,
        actual_lens=actual_lens,
    )
    z_pred = F.normalize(z_pred, p=2, dim=-1)

    scores = z_pred @ candidate_embeddings.T

    top_scores, top_indices = torch.topk(
        scores,
        k=min(top_k, candidate_embeddings.shape[0]),
        dim=1,
    )

    return candidate_ids[top_indices], top_scores
