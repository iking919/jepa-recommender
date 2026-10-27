"""
Recommendation utilities for JEPA evaluation.

Two candidate-scoring modes are supported, selected by ``model.scoring``
(a plain string attribute; defaults to ``"ema_cosine"``):

    ema_cosine   cosine similarity against the EMA target encoder's item
                 embeddings. This is the original behaviour.
    tied_cosine  cosine similarity against the context encoder's own item
                 embeddings (used by the ``infonce_tied`` ablation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.jepa import JEPA

SCORING_EMA_COSINE = "ema_cosine"
SCORING_TIED_COSINE = "tied_cosine"


def get_scoring_mode(model: JEPA) -> str:
    """Return the candidate-scoring mode of ``model``."""

    mode = getattr(model, "scoring", SCORING_EMA_COSINE)

    if mode not in (SCORING_EMA_COSINE, SCORING_TIED_COSINE):
        raise ValueError(f"Unknown scoring mode: {mode!r}.")

    return mode


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

    if get_scoring_mode(model) == SCORING_TIED_COSINE:
        candidate_embeddings = model.context_encoder.item_embedding(
            candidate_ids
        )
    else:
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
def score_candidates(
    model: JEPA,
    context: torch.Tensor,
    actual_lens: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return the (batch, num_candidates) cosine-score matrix."""

    z_pred = model(
        context_sequence=context,
        actual_lens=actual_lens,
    )
    z_pred = F.normalize(z_pred, p=2, dim=-1)

    return z_pred @ candidate_embeddings.T


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

    scores = score_candidates(
        model,
        context,
        actual_lens,
        candidate_embeddings,
    )

    top_scores, top_indices = torch.topk(
        scores,
        k=min(top_k, candidate_embeddings.shape[0]),
        dim=1,
    )

    return candidate_ids[top_indices], top_scores
