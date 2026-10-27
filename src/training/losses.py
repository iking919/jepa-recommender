"""
Training losses for one-pass (causal) JEPA, and the matching eval scoring.

``cosine``        The original JEPA loss: 1 - cos(prediction, EMA target
                  embedding). Positive pairs only, no negatives.
``infonce_ema``   Softmax over the whole catalogue of cosine similarities
                  to the EMA target embeddings (the item bank gets no
                  gradient). JEPA's EMA target is kept; negatives are added.
``infonce_tied``  Same softmax, but the item bank is the context encoder's
                  own (online) item embedding, so there is no EMA target.
                  This is a SASRec-style objective with cosine logits.

Comparing the three attributes any accuracy gap to "no negatives"
(cosine -> infonce_ema) versus "EMA target" (infonce_ema -> infonce_tied).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from src.evaluation.recommendation import (
    SCORING_EMA_COSINE,
    SCORING_TIED_COSINE,
)

FIRST_ITEM_ID = 2  # ids 0 and 1 are PAD and MASK

LOSSES = ("cosine", "infonce_ema", "infonce_tied")

_SCORING = {
    "cosine": SCORING_EMA_COSINE,
    "infonce_ema": SCORING_EMA_COSINE,
    "infonce_tied": SCORING_TIED_COSINE,
}


def scoring_for_loss(loss_name: str) -> str:
    """Return the evaluation scoring mode that matches ``loss_name``."""

    if loss_name not in _SCORING:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss_name!r}.")

    return _SCORING[loss_name]


def _cosine_loss(model, z_pred, z_target, batch) -> torch.Tensor:
    cosine = F.cosine_similarity(z_pred.float(), z_target.float(), dim=-1)
    mask = batch["mask"].float()

    return ((1.0 - cosine) * mask).sum() / mask.sum().clamp(min=1.0)


def _make_infonce(tied: bool, temperature: float):
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    def loss_fn(model, z_pred, z_target, batch) -> torch.Tensor:
        selected = batch["mask"]

        z = F.normalize(z_pred[selected].float(), dim=-1)
        labels = batch["target"][selected] - FIRST_ITEM_ID

        if tied:
            bank = model.context_encoder.item_embedding.weight
        else:
            bank = model.target_encoder.item_embedding.weight

        bank = F.normalize(bank[FIRST_ITEM_ID:].float(), dim=-1)

        logits = (z @ bank.T) / temperature

        return F.cross_entropy(logits, labels)

    return loss_fn


def build_loss(name: str, temperature: float = 0.1) -> Callable:
    """Return ``loss_fn(model, z_pred, z_target, batch) -> scalar``."""

    if name == "cosine":
        return _cosine_loss
    if name == "infonce_ema":
        return _make_infonce(tied=False, temperature=temperature)
    if name == "infonce_tied":
        return _make_infonce(tied=True, temperature=temperature)

    raise ValueError(f"loss must be one of {LOSSES}, got {name!r}.")
