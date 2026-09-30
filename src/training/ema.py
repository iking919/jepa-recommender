"""
Exponential moving average utilities for the JEPA target encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """
    Update a target model's item embedding from a source model.

    The source and target modules need only expose compatible
    ``item_embedding.weight`` tensors.
    """

    def __init__(
        self,
        target_model: nn.Module,
        source_model: nn.Module,
        decay: float = 0.996,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                f"EMA decay must be in [0, 1), got {decay}."
            )

        self.target_model = target_model
        self.source_model = source_model
        self.decay = decay

        self._validate_embeddings()

    def _validate_embeddings(self) -> None:
        """Validate that source and target expose compatible embeddings."""

        if not hasattr(self.source_model, "item_embedding"):
            raise AttributeError(
                "Source model must have an item_embedding attribute."
            )

        if not hasattr(self.target_model, "item_embedding"):
            raise AttributeError(
                "Target model must have an item_embedding attribute."
            )

        source_shape = self.source_model.item_embedding.weight.shape
        target_shape = self.target_model.item_embedding.weight.shape

        if source_shape != target_shape:
            raise ValueError(
                "Source and target item embeddings must have the same shape. "
                f"Got source={source_shape}, target={target_shape}."
            )

    @torch.no_grad()
    def update(self) -> None:
        """Apply one EMA update to the target item embedding."""

        source_embedding = self.source_model.item_embedding.weight
        target_embedding = self.target_model.item_embedding.weight

        target_embedding.mul_(self.decay)
        target_embedding.add_(
            source_embedding,
            alpha=1.0 - self.decay,
        )

    def state_dict(self) -> dict:
        """Return serializable EMA configuration."""

        return {"decay": self.decay}
