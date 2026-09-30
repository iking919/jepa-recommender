"""
Shared Transformer encoder used by sequential recommendation models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Thin wrapper around PyTorch's batch-first Transformer encoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of sequences."""

        return self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )
