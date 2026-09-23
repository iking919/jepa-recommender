"""
transformer.py

Reusable Transformer encoder components for the JEPA Recommender project.
"""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Reusable Transformer encoder.

    This module wraps PyTorch's TransformerEncoder and provides a common
    interface for BERT4Rec and JEPA.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
    ):
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
        """
        Args:
            x:
                Tensor of shape (batch_size, seq_len, hidden_dim).

            padding_mask:
                Boolean tensor of shape (batch_size, seq_len).
                True indicates a padded position.

        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim).
        """

        return self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )