"""
predictor.py

Predictor network used by the JEPA model.
"""

import torch
import torch.nn as nn

from .transformer import TransformerEncoder


class JEPAPredictor(nn.Module):
    """
    Transformer-based predictor for JEPA.

    The predictor receives contextual representations and attempts to
    predict the representation of the target item.
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

        self.transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        context_embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings:
                Tensor of shape (batch_size, seq_len, hidden_dim).

            padding_mask:
                Optional boolean tensor indicating padded positions.

        Returns:
            Predicted representations with shape
            (batch_size, seq_len, hidden_dim).
        """

        return self.transformer(
            context_embeddings,
            padding_mask=padding_mask,
        )