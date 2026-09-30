"""
JEPA model components for sequential recommendation.

The context encoder and predictor are trained by backpropagation. The
target encoder is gradient-free and its item embeddings are updated from
the context encoder using exponential moving average (EMA).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """Encode an item sequence into a fixed-dimensional representation."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.item_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        actual_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Return the hidden representation of the final valid item."""

        batch_size, seq_len = x.shape
        padding_mask = x.eq(self.pad_token_id)

        item_embeddings = self.item_embedding(x)

        positions = torch.arange(
            seq_len,
            dtype=torch.long,
            device=x.device,
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.position_embedding(positions)

        transformer_input = self.dropout(
            item_embeddings + position_embeddings
        )

        encoded = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=padding_mask,
        )

        last_indices = (actual_lens - 1).clamp(min=0)

        batch_indices = torch.arange(
            batch_size,
            device=x.device,
        )

        return encoded[batch_indices, last_indices]


class TargetEncoder(nn.Module):
    """Gradient-free item encoder updated through EMA."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.item_embedding = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=pad_token_id,
        )

        for parameter in self.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Encode target item IDs."""

        return self.item_embedding(target_ids)


class Predictor(nn.Module):
    """Predict target representations from context representations."""

    def __init__(
        self,
        hidden_dim: int = 128,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, hidden_dim),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Predict target representations."""

        return self.net(z_context)


class JEPA(nn.Module):
    """Complete JEPA sequential recommendation model."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feed_forward_dim: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.context_encoder = ContextEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )

        self.target_encoder = TargetEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            pad_token_id=pad_token_id,
        )

        self.predictor = Predictor(
            hidden_dim=hidden_dim,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )

    def forward(
        self,
        context_sequence: torch.Tensor,
        actual_lens: torch.Tensor,
        target_item: torch.Tensor | None = None,
    ):
        """
        Forward pass.

        When ``target_item`` is provided, return ``(prediction, target)``.
        Otherwise, return only the predicted representation.
        """

        z_context = self.context_encoder(
            context_sequence,
            actual_lens,
        )
        z_pred = self.predictor(z_context)

        if target_item is not None:
            z_target = self.target_encoder(target_item)
            return z_pred, z_target

        return z_pred
