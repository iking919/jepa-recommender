"""
BERT4Rec sequential recommendation model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerEncoder


class BERT4Rec(nn.Module):
    """
    BERT4Rec-style bidirectional Transformer recommender.

    Item and positional embeddings are passed through a shared Transformer
    encoder, and each hidden state is projected into the item vocabulary.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
        dropout: float,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.item_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

        self.dropout = nn.Dropout(dropout)

        self.transformer_encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
        )

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize multi-dimensional parameters with Xavier uniform."""

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape ``(batch_size, seq_len)``.

        Returns:
            Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds "
                f"max_seq_len ({self.max_seq_len})."
            )

        padding_mask = input_ids.eq(self.pad_token_id)

        item_embeddings = self.item_embedding(input_ids)

        positions = torch.arange(
            seq_len,
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.position_embedding(positions)

        x = self.dropout(item_embeddings + position_embeddings)

        encoded = self.transformer_encoder(
            x,
            padding_mask=padding_mask,
        )

        return self.output_projection(encoded)
