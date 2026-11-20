"""Build models from the ``model`` section of an experiment config."""

from __future__ import annotations

from src.data.tokens import PAD_TOKEN_ID
from src.models.bert4rec import BERT4Rec
from src.models.jepa import JEPA


def build_bert4rec(model_config: dict) -> BERT4Rec:
    """Create a BERT4Rec model from a config's ``model`` section."""

    return BERT4Rec(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
    )


def build_jepa(model_config: dict, causal: bool) -> JEPA:
    """Create a JEPA model from a config's ``model`` section.

    ``causal`` is an explicit argument rather than a config lookup: the
    standard trainer always builds a non-causal model, the causal trainer
    always builds a causal one, and evaluation reads ``model.causal``.
    """

    return JEPA(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config.get("pad_token_id", PAD_TOKEN_ID),
        causal=causal,
    )
