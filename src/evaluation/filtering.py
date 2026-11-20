"""Seen-item filtering shared by every evaluation routine."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.tokens import MASK_TOKEN_ID, PAD_TOKEN_ID


def filter_ranked_recommendations(
    ranked_candidates: torch.Tensor,
    histories: Sequence[Sequence[int]] | torch.Tensor,
    *,
    k: int,
    pad_token_id: int = PAD_TOKEN_ID,
    mask_token_id: int | None = MASK_TOKEN_ID,
    history_lengths: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Remove previously seen items and special tokens from a ranking.

    ``ranked_candidates`` must contain the complete candidate ranking,
    sorted from highest to lowest score, rather than only the original
    top-k.

    Returns:
        Tensor of shape ``[batch_size, k]``. Missing recommendations are
        represented by -1.
    """
    if ranked_candidates.ndim != 2:
        raise ValueError("ranked_candidates must be a 2D tensor.")
    if k <= 0:
        raise ValueError("k must be positive.")

    batch_size = ranked_candidates.shape[0]

    if isinstance(histories, torch.Tensor):
        if histories.ndim != 2 or histories.shape[0] != batch_size:
            raise ValueError("Tensor histories must have shape [batch, seq_len].")
        history_rows = histories.detach().cpu().tolist()
    else:
        history_rows = list(histories)

    if len(history_rows) != batch_size:
        raise ValueError("Number of histories must match ranking batch size.")

    if history_lengths is not None:
        if isinstance(history_lengths, torch.Tensor):
            lengths = history_lengths.detach().cpu().tolist()
        else:
            lengths = list(history_lengths)

        if len(lengths) != batch_size:
            raise ValueError("history_lengths must match ranking batch size.")
    else:
        lengths = None

    filtered_rows: list[list[int]] = []

    for row_idx, history in enumerate(history_rows):
        if lengths is not None:
            valid_length = max(0, min(int(lengths[row_idx]), len(history)))
            history = history[:valid_length]

        seen = {
            int(item)
            for item in history
            if int(item) != pad_token_id
            and (mask_token_id is None or int(item) != mask_token_id)
        }

        selected: list[int] = []

        for candidate in ranked_candidates[row_idx].detach().cpu().tolist():
            candidate = int(candidate)

            if candidate == pad_token_id:
                continue
            if mask_token_id is not None and candidate == mask_token_id:
                continue
            if candidate in seen:
                continue

            selected.append(candidate)

            if len(selected) == k:
                break

        selected.extend([-1] * (k - len(selected)))
        filtered_rows.append(selected)

    return torch.tensor(
        filtered_rows,
        dtype=torch.long,
        device=ranked_candidates.device,
    )
