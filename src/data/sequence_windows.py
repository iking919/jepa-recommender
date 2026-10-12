"""
Window construction for one-pass (causal) JEPA training.

The existing JEPA training file holds one (context, target) pair per
interaction, so each user's history is re-encoded once per target. Here the
same data is rebuilt into whole sequences and cut into fixed-size windows.
A causal encoder then produces the prediction for *every* position in a
single forward pass.

Every target in the original pair file is supervised exactly once.

This module uses only numpy/pandas so it can be tested without PyTorch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruct_train_sequences(
    pairs: pd.DataFrame,
    max_seq_len: int = 200,
) -> list[list[int]]:
    """
    Rebuild each user's full training sequence from the context-target pairs.

    ``generate_jepa_datasets`` emits, for a training sequence ``s``, one row
    per index ``i >= 1`` with ``context = s[:i][-max_seq_len:]`` and
    ``target = s[i]``. So ``s = first_context + all targets, in order``.

    Raises ``ValueError`` if a user's rows do not follow that structure
    (for example if the file was shuffled), rather than training on a
    silently wrong sequence.
    """

    required = {"user_id", "context_sequence", "target_item"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    sequences: list[list[int]] = []

    for user_id, group in pairs.groupby("user_id", sort=False):
        first_context = [int(x) for x in group["context_sequence"].iloc[0]]
        targets = [int(x) for x in group["target_item"]]

        if len(first_context) != 1:
            raise ValueError(
                f"User {user_id}: the first pair should have a 1-item "
                f"context, got {len(first_context)}. Is the file shuffled?"
            )

        sequence = first_context + targets
        length = len(sequence)

        # The final pair predicts sequence[-1] from the preceding items.
        expected = sequence[max(0, length - 1 - max_seq_len): length - 1]
        actual = [int(x) for x in group["context_sequence"].iloc[-1]]

        if actual != expected:
            raise ValueError(
                f"User {user_id}: reconstructed sequence does not match "
                "the stored final context."
            )

        sequences.append(sequence)

    return sequences


def build_windows(
    sequences: list[list[int]],
    max_seq_len: int = 200,
    stride: int | None = None,
    pad_token: int = 0,
) -> dict[str, np.ndarray]:
    """
    Cut sequences into windows for causal training.

    For a window starting at ``a``:
        inputs[j]  = s[a + j]
        targets[j] = s[a + j + 1]        (the item that follows)
        mask[j]    = True if targets[j] should contribute to the loss

    Windows start every ``stride`` items. The first window of a sequence
    supervises all of its targets. Later windows supervise only positions
    ``j >= max_seq_len - stride``, i.e. targets the previous window did not
    already cover, and each of those still sees at least
    ``max_seq_len - stride + 1`` preceding items. Every target index
    ``1..len(s)-1`` is therefore covered exactly once.

    Returns int64 arrays ``inputs``, ``targets`` of shape (N, max_seq_len),
    a bool array ``mask`` of the same shape, and ``user_index`` of shape (N,).
    """

    if stride is None:
        stride = max_seq_len // 2

    if not 1 <= stride <= max_seq_len:
        raise ValueError(
            f"stride must be in [1, {max_seq_len}], got {stride}."
        )

    first_supervised = max_seq_len - stride

    inputs, targets, masks, owners = [], [], [], []

    for user, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.int64)
        length = len(sequence)

        if length < 2:
            continue

        start = 0
        is_first = True

        while True:
            window_in = sequence[start: start + max_seq_len]
            window_tg = sequence[start + 1: start + 1 + max_seq_len]

            row_in = np.full(max_seq_len, pad_token, dtype=np.int64)
            row_tg = np.full(max_seq_len, pad_token, dtype=np.int64)
            row_mask = np.zeros(max_seq_len, dtype=bool)

            row_in[: len(window_in)] = window_in
            row_tg[: len(window_tg)] = window_tg
            row_mask[(0 if is_first else first_supervised): len(window_tg)] = True

            inputs.append(row_in)
            targets.append(row_tg)
            masks.append(row_mask)
            owners.append(user)

            # This window reaches target index start + max_seq_len.
            if start + max_seq_len >= length - 1:
                break

            start += stride
            is_first = False

    if not inputs:
        raise ValueError("No sequence was long enough to build a window.")

    return {
        "inputs": np.stack(inputs),
        "targets": np.stack(targets),
        "mask": np.stack(masks),
        "user_index": np.asarray(owners, dtype=np.int64),
    }
