"""Chronological train/validation/test splitting."""

from __future__ import annotations

import pandas as pd


def leave_last_two_out(user_sequences: pd.DataFrame) -> pd.DataFrame:
    """Split each user's chronological sequence, leaving the last two out.

    Users with fewer than three interactions are dropped. For the rest:

        train      = every interaction except the final two
        validation = the second-to-last interaction
        test       = the final interaction

    Args:
        user_sequences: Frame with ``user_id`` and ``sequence`` columns.

    Returns:
        Frame with ``user_id``, ``train_sequence``, ``validation_item`` and
        ``test_item`` columns.
    """

    split_data = []

    for _, row in user_sequences.iterrows():
        sequence = row["sequence"]

        if len(sequence) < 3:
            continue

        split_data.append(
            {
                "user_id": row["user_id"],
                "train_sequence": sequence[:-2],
                "validation_item": sequence[-2],
                "test_item": sequence[-1],
            }
        )

    return pd.DataFrame(
        split_data,
        columns=["user_id", "train_sequence", "validation_item", "test_item"],
    )
