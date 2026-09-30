"""
Chronological train/validation/test splitting utilities.
"""

from __future__ import annotations

import pandas as pd


def leave_last_two_out(user_sequences: pd.DataFrame) -> pd.DataFrame:
    """
    Perform a chronological leave-last-two-out split.

    For each user with at least three interactions:
        train      = all interactions except the final two
        validation = second-to-last interaction
        test       = final interaction
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

    return pd.DataFrame(split_data)
