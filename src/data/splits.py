"""
Train/validation/test splitting utilities for sequential recommendation.
"""

import pandas as pd


def leave_last_two_out(user_sequences):
    """
    Perform a chronological leave-last-two-out split.

    For each user:

        train = all interactions except the final two
        validation = second-to-last interaction
        test = final interaction

    Users must have at least three interactions.
    """

    split_data = []

    for _, row in user_sequences.iterrows():
        user_id = row["user_id"]
        sequence = row["sequence"]

        if len(sequence) < 3:
            continue

        split_data.append(
            {
                "user_id": user_id,
                "train_sequence": sequence[:-2],
                "validation_item": sequence[-2],
                "test_item": sequence[-1],
            }
        )

    return pd.DataFrame(split_data)