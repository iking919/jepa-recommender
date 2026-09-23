"""
evaluate.py

Evaluation utilities for sequential recommendation models.

This module provides evaluation for BERT4Rec-style models using
Hit Rate (HR) and Normalized Discounted Cumulative Gain (NDCG)
at K.

The evaluator supports both validation and test datasets:

Validation:
    user_id
    train_sequence
    validation_target

Test:
    user_id
    test_input
    test_item
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import calculate_metrics


def evaluate_bert4rec(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    max_seq_len: int,
    vocab_size: int,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
    target_column: str = "validation_target",
    input_column: str = "train_sequence",
    input_requires_mask: bool = True,
    limit_users: int | None = None,
    exclude_seen: bool = True,
) -> dict:
    """
    Evaluate a BERT4Rec model on a user-level evaluation dataset.

    Args:
        model:
            Trained BERT4Rec model.

        df:
            Evaluation DataFrame.

        device:
            Device on which the model is evaluated.

        max_seq_len:
            Maximum sequence length used by the model.

        vocab_size:
            Size of the item vocabulary, including PAD and MASK.

        pad_token_id:
            Integer ID used for padding.

        mask_token_id:
            Integer ID used for the MASK token.

        target_column:
            DataFrame column containing the held-out target item.

        input_column:
            DataFrame column containing the input sequence.

        input_requires_mask:
            If True, append the MASK token to the input sequence.
            If False, the input sequence is assumed to already contain
            the MASK token.

        limit_users:
            Optional maximum number of users to evaluate.

        exclude_seen:
            If True, items appearing in the input history are excluded
            from the recommendation candidates.

    Returns:
        Dictionary containing HR@5, HR@10, NDCG@5, NDCG@10,
        and debug information for the first five users.
    """

    required_columns = {
        "user_id",
        input_column,
        target_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Evaluation DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if vocab_size <= 2:
        raise ValueError(
            f"vocab_size must be greater than 2, got {vocab_size}."
        )

    if max_seq_len <= 0:
        raise ValueError(
            f"max_seq_len must be positive, got {max_seq_len}."
        )

    model.eval()

    all_hr_5 = []
    all_hr_10 = []
    all_ndcg_5 = []
    all_ndcg_10 = []

    debug_outputs = []

    if limit_users is not None:
        if limit_users <= 0:
            raise ValueError(
                f"limit_users must be positive, got {limit_users}."
            )

        eval_users = df.iloc[:limit_users]
    else:
        eval_users = df

    with torch.no_grad():
        for _, row in eval_users.iterrows():

            user_id = row["user_id"]

            # Load the evaluation input sequence.
            sequence = row[input_column]

            if hasattr(sequence, "tolist"):
                sequence = sequence.tolist()
            else:
                sequence = list(sequence)

            target = int(row[target_column])
            original_len = len(sequence)

            # Validation data contains the history without MASK.
            # Test data contains the prepared test input.
            if input_requires_mask:
                eval_seq = sequence + [mask_token_id]
            else:
                eval_seq = sequence.copy()

            # Truncate while keeping the most recent interactions.
            if len(eval_seq) > max_seq_len:
                eval_seq = eval_seq[-max_seq_len:]

            # Find the MASK position.
            try:
                mask_position = len(eval_seq) - 1 - eval_seq[::-1].index(
                    mask_token_id
                )
            except ValueError as exc:
                raise ValueError(
                    f"User {user_id} evaluation sequence does not contain "
                    f"the MASK token ({mask_token_id})."
                ) from exc

            # Pad on the right.
            pad_len = max_seq_len - len(eval_seq)

            eval_seq_padded = (
                eval_seq
                + [pad_token_id] * pad_len
            )

            input_ids_tensor = torch.tensor(
                [eval_seq_padded],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids_tensor)

            # Extract predictions at the MASK position.
            mask_logits = (
                logits[0, mask_position]
                .detach()
                .cpu()
                .numpy()
            )

            # Exclude special tokens.
            mask_logits[pad_token_id] = -float("inf")
            mask_logits[mask_token_id] = -float("inf")

            # Exclude items already seen in the input history.
            if exclude_seen:
                for seen_id in sequence:
                    seen_id = int(seen_id)

                    if 0 <= seen_id < vocab_size:
                        mask_logits[seen_id] = -float("inf")

            # Candidate universe:
            #   0 = PAD
            #   1 = MASK
            #   2 ... vocab_size-1 = actual items
            item_scores = [
                (
                    item_id,
                    float(mask_logits[item_id]),
                )
                for item_id in range(2, vocab_size)
            ]

            item_scores.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            ranked_items = [
                item_id
                for item_id, _ in item_scores
            ]

            metrics = calculate_metrics(
                ranked_items,
                target,
                top_ks=[5, 10],
            )

            all_hr_5.append(metrics["hr_5"])
            all_hr_10.append(metrics["hr_10"])
            all_ndcg_5.append(metrics["ndcg_5"])
            all_ndcg_10.append(metrics["ndcg_10"])

            # Store first five users for debugging.
            if len(debug_outputs) < 5:

                if target in ranked_items:
                    target_rank = (
                        ranked_items.index(target) + 1
                    )
                else:
                    target_rank = -1

                debug_outputs.append(
                    {
                        "user_id": user_id,
                        "history_len": original_len,
                        "target": target,
                        "top_10": ranked_items[:10],
                        "rank": target_rank,
                        "hit_5": bool(metrics["hr_5"]),
                        "hit_10": bool(metrics["hr_10"]),
                    }
                )

    if not all_hr_5:
        raise ValueError(
            "No users were available for evaluation."
        )

    return {
        "hr_5": float(np.mean(all_hr_5)),
        "hr_10": float(np.mean(all_hr_10)),
        "ndcg_5": float(np.mean(all_ndcg_5)),
        "ndcg_10": float(np.mean(all_ndcg_10)),
        "debug_outputs": debug_outputs,
    }