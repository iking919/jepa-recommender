"""
Evaluation routines for sequential recommendation models.

Both BERT4Rec and JEPA are evaluated with the same canonical metrics:
    HR@5, HR@10, NDCG@5, NDCG@10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_metrics, compute_ranking_metrics
from src.evaluation.recommendation import build_candidate_embeddings, recommend_batch
from src.models.jepa import JEPA


@torch.no_grad()
def evaluate_jepa(
    model: JEPA,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """
    Evaluate JEPA as a next-item recommender.

    Each sample contains a context sequence and one held-out target.
    The predicted representation is compared with every valid item
    embedding, and the target's rank determines the ranking metrics.
    """

    was_training = model.training
    model.eval()

    candidate_ids, candidate_embeddings = build_candidate_embeddings(
        model=model,
        num_items=num_items,
        device=device,
    )

    max_k = max(k_values)
    all_recommendations = []
    all_targets = []

    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        actual_lens = batch["actual_len"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        recommendations, _ = recommend_batch(
            model=model,
            context=context,
            actual_lens=actual_lens,
            candidate_ids=candidate_ids,
            candidate_embeddings=candidate_embeddings,
            top_k=max_k,
        )

        all_recommendations.append(recommendations.cpu())
        all_targets.append(targets.cpu())

    if not all_recommendations:
        raise ValueError("No samples were available for JEPA evaluation.")

    recommendations = torch.cat(all_recommendations, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_ranking_metrics(
        recommendations=recommendations,
        targets=targets,
        k_values=k_values,
    )

    if was_training:
        model.train()

    return metrics


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
    Evaluate BERT4Rec on a user-level held-out dataset.

    The evaluation sequence ends with a MASK token. The model's logits at
    the MASK position are ranked over actual item IDs only.
    """

    required_columns = {"user_id", input_column, target_column}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Evaluation DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if vocab_size <= 2:
        raise ValueError(f"vocab_size must be greater than 2, got {vocab_size}.")

    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    was_training = model.training
    model.eval()

    all_metrics = []
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
            sequence = row[input_column]

            if hasattr(sequence, "tolist"):
                sequence = sequence.tolist()
            else:
                sequence = list(sequence)

            target = int(row[target_column])
            original_len = len(sequence)

            if input_requires_mask:
                eval_seq = sequence + [mask_token_id]
            else:
                eval_seq = sequence.copy()

            if len(eval_seq) > max_seq_len:
                eval_seq = eval_seq[-max_seq_len:]

            try:
                mask_position = (
                    len(eval_seq)
                    - 1
                    - eval_seq[::-1].index(mask_token_id)
                )
            except ValueError as exc:
                raise ValueError(
                    f"User {user_id} evaluation sequence does not contain "
                    f"the MASK token ({mask_token_id})."
                ) from exc

            pad_len = max_seq_len - len(eval_seq)
            eval_seq_padded = eval_seq + [pad_token_id] * pad_len

            input_ids = torch.tensor(
                [eval_seq_padded],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids)
            item_scores = logits[0, mask_position].detach().cpu().numpy()

            item_scores[pad_token_id] = -float("inf")
            item_scores[mask_token_id] = -float("inf")

            if exclude_seen:
                for seen_id in sequence:
                    seen_id = int(seen_id)
                    if 0 <= seen_id < vocab_size:
                        item_scores[seen_id] = -float("inf")

            ranked_items = sorted(
                range(2, vocab_size),
                key=lambda item_id: float(item_scores[item_id]),
                reverse=True,
            )

            user_metrics = calculate_metrics(
                ranked_items=ranked_items,
                target=target,
                top_ks=(5, 10),
            )
            all_metrics.append(user_metrics)

            if len(debug_outputs) < 5:
                target_rank = (
                    ranked_items.index(target) + 1
                    if target in ranked_items
                    else -1
                )

                debug_outputs.append(
                    {
                        "user_id": user_id,
                        "history_len": original_len,
                        "target": target,
                        "top_10": ranked_items[:10],
                        "rank": target_rank,
                        "hit_5": bool(user_metrics["HR@5"]),
                        "hit_10": bool(user_metrics["HR@10"]),
                    }
                )

    if not all_metrics:
        raise ValueError("No users were available for BERT4Rec evaluation.")

    if was_training:
        model.train()

    return {
        "HR@5": float(np.mean([m["HR@5"] for m in all_metrics])),
        "HR@10": float(np.mean([m["HR@10"] for m in all_metrics])),
        "NDCG@5": float(np.mean([m["NDCG@5"] for m in all_metrics])),
        "NDCG@10": float(np.mean([m["NDCG@10"] for m in all_metrics])),
        "debug_outputs": debug_outputs,
    }
