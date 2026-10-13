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
    batch_size: int = 256,
) -> dict:
    """Batched BERT4Rec evaluation; preserves the user-level ranking metrics."""
    required_columns = {"user_id", input_column, target_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Evaluation DataFrame is missing required columns: {sorted(missing_columns)}")
    if vocab_size <= 2 or max_seq_len <= 0:
        raise ValueError("vocab_size must be > 2 and max_seq_len must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if limit_users is not None:
        if limit_users <= 0:
            raise ValueError("limit_users must be positive.")
        eval_users = df.iloc[:limit_users]
    else:
        eval_users = df

    # Prepare fixed-width batches on CPU, then transfer once per batch.
    prepared = []
    for _, row in eval_users.iterrows():
        sequence = row[input_column]
        sequence = sequence.tolist() if hasattr(sequence, "tolist") else list(sequence)
        sequence = [int(x) for x in sequence]
        target = int(row[target_column])
        eval_seq = sequence + [mask_token_id] if input_requires_mask else sequence.copy()
        eval_seq = eval_seq[-max_seq_len:]
        if mask_token_id not in eval_seq:
            raise ValueError(f"User {row['user_id']} evaluation sequence does not contain MASK token ({mask_token_id}).")
        mask_position = len(eval_seq) - 1 - eval_seq[::-1].index(mask_token_id)
        pad_len = max_seq_len - len(eval_seq)
        padded = [pad_token_id] * pad_len + eval_seq
        mask_position += pad_len
        prepared.append((row["user_id"], len(sequence), target, sequence, padded, mask_position))

    if not prepared:
        raise ValueError("No users were available for BERT4Rec evaluation.")

    was_training = model.training
    model.eval()
    metric_rows = []
    debug_outputs = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(prepared), batch_size):
                batch_rows = prepared[offset:offset + batch_size]
                input_ids = torch.tensor([r[4] for r in batch_rows], dtype=torch.long, device=device)
                mask_positions = torch.tensor([r[5] for r in batch_rows], dtype=torch.long, device=device)
                logits = model(input_ids)
                row_indices = torch.arange(len(batch_rows), device=device)
                scores = logits[row_indices, mask_positions].float()
                scores[:, pad_token_id] = -float("inf")
                scores[:, mask_token_id] = -float("inf")

                if exclude_seen:
                    for i, row in enumerate(batch_rows):
                        seen = [x for x in row[3] if 0 <= x < vocab_size]
                        if seen:
                            scores[i, torch.tensor(seen, dtype=torch.long, device=device)] = -float("inf")

                scores[:, :2] = -float("inf")
                # Sorting on device avoids Python sorting every item's score.
                ranked = torch.argsort(scores[:, 2:], dim=1, descending=True) + 2
                top10 = ranked[:, :10].cpu()
                targets = torch.tensor([r[2] for r in batch_rows], dtype=torch.long)
                hits5 = top10[:, :5].eq(targets[:, None])
                hits10 = top10.eq(targets[:, None])
                rank_pos = torch.where(
                    ranked.cpu().eq(targets[:, None]),
                    torch.arange(1, ranked.shape[1] + 1)[None, :],
                    torch.full((len(batch_rows), ranked.shape[1]), ranked.shape[1] + 1),
                ).min(dim=1).values
                for i, row in enumerate(batch_rows):
                    rank = int(rank_pos[i])
                    user_metrics = {
                        "HR@5": float(bool(hits5[i].any())),
                        "HR@10": float(bool(hits10[i].any())),
                        "NDCG@5": 1.0 / np.log2(rank + 1) if rank <= 5 else 0.0,
                        "NDCG@10": 1.0 / np.log2(rank + 1) if rank <= 10 else 0.0,
                    }
                    metric_rows.append(user_metrics)
                    if len(debug_outputs) < 5:
                        debug_outputs.append({
                            "user_id": row[0], "history_len": row[1], "target": row[2],
                            "top_10": top10[i].tolist(), "rank": rank if rank <= vocab_size - 2 else -1,
                            "hit_5": bool(user_metrics["HR@5"]), "hit_10": bool(user_metrics["HR@10"]),
                        })
    finally:
        if was_training:
            model.train()

    return {
        "HR@5": float(np.mean([m["HR@5"] for m in metric_rows])),
        "HR@10": float(np.mean([m["HR@10"] for m in metric_rows])),
        "NDCG@5": float(np.mean([m["NDCG@5"] for m in metric_rows])),
        "NDCG@10": float(np.mean([m["NDCG@10"] for m in metric_rows])),
        "debug_outputs": debug_outputs,
    }
