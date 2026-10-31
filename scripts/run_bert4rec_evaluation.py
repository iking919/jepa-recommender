"""
Evaluate a trained BERT4Rec checkpoint on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.evaluation.evaluate import evaluate_bert4rec
from src.models import BERT4Rec
from src.utils.checkpointing import load_checkpoint
from src.utils.config import add_config_args, load_config_from_args
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BERT4Rec model on the test set."
    )
    add_config_args(parser)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the trained BERT4Rec checkpoint. Defaults to the run's "
             "own checkpoint: checkpoints/<dataset>/<experiment>/"
             "<experiment>_best.pt",
    )
    args = parser.parse_args()

    setup_logging()

    config = load_config_from_args(args)
    model_config = config["model"]
    data_config = config["data"]
    output_config = config["output"]

    configured_device = config.get("device")
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(
        configured_device
        if configured_device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_df = pd.read_parquet(data_config["test_path"])

    model = BERT4Rec(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
    ).to(device)

    checkpoint_path = Path(
        args.checkpoint
        or Path(output_config["checkpoint_dir"])
        / output_config["best_checkpoint_name"]
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_bert4rec(
        model=model,
        df=test_df,
        device=device,
        max_seq_len=model_config["max_seq_len"],
        vocab_size=model_config["vocab_size"],
        pad_token_id=model_config["pad_token_id"],
        mask_token_id=model_config["mask_token_id"],
        target_column="test_item",
        input_column="test_input",
        input_requires_mask=True,
        exclude_seen=True,
    )

    print("\nBERT4Rec Test Results")
    print("=" * 40)
    for name in ("HR@5", "HR@10", "NDCG@5", "NDCG@10"):
        print(f"{name:<8} {metrics[name]:.4f}")

    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    result_path = results_dir / "test_results.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "BERT4Rec",
                "dataset": config["dataset"],
                "checkpoint": str(checkpoint_path),
                "metrics": {
                    name: metrics[name]
                    for name in ("HR@5", "HR@10", "NDCG@5", "NDCG@10")
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved test results to: {result_path}")


if __name__ == "__main__":
    main()
