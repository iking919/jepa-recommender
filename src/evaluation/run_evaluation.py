"""
evaluate.py

Evaluate a trained BERT4Rec model on the held-out test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.evaluation.evaluate import evaluate_bert4rec
from src.models import BERT4Rec
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BERT4Rec model on the test set."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the model configuration YAML file.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    config = load_config(args.config)

    model_config = config["model"]
    data_config = config["data"]

    # ------------------------------------------------------------------
    # Select device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------

    test_path = Path(data_config["test_path"])

    print(f"Loading test data from: {test_path}")

    test_df = pd.read_parquet(test_path)

    print(f"Loaded {len(test_df)} test users.")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------

    model = BERT4Rec(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config["pad_token_id"],
    )

    model.to(device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    print(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # ------------------------------------------------------------------
    # Evaluate test set
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print("\nTest Results")
    print("=" * 40)
    print(f"HR@5:    {metrics['hr_5']:.4f}")
    print(f"HR@10:   {metrics['hr_10']:.4f}")
    print(f"NDCG@5:  {metrics['ndcg_5']:.4f}")
    print(f"NDCG@10: {metrics['ndcg_10']:.4f}")

    print("\nDebug Outputs")
    print("=" * 40)

    for output in metrics["debug_outputs"]:
        print(output)


if __name__ == "__main__":
    main()