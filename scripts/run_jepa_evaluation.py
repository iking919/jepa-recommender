"""
Evaluate a trained JEPA checkpoint on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import JEPADataset
from src.evaluation.evaluate import evaluate_jepa
from src.models import JEPA
from src.utils.checkpointing import load_checkpoint
from src.utils.config import add_config_args, load_config_from_args
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained JEPA model on the test set."
    )
    add_config_args(parser)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the trained JEPA checkpoint. Defaults to the run's "
             "own checkpoint: checkpoints/<dataset>/<experiment>/"
             "<experiment>_best.pt",
    )
    args = parser.parse_args()

    setup_logging()

    config = load_config_from_args(args)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
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

    test_dataset = JEPADataset(
        contexts=test_df["context_sequence"].tolist(),
        targets=test_df["target_item"].tolist(),
        max_seq_len=model_config["max_seq_len"],
        pad_token=model_config.get("pad_token_id", 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = JEPA(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config.get("pad_token_id", 0),
        causal=model_config.get("causal", False),
    ).to(device)

    checkpoint_path = Path(
        args.checkpoint
        or Path(output_config["checkpoint_dir"])
        / output_config["best_checkpoint_name"]
    )
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_jepa(
        model=model,
        loader=test_loader,
        num_items=model_config["vocab_size"],
        device=device,
        pad_token_id=model_config.get("pad_token_id", 0),
        mask_token_id=model_config.get("mask_token_id", 1),
        exclude_seen=True,
    )

    print("\nJEPA Test Results")
    print("=" * 40)
    for name in ("HR@5", "HR@10", "NDCG@5", "NDCG@10"):
        print(f"{name:<8} {metrics[name]:.4f}")

    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    result_path = results_dir / "test_results.json"
    result_path.write_text(
        json.dumps(
            {
                "model": "JEPA",
                "dataset": config["dataset"],
                "checkpoint": str(checkpoint_path),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved test results to: {result_path}")


if __name__ == "__main__":
    main()
