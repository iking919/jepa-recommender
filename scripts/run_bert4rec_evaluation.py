"""Evaluate a trained BERT4Rec checkpoint on the held-out test set.

Usage (from the repository root):
    python -m scripts.run_bert4rec_evaluation --config configs/ml-1m/bert4rec_tuned.yaml

Reads the checkpoint saved by training (``--checkpoint`` overrides it) and
writes ``test_results.json`` to the experiment's results folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate import evaluate_bert4rec
from src.models.factory import build_bert4rec
from src.training.common import METRIC_NAMES, resolve_device
from src.utils.checkpointing import load_checkpoint
from src.utils.config import add_config_args, load_config_from_args
from src.utils.logging_utils import setup_logging


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

    device = resolve_device(config)

    test_df = pd.read_parquet(data_config["test_path"])

    model = build_bert4rec(model_config).to(device)

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
    for name in METRIC_NAMES:
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
                "metrics": {name: metrics[name] for name in METRIC_NAMES},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved test results to: {result_path}")


if __name__ == "__main__":
    main()
