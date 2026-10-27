"""
Check that the fast GPU evaluation matches the original implementation on a
real checkpoint, and measure how much faster it is.

    python -m scripts.check_eval_equivalence \
        --config configs/jepa_causal_tuned.yaml \
        --checkpoint checkpoints/jepa_causal_tuned/jepa_causal_tuned_best.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.datasets import JEPADataset
from src.data.gpu_loader import GPUBatchLoader
from src.evaluation.evaluate import evaluate_jepa, evaluate_jepa_reference
from src.models import JEPA
from src.training.losses import scoring_for_loss
from src.utils.checkpointing import load_checkpoint
from src.utils.config import load_config

METRICS = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device, repeats: int = 3):
    fn()  # warm-up
    _sync(device)

    best = float("inf")
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        _sync(device)
        best = min(best, time.perf_counter() - start)

    return result, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    path = data_config[f"{args.split}_path"]
    df = pd.read_parquet(path)

    dataset = JEPADataset(
        contexts=df["context_sequence"].tolist(),
        targets=df["target_item"].tolist(),
        max_seq_len=model_config["max_seq_len"],
        pad_token=model_config.get("pad_token_id", 0),
    )

    loader = GPUBatchLoader(
        dataset,
        batch_size=training_config["batch_size"],
        device=device,
        shuffle=False,
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

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.scoring = scoring_for_loss(training_config.get("loss", "cosine"))
    model.eval()

    kwargs = dict(
        model=model,
        loader=loader,
        num_items=model_config["vocab_size"],
        device=device,
    )

    reference, reference_seconds = _timed(
        lambda: evaluate_jepa_reference(**kwargs), device
    )
    fast, fast_seconds = _timed(lambda: evaluate_jepa(**kwargs), device)

    print(f"\n{args.split} users: {len(dataset)} | device: {device}")
    print(f"{'metric':<9}{'reference':>12}{'fast':>12}{'|diff|':>12}")

    worst = 0.0

    for name in METRICS:
        diff = abs(reference[name] - fast[name])
        worst = max(worst, diff)
        print(
            f"{name:<9}{reference[name]:>12.6f}{fast[name]:>12.6f}"
            f"{diff:>12.2e}"
        )

    print(
        f"\nreference: {reference_seconds:.3f}s | fast: {fast_seconds:.3f}s "
        f"| speed-up: {reference_seconds / fast_seconds:.1f}x"
    )

    if worst > 1e-6:
        print(
            "\nMetrics differ by more than 1e-6. Check for ties in the "
            "scores before trusting the fast path."
        )
        sys.exit(1)

    print("\nOK: metrics match.")


if __name__ == "__main__":
    main()
