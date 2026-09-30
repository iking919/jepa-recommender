"""
Training pipeline for BERT4Rec.

The module contains reusable training logic and a command-line entry point.
Experiment artifacts are split into:
    checkpoints/<experiment>/  -> model checkpoints only
    results/<experiment>/      -> histories and run summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.datasets import BERT4RecDataset
from src.evaluation.evaluate import evaluate_bert4rec
from src.models.bert4rec import BERT4Rec
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed


logger = logging.getLogger(__name__)


def _save_csv(records: list[dict], path: Path) -> None:
    """Write a list of dictionaries to CSV."""

    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _save_json(data: dict, path: Path) -> None:
    """Write a dictionary to formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _resolve_device(config: dict) -> torch.device:
    """Resolve and validate the configured training device."""

    configured_device = config.get("device")

    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested in the configuration, but CUDA is not "
            "available in the installed PyTorch environment."
        )

    if configured_device:
        return torch.device(configured_device)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def train_bert4rec(config_path: str) -> dict:
    """Train BERT4Rec using the supplied YAML configuration."""

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    device = _resolve_device(config)
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    train_path = Path(data_config["train_sequences_path"])
    validation_path = Path(data_config["validation_path"])

    logger.info("Loading training data from: %s", train_path)
    logger.info("Loading validation data from: %s", validation_path)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(validation_path)

    if "train_sequence" not in train_df.columns:
        raise ValueError(
            "Training data must contain a 'train_sequence' column."
        )

    if "validation_target" not in val_df.columns:
        if "validation_item" in val_df.columns:
            val_df = val_df.rename(
                columns={"validation_item": "validation_target"}
            )
        else:
            raise ValueError(
                "Validation data must contain either 'validation_target' "
                "or 'validation_item'."
            )

    train_dataset = BERT4RecDataset(
        sequences=train_df["train_sequence"].tolist(),
        max_seq_len=model_config["max_seq_len"],
        mask_token=model_config["mask_token_id"],
        mask_probability=training_config["mask_probability"],
        pad_token=model_config["pad_token_id"],
    )

    num_workers = training_config.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = training_config.get(
            "persistent_workers",
            True,
        )

    train_loader = DataLoader(train_dataset, **loader_kwargs)

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

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation users: %d", len(val_df))
    logger.info("Training batches per epoch: %d", len(train_loader))
    logger.info("Trainable parameters: %s", f"{trainable_parameters:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    scheduler = None
    if training_config.get("use_scheduler", False):
        scheduler_type = training_config.get(
            "scheduler_type",
            "cosine_annealing",
        )

        if scheduler_type != "cosine_annealing":
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}"
            )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config["num_epochs"],
            eta_min=training_config.get("min_lr", 1e-5),
        )

    experiment_name = output_config["experiment_name"]
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    results_dir = Path(output_config["results_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = (
        checkpoint_dir / output_config["best_checkpoint_name"]
    )
    history_path = results_dir / "training_history.csv"
    summary_path = results_dir / "run_summary.json"

    best_ndcg10 = float("-inf")
    best_epoch = 0
    training_history = []

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()

    num_epochs = training_config["num_epochs"]

    logger.info(
        "Starting %s training for %d epochs.",
        experiment_name,
        num_epochs,
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels = batch["labels"].to(
                device,
                non_blocking=device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids)
            loss = criterion(
                logits.reshape(-1, model_config["vocab_size"]),
                labels.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / len(train_dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics = evaluate_bert4rec(
            model=model,
            df=val_df,
            device=device,
            max_seq_len=model_config["max_seq_len"],
            vocab_size=model_config["vocab_size"],
            pad_token_id=model_config["pad_token_id"],
            mask_token_id=model_config["mask_token_id"],
            target_column="validation_target",
            input_column="train_sequence",
            input_requires_mask=True,
            exclude_seen=True,
        )

        epoch_time = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "learning_rate": current_lr,
            "HR@5": val_metrics["HR@5"],
            "HR@10": val_metrics["HR@10"],
            "NDCG@5": val_metrics["NDCG@5"],
            "NDCG@10": val_metrics["NDCG@10"],
            "epoch_time_seconds": epoch_time,
        }
        training_history.append(record)
        _save_csv(training_history, history_path)

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | HR@5=%.4f | HR@10=%.4f | "
            "NDCG@5=%.4f | NDCG@10=%.4f | lr=%.6f | time=%.1fs",
            epoch,
            num_epochs,
            train_loss,
            record["HR@5"],
            record["HR@10"],
            record["NDCG@5"],
            record["NDCG@10"],
            current_lr,
            epoch_time,
        )

        if record["NDCG@10"] > best_ndcg10:
            best_ndcg10 = record["NDCG@10"]
            best_epoch = epoch

            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_metrics": val_metrics,
                "training_history": training_history,
                "config": config,
            }

            if scheduler is not None:
                checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()

            save_checkpoint(checkpoint_state, best_checkpoint_path)
            logger.info(
                "Saved new best checkpoint: %s",
                best_checkpoint_path,
            )

    total_time = time.perf_counter() - total_start
    run_end = datetime.now().isoformat()

    summary = {
        "model": "BERT4Rec",
        "experiment_name": experiment_name,
        "dataset": "MovieLens-1M",
        "run_start": run_start,
        "run_end": run_end,
        "device": str(device),
        "seed": config.get("seed", 42),
        "configured_epochs": num_epochs,
        "completed_epochs": len(training_history),
        "best_epoch": best_epoch,
        "best_validation_metrics": (
            training_history[best_epoch - 1]
            if best_epoch > 0
            else {}
        ),
        "timing": {
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60.0,
            "average_epoch_time_seconds": (
                total_time / len(training_history)
                if training_history
                else 0.0
            ),
        },
        "model_parameters": {
            "trainable": trainable_parameters,
        },
        "files": {
            "checkpoint": str(best_checkpoint_path),
            "training_history": str(history_path),
        },
    }

    _save_json(summary, summary_path)

    logger.info(
        "Training complete. Best validation NDCG@10: %.4f at epoch %d.",
        best_ndcg10,
        best_epoch,
    )

    return {
        "best_ndcg_10": best_ndcg10,
        "best_epoch": best_epoch,
        "best_model_path": str(best_checkpoint_path),
        "results_path": str(results_dir),
        "training_history": training_history,
    }


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Train a BERT4Rec sequential recommender."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    args = parser.parse_args()

    setup_logging()
    train_bert4rec(args.config)


if __name__ == "__main__":
    main()
