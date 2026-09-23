"""
train_bert4rec.py

Training pipeline for the BERT4Rec sequential recommendation model.

This module contains the reusable training logic. The command-line entry
point is provided separately by scripts/train_bert4rec.py.

Example:
    python -m src.training.train_bert4rec \
        --config configs/bert4rec_tuned.yaml
"""

from __future__ import annotations

import argparse
import os
import time

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


logger = setup_logging(__name__)


def train_bert4rec(config_path: str) -> dict:
    """
    Train BERT4Rec using the supplied configuration.

    Args:
        config_path:
            Path to the YAML configuration file.

    Returns:
        Dictionary containing training results and history.
    """

    # ------------------------------------------------------------------
    # Configuration and reproducibility
    # ------------------------------------------------------------------

    config = load_config(config_path)

    seed = config.get("seed", 42)
    set_seed(seed)

    configured_device = config.get("device")

    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested in the configuration, but CUDA is not "
            "available in the installed PyTorch environment."
        )

    if configured_device:
        device = torch.device(configured_device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------

    train_sequences_path = config["data"]["train_sequences_path"]
    validation_path = config["data"]["validation_path"]

    logger.info(
        "Loading training data from: %s",
        train_sequences_path,
    )

    logger.info(
        "Loading validation data from: %s",
        validation_path,
    )

    # ------------------------------------------------------------------
    # Load training and validation data
    # ------------------------------------------------------------------

    try:
        train_df = pd.read_parquet(train_sequences_path)
        val_df = pd.read_parquet(validation_path)
    except Exception as exc:
        logger.error("Failed to load dataset files: %s", exc)
        raise

    if "train_sequence" not in train_df.columns:
        raise ValueError(
            "Training data must contain a 'train_sequence' column."
        )

    if "validation_target" not in val_df.columns:
        if "validation_item" in val_df.columns:
            val_df = val_df.rename(
                columns={
                    "validation_item": "validation_target"
                }
            )
        else:
            raise ValueError(
                "Validation data must contain either "
                "'validation_target' or 'validation_item'."
            )

    raw_sequences = train_df["train_sequence"].tolist()

    logger.info(
        "Loaded %d training sequences.",
        len(raw_sequences),
    )

    logger.info(
        "Loaded %d validation users.",
        len(val_df),
    )

    # ------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------

    model_config = config["model"]
    training_config = config["training"]

    vocab_size = model_config["vocab_size"]
    max_seq_len = model_config["max_seq_len"]
    pad_token_id = model_config["pad_token_id"]
    mask_token_id = model_config["mask_token_id"]

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_dataset = BERT4RecDataset(
        sequences=raw_sequences,
        max_seq_len=max_seq_len,
        mask_token=mask_token_id,
        mask_probability=training_config["mask_probability"],
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    num_workers = training_config.get("num_workers", 0)

    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            training_config.get("persistent_workers", True)
        )

    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs,
    )

    logger.info(
        "Training batches per epoch: %d",
        len(train_loader),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = BERT4Rec(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=pad_token_id,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logger.info(
        "Trainable parameters: %s",
        f"{trainable_parameters:,}",
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    # ------------------------------------------------------------------
    # Learning-rate scheduler
    # ------------------------------------------------------------------

    scheduler = None

    if training_config.get("use_scheduler", False):
        scheduler_type = training_config.get(
            "scheduler_type",
            "cosine_annealing",
        )

        if scheduler_type == "cosine_annealing":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=training_config["num_epochs"],
                eta_min=training_config.get(
                    "min_lr",
                    1e-5,
                ),
            )

            logger.info(
                "Using cosine annealing scheduler."
            )

        else:
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}"
            )

    # ------------------------------------------------------------------
    # Checkpoint configuration
    # ------------------------------------------------------------------

    checkpoint_config = config["checkpointing"]

    checkpoint_dir = checkpoint_config["save_dir"]

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    best_checkpoint_path = os.path.join(
        checkpoint_dir,
        checkpoint_config["best_model_name"],
    )

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------

    best_ndcg_10 = float("-inf")
    best_epoch = -1

    training_history = []

    num_epochs = training_config["num_epochs"]

    logger.info(
        "Starting BERT4Rec training for %d epochs.",
        num_epochs,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    for epoch in range(1, num_epochs + 1):

        epoch_start = time.time()

        model.train()

        epoch_loss = 0.0

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
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
                * input_ids.size(0)
            )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        average_loss = (
            epoch_loss / len(train_dataset)
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        val_metrics = evaluate_bert4rec(
            model=model,
            df=val_df,
            device=device,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
            exclude_seen=True,
        )

        epoch_time = time.time() - epoch_start

        hr_10 = val_metrics["hr_10"]
        ndcg_10 = val_metrics["ndcg_10"]

        logger.info(
            "Epoch %02d/%02d | "
            "Loss: %.4f | "
            "LR: %.6f | "
            "Val HR@10: %.4f | "
            "Val NDCG@10: %.4f | "
            "Time: %.1fs",
            epoch,
            num_epochs,
            average_loss,
            current_lr,
            hr_10,
            ndcg_10,
            epoch_time,
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "learning_rate": current_lr,
                "hr_10": hr_10,
                "ndcg_10": ndcg_10,
                "epoch_time": epoch_time,
            }
        )

        # --------------------------------------------------------------
        # Best-model checkpoint
        # --------------------------------------------------------------

        if ndcg_10 > best_ndcg_10:

            best_ndcg_10 = ndcg_10
            best_epoch = epoch

            checkpoint_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_metrics": val_metrics,
                "training_history": training_history,
                "config": config,
            }

            if scheduler is not None:
                checkpoint_state[
                    "scheduler_state_dict"
                ] = scheduler.state_dict()

            save_checkpoint(
                checkpoint_state,
                best_checkpoint_path,
            )

            logger.info(
                "Saved new best checkpoint: %s",
                best_checkpoint_path,
            )

    # ------------------------------------------------------------------
    # Training complete
    # ------------------------------------------------------------------

    logger.info("BERT4Rec training finished.")

    logger.info(
        "Best validation NDCG@10: %.4f at epoch %d.",
        best_ndcg_10,
        best_epoch,
    )

    return {
        "best_ndcg_10": best_ndcg_10,
        "best_epoch": best_epoch,
        "best_model_path": best_checkpoint_path,
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
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    results = train_bert4rec(
        config_path=args.config
    )

    logger.info(
        "Training results: %s",
        results,
    )


if __name__ == "__main__":
    main()