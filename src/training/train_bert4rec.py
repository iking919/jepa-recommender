"""
Training pipeline for BERT4Rec.

Normal training:
    checkpoints/<dataset>/<experiment>/  -> best model checkpoint
    results/<dataset>/<experiment>/      -> history, summary, resolved config

Hyperparameter search:
    train_bert4rec(config_dict, save_artifacts=False) writes no per-trial
    checkpoint, history, summary, or config files.
"""

from __future__ import annotations

import argparse
import logging
import time
from copy import deepcopy
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
from src.training.common import (
    count_parameters,
    get_gpu_name,
    get_peak_gpu_memory_gb,
    log_hardware,
    reset_peak_gpu_memory,
    resolve_device,
    save_epoch_history,
    save_run_summary,
    set_seed,
    synchronize_cuda,
)
from src.utils.checkpointing import save_checkpoint
from src.utils.config import (
    add_config_args,
    load_config,
    load_config_from_args,
    save_resolved_config,
)
from src.utils.logging import setup_logging
from src.utils.paths import resolve_output

logger = logging.getLogger(__name__)


def train_bert4rec(
    config_path: str | Path | dict,
    *,
    save_artifacts: bool = True,
) -> dict:
    """
    Train BERT4Rec using a YAML path or an in-memory configuration.

    save_artifacts=False disables all per-trial file output for hyperparameter
    search. The caller remains responsible for aggregate search reports.
    """
    if isinstance(config_path, dict):
        config = deepcopy(config_path)
    else:
        config = load_config(str(config_path))

    resolve_output(config)
    seed = config.get("seed", 42)
    set_seed(seed)

    device = resolve_device(config)
    log_hardware(device)

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
        raise ValueError("Training data must contain a 'train_sequence' column.")

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
            "persistent_workers", True,
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

    total_params, trainable_params = count_parameters(model)
    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation users: %d", len(val_df))
    logger.info("Training batches per epoch: %d", len(train_loader))
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    optimizer_kwargs = {
        "lr": training_config["learning_rate"],
        "weight_decay": training_config["weight_decay"],
    }
    fused_adamw_requested = (
        device.type == "cuda"
        and training_config.get("fused_adamw", True)
    )
    fused_adamw_used = False
    if fused_adamw_requested:
        try:
            optimizer = AdamW(
                model.parameters(), **optimizer_kwargs, fused=True,
            )
            fused_adamw_used = True
        except (TypeError, RuntimeError):
            optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    # Preserve the original precision-selection behavior.
    use_bf16 = (
        device.type == "cuda"
        and training_config.get("mixed_precision", "bf16") == "bf16"
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        device.type == "cuda"
        and training_config.get("mixed_precision", "bf16") == "fp16"
        and not torch.cuda.is_bf16_supported()
    )
    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_dtype_name = (
        "bf16" if use_bf16 else "fp16" if use_fp16 else "off"
    )
    logger.info("Mixed precision: %s", amp_dtype_name)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scheduler = None
    scheduler_type = None
    if training_config.get("use_scheduler", False):
        scheduler_type = training_config.get(
            "scheduler_type", "cosine_annealing",
        )
        if scheduler_type != "cosine_annealing":
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config["num_epochs"],
            eta_min=training_config.get("min_lr", 1e-5),
        )

    experiment_name = output_config["experiment_name"]
    save_checkpoint_enabled = (
        save_artifacts and training_config.get("save_checkpoint", True)
    )

    if save_artifacts:
        results_dir = Path(output_config["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        history_path = results_dir / "training_history.csv"
        summary_path = results_dir / "run_summary.json"
        config_path = results_dir / "config.yaml"

        if save_checkpoint_enabled:
            checkpoint_dir = Path(output_config["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = (
                checkpoint_dir / output_config["best_checkpoint_name"]
            )
        else:
            checkpoint_dir = None
            best_checkpoint_path = None

        output_config["checkpoint_dir"] = (
            str(checkpoint_dir) if checkpoint_dir is not None else None
        )
        output_config["results_dir"] = str(results_dir)
        output_config["save_checkpoint"] = save_checkpoint_enabled
        save_resolved_config(config, results_dir)
    else:
        results_dir = None
        history_path = None
        summary_path = None
        config_path = None
        best_checkpoint_path = None

    best_ndcg10 = float("-inf")
    best_epoch = 0
    best_metrics: dict = {}
    training_history: list[dict] = []

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()
    total_train_time = 0.0
    total_validation_time = 0.0
    num_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", num_epochs)

    if patience < 1:
        raise ValueError("Training patience must be at least 1.")

    epochs_without_improvement = 0
    early_stopping_triggered = False

    logger.info(
        "Starting %s training for %d epochs. Artifact saving: %s; "
        "checkpoint saving: %s",
        experiment_name, num_epochs, save_artifacts, save_checkpoint_enabled,
    )

    for epoch in range(1, num_epochs + 1):
        reset_peak_gpu_memory(device)
        synchronize_cuda(device)
        train_start = time.perf_counter()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(
                device, non_blocking=device.type == "cuda",
            )
            labels = batch["labels"].to(
                device, non_blocking=device.type == "cuda",
            )

            optimizer.zero_grad(set_to_none=True)
            supervised = labels.ne(-100)
            if not supervised.any():
                continue

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(input_ids, target_mask=supervised)
                loss = criterion(logits.float(), labels[supervised])

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

        if scheduler is not None:
            scheduler.step()

        synchronize_cuda(device)
        train_time = time.perf_counter() - train_start
        total_train_time += train_time

        train_loss = running_loss / len(train_dataset)
        current_lr = optimizer.param_groups[0]["lr"]

        validation_start = time.perf_counter()
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
            batch_size=training_config.get("evaluation_batch_size", 256),
        )
        synchronize_cuda(device)
        validation_time = time.perf_counter() - validation_start
        total_validation_time += validation_time

        epoch_time = train_time + validation_time
        peak_gpu_memory_gb = get_peak_gpu_memory_gb(device)

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "lr": current_lr,
            "HR@5": val_metrics["HR@5"],
            "HR@10": val_metrics["HR@10"],
            "NDCG@5": val_metrics["NDCG@5"],
            "NDCG@10": val_metrics["NDCG@10"],
            "train_time_seconds": train_time,
            "validation_time_seconds": validation_time,
            "epoch_time_seconds": epoch_time,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
        training_history.append(record)
        if save_artifacts:
            save_epoch_history(training_history, history_path)

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | HR@5=%.4f | HR@10=%.4f | "
            "NDCG@5=%.4f | NDCG@10=%.4f | lr=%.6f | train=%.1fs | "
            "val=%.1fs | epoch=%.1fs | peak_gpu=%.2f GB",
            epoch, num_epochs, train_loss,
            record["HR@5"], record["HR@10"],
            record["NDCG@5"], record["NDCG@10"],
            current_lr, train_time, validation_time,
            epoch_time, peak_gpu_memory_gb,
        )

        # BERT4Rec retains raw validation NDCG@10 for model selection.
        if record["NDCG@10"] > best_ndcg10:
            best_ndcg10 = record["NDCG@10"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            epochs_without_improvement = 0

            if save_checkpoint_enabled:
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": val_metrics,
                    "training_history": training_history,
                    "config": config,
                }
                if scheduler is not None:
                    checkpoint_state["scheduler_state_dict"] = (
                        scheduler.state_dict()
                    )
                save_checkpoint(checkpoint_state, best_checkpoint_path)
                logger.info(
                    "Saved new best checkpoint: %s", best_checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stopping_triggered = True

            logger.info(
                "Early stopping at epoch %d. "
                "No validation NDCG@10 improvement for %d consecutive epochs. "
                "Best epoch: %d (NDCG@10=%.4f).",
                epoch,
                patience,
                best_epoch,
                best_ndcg10,
            )
            break

    synchronize_cuda(device)
    total_time = time.perf_counter() - total_start
    run_end = datetime.now().isoformat()
    completed_epochs = len(training_history)
    best_record = (
        training_history[best_epoch - 1] if best_epoch > 0 else {}
    )

    summary = {
        "model": "BERT4Rec",
        "experiment_name": experiment_name,
        "dataset": config.get(
            "dataset",
            data_config.get("dataset_name", "unknown"),
        ),
        "run_start": run_start,
        "run_end": run_end,
        "device": str(device),
        "seed": seed,
        "configured_epochs": num_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_metrics": {
            name: best_metrics.get(name, 0.0)
            for name in ("HR@5", "HR@10", "NDCG@5", "NDCG@10")
        },
        "best_epoch_loss": best_record.get("loss"),
        # BERT4Rec selects on raw NDCG@10, not a smoothed score.
        "best_selection_score": best_ndcg10 if best_epoch else None,
        "timing": {
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60.0,
            "total_train_only_seconds": total_train_time,
            "time_to_best_epoch_seconds": sum(
                row["epoch_time_seconds"]
                for row in training_history[:best_epoch]
            ),
            "average_epoch_time_seconds": (
                sum(row["epoch_time_seconds"] for row in training_history)
                / completed_epochs if completed_epochs else 0.0
            ),
            "average_train_time_seconds": (
                total_train_time / completed_epochs
                if completed_epochs else 0.0
            ),
            "average_validation_time_seconds": (
                total_validation_time / completed_epochs
                if completed_epochs else 0.0
            ),
        },
        "hardware": {
            "device": str(device),
            "gpu": get_gpu_name(device),
            "peak_gpu_memory_gb": max(
                (row["peak_gpu_memory_gb"] for row in training_history),
                default=0.0,
            ),
        },
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "training_configuration": {
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config["weight_decay"],
            "num_workers": num_workers,
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name,
            "fused_adamw": fused_adamw_used,
            "use_scheduler": scheduler is not None,
            "scheduler_type": scheduler_type,
            "evaluation_batch_size": training_config.get(
                "evaluation_batch_size", 256,
            ),
            "save_checkpoint": save_checkpoint_enabled,
            "num_epochs": num_epochs,
            "patience": patience,
            "early_stopping_triggered": early_stopping_triggered,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "model_configuration": model_config,
        "data": {
            "training_samples": len(train_dataset),
            "validation_samples": len(val_df),
            "training_batches_per_epoch": len(train_loader),
        },
        "files": {
            "checkpoint": (
                str(best_checkpoint_path)
                if best_checkpoint_path is not None else None
            ),
            "epoch_history": (
                str(history_path) if history_path is not None else None
            ),
            "run_summary": (
                str(summary_path) if summary_path is not None else None
            ),
            "config": str(config_path) if config_path is not None else None,
        },
    }

    if save_artifacts:
        save_run_summary(summary, summary_path)

    logger.info(
        "Training complete. Best validation NDCG@10: %.4f at epoch %d.",
        best_ndcg10, best_epoch,
    )

    return {
        "best_ndcg_10": best_ndcg10,
        "best_epoch": best_epoch,
        "best_model_path": (
            str(best_checkpoint_path)
            if best_checkpoint_path is not None else None
        ),
        "results_path": str(results_dir) if results_dir is not None else None,
        "training_history": training_history,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BERT4Rec sequential recommender."
    )
    add_config_args(parser)
    args = parser.parse_args()
    setup_logging()
    train_bert4rec(load_config_from_args(args))


if __name__ == "__main__":
    main()
