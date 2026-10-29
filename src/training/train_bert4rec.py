"""
Training pipeline for BERT4Rec.

Normal training:
    checkpoints/<experiment>/  -> best model checkpoint
    results/<experiment>/      -> training history, config, and run summary

Hyperparameter search:
    Call train_bert4rec(config_dict, save_artifacts=False) to train without
    writing per-trial checkpoints, histories, or run summaries. The summary
    (including memory measurements) is still computed and returned.

The run summary follows the uniform schema defined in
``src.utils.run_profiling`` (shared with JEPA and causal JEPA).
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
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.datasets import BERT4RecDataset
from src.evaluation.evaluate import evaluate_bert4rec
from src.models.bert4rec import BERT4Rec
from src.training.train_jepa import (
    _resolve_device,
    save_epoch_history,
    save_run_summary,
    synchronize_cuda,
)
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.overrides import apply_name, apply_overrides
from src.utils.reproducibility import set_seed
from src.utils.run_profiling import (
    MemoryTracker,
    build_run_summary,
    count_parameters,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "BERT4Rec"
DATASET_NAME = "MovieLens-1M"
IGNORE_INDEX = -100


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------

def _build_optimizer(
    model: nn.Module,
    training_config: dict,
    device: torch.device,
) -> torch.optim.Optimizer:
    """AdamW, using the fused kernel when supported."""

    kwargs = {
        "lr": training_config["learning_rate"],
        "weight_decay": training_config["weight_decay"],
    }

    if device.type == "cuda" and training_config.get("fused_adamw", True):
        try:
            return AdamW(model.parameters(), **kwargs, fused=True)
        except (TypeError, RuntimeError):
            pass

    return AdamW(model.parameters(), **kwargs)


def _resolve_amp(
    training_config: dict,
    device: torch.device,
) -> tuple[bool, torch.dtype, str | None, torch.amp.GradScaler]:
    """Return (use_amp, dtype, dtype name, scaler)."""

    requested = training_config.get("mixed_precision", "bf16")
    on_cuda = device.type == "cuda"

    use_bf16 = (
        on_cuda and requested == "bf16" and torch.cuda.is_bf16_supported()
    )
    use_fp16 = (
        on_cuda and requested == "fp16" and not torch.cuda.is_bf16_supported()
    )

    use_amp = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_dtype_name = (
        "bfloat16" if use_bf16 else "float16" if use_fp16 else None
    )

    # Loss scaling is only needed for float16; a disabled scaler is a no-op.
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    logger.info(
        "Mixed precision: %s",
        "bf16" if use_bf16 else "fp16" if use_fp16 else "off",
    )
    return use_amp, amp_dtype, amp_dtype_name, scaler


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict,
) -> CosineAnnealingLR | None:
    if not training_config.get("use_scheduler", False):
        return None

    scheduler_type = training_config.get("scheduler_type", "cosine_annealing")
    if scheduler_type != "cosine_annealing":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return CosineAnnealingLR(
        optimizer,
        T_max=training_config["num_epochs"],
        eta_min=training_config.get("min_lr", 1e-5),
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: BERT4Rec,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    num_samples: int,
) -> tuple[float, float]:
    """Train for one epoch; return mean per-sample loss and elapsed time."""

    model.train()

    non_blocking = device.type == "cuda"
    scale_loss = scaler.is_enabled()

    running_loss = 0.0
    epoch_start = time.perf_counter()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels = batch["labels"].to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)

        supervised = labels.ne(IGNORE_INDEX)
        if not supervised.any():
            continue

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            # Project only masked positions, not all sequence positions.
            logits = model(input_ids, target_mask=supervised)
            loss = criterion(logits.float(), labels[supervised])

        if scale_loss:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

    synchronize_cuda(device)

    elapsed = time.perf_counter() - epoch_start
    return running_loss / num_samples, elapsed


def train_bert4rec(
    config_path: str | Path | dict,
    *,
    save_artifacts: bool = True,
) -> dict:
    """
    Train BERT4Rec using a YAML path or an in-memory configuration.

    Args:
        config_path:
            Path to a YAML file, or a complete configuration dictionary.
        save_artifacts:
            True for normal training. Saves checkpoints, histories, the
            resolved config, and run summaries.
            False for hyperparameter search. Saves none of those artifacts;
            the caller is responsible for saving aggregate search reports.

    Returns:
        A dictionary containing the best validation NDCG@10, best epoch,
        best checkpoint path (or None), results path (or None), per-epoch
        training history, and the uniform run summary.
    """

    if isinstance(config_path, dict):
        config = deepcopy(config_path)
    else:
        config = load_config(str(config_path))

    seed = config.get("seed", 42)
    set_seed(seed)

    device = _resolve_device(config)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config.setdefault("output", {})

    num_epochs = training_config["num_epochs"]
    batch_size = training_config["batch_size"]
    num_workers = training_config.get("num_workers", 0)
    experiment_name = output_config["experiment_name"]

    # -----------------------------------------------------------------
    # Output paths (only when saving artifacts)
    # -----------------------------------------------------------------

    results_dir = None
    best_checkpoint_path = None
    history_path = None
    summary_path = None
    config_path_out = None

    if save_artifacts:
        results_dir = Path(
            output_config.get("results_dir")
            or Path("results") / experiment_name
        )
        checkpoint_dir = Path(
            output_config.get("checkpoint_dir")
            or Path("checkpoints") / experiment_name
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_checkpoint_path = checkpoint_dir / output_config.get(
            "best_checkpoint_name", f"{experiment_name}_best.pt"
        )
        history_path = results_dir / "training_history.csv"
        summary_path = results_dir / "run_summary.json"
        config_path_out = results_dir / "config.yaml"

        output_config["checkpoint_dir"] = str(checkpoint_dir)
        output_config["results_dir"] = str(results_dir)
        output_config["best_checkpoint_name"] = best_checkpoint_path.name

    # -----------------------------------------------------------------
    # Setup (measured)
    # -----------------------------------------------------------------

    tracker = MemoryTracker(device)
    tracker.mark("process_start")

    with tracker.phase("setup"):
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
                    "Validation data must contain either "
                    "'validation_target' or 'validation_item'."
                )

        train_dataset = BERT4RecDataset(
            sequences=train_df["train_sequence"].tolist(),
            max_seq_len=model_config["max_seq_len"],
            mask_token=model_config["mask_token_id"],
            mask_probability=training_config["mask_probability"],
            pad_token=model_config["pad_token_id"],
        )

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = training_config.get(
                "persistent_workers", True
            )
        train_loader = DataLoader(train_dataset, **loader_kwargs)
        tracker.mark("after_data")

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
        tracker.mark("after_model")

        optimizer = _build_optimizer(model, training_config, device)
        use_amp, amp_dtype, amp_dtype_name, scaler = _resolve_amp(
            training_config, device
        )
        criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        scheduler = _build_scheduler(optimizer, training_config)

    tracker.mark("after_setup")

    total_params, trainable_params = count_parameters(model)

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation users: %d", len(val_df))
    logger.info("Training batches per epoch: %d", len(train_loader))
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # -----------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------

    best_ndcg10 = float("-inf")
    best_epoch = 0
    best_metrics: dict = {}
    best_epoch_record: dict = {}
    epoch_history: list[dict] = []

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()

    logger.info(
        "Starting %s training for %d epochs. Artifact saving: %s",
        experiment_name,
        num_epochs,
        save_artifacts,
    )

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------

    for epoch in range(1, num_epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]

        with tracker.phase("train"):
            train_loss, train_time = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                num_samples=len(train_dataset),
            )

        if scheduler is not None:
            scheduler.step()

        with tracker.phase("validation"):
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

        memory_record = tracker.end_epoch(epoch, train_time, validation_time)

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "HR@5": val_metrics["HR@5"],
            "HR@10": val_metrics["HR@10"],
            "NDCG@5": val_metrics["NDCG@5"],
            "NDCG@10": val_metrics["NDCG@10"],
            "NDCG@10_selection": val_metrics["NDCG@10"],
            "lr": epoch_lr,
            "train_time_seconds": train_time,
            "validation_time_seconds": validation_time,
            "epoch_time_seconds": train_time + validation_time,
            **memory_record,
        }
        epoch_history.append(record)

        if save_artifacts:
            save_epoch_history(epoch_history, history_path)

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | HR@10=%.4f | NDCG@10=%.4f | "
            "lr=%.2e | train=%.1fs | val=%.1fs | "
            "peak_alloc=%.2f GB | peak_reserved=%.2f GB",
            epoch,
            num_epochs,
            train_loss,
            record["HR@10"],
            record["NDCG@10"],
            epoch_lr,
            train_time,
            validation_time,
            record["gpu_epoch_peak_allocated_gb"],
            record["gpu_epoch_peak_reserved_gb"],
        )

        if record["NDCG@10"] > best_ndcg10:
            best_ndcg10 = record["NDCG@10"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_epoch_record = dict(record)

            if save_artifacts:
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": val_metrics,
                    "training_history": epoch_history,
                    "config": config,
                }
                if scheduler is not None:
                    checkpoint_state["scheduler_state_dict"] = (
                        scheduler.state_dict()
                    )

                save_checkpoint(checkpoint_state, best_checkpoint_path)
                logger.info(
                    "Saved new best checkpoint: %s", best_checkpoint_path
                )

    # -----------------------------------------------------------------
    # Run summary
    # -----------------------------------------------------------------

    synchronize_cuda(device)
    tracker.mark("after_training")

    total_wall_seconds = time.perf_counter() - total_start

    use_scheduler = scheduler is not None
    base_lr = training_config["learning_rate"]
    min_lr = training_config.get("min_lr", 1e-5)

    summary = build_run_summary(
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        dataset=DATASET_NAME,
        device=device,
        seed=seed,
        run_start=run_start,
        run_end=datetime.now().isoformat(),
        total_wall_seconds=total_wall_seconds,
        configured_epochs=num_epochs,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_epoch_loss=best_epoch_record.get("loss"),
        best_selection_score=best_ndcg10 if best_epoch else None,
        early_stopped=False,
        training_configuration={
            "batch_size": batch_size,
            "learning_rate": base_lr,
            "weight_decay": training_config["weight_decay"],
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name,
            "compile": False,
            "patience": None,
            "lr_schedule": (
                training_config.get("scheduler_type", "cosine_annealing")
                if use_scheduler else "constant"
            ),
            "warmup_epochs": 0,
            "min_lr_ratio": (min_lr / base_lr) if use_scheduler else 0.0,
            "selection_window": 1,
            "save_checkpoint": save_artifacts,
            # Model-specific settings.
            "mask_probability": training_config["mask_probability"],
            "fused_adamw": training_config.get("fused_adamw", True),
            "num_workers": num_workers,
            "evaluation_batch_size": training_config.get(
                "evaluation_batch_size", 256
            ),
        },
        model_configuration=model_config,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        data_info={
            "training_samples": len(train_dataset),
            "validation_samples": len(val_df),
            "training_batches_per_epoch": len(train_loader),
        },
        samples_per_epoch=len(train_dataset),
        files={
            "checkpoint": (
                str(best_checkpoint_path) if best_checkpoint_path else None
            ),
            "epoch_history": str(history_path) if history_path else None,
            "run_summary": str(summary_path) if summary_path else None,
            "config": str(config_path_out) if config_path_out else None,
        },
    )

    if save_artifacts:
        save_run_summary(summary, summary_path)
        with open(config_path_out, "w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    memory_gpu = summary["memory"]["gpu"]
    logger.info(
        "Training complete. Best validation NDCG@10: %.4f at epoch %d.",
        best_ndcg10,
        best_epoch,
    )
    logger.info(
        "Memory | peak allocated: %.2f GB | peak reserved: %.2f GB | "
        "CPU peak RSS: %.2f GB",
        memory_gpu["peak_allocated_gb"],
        memory_gpu["peak_reserved_gb"],
        summary["memory"]["cpu"]["peak_rss_gb"],
    )

    return {
        "best_ndcg_10": best_ndcg10,
        "best_epoch": best_epoch,
        "best_model_path": (
            str(best_checkpoint_path) if best_checkpoint_path else None
        ),
        "results_path": str(results_dir) if results_dir else None,
        "training_history": epoch_history,
        "summary": summary,
    }


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

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
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config value, e.g. "
            "--set training.learning_rate=0.001. "
            "Repeatable. Use +KEY=VALUE to add a new key."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Experiment name; redirects artifacts to "
            "checkpoints/<name> and results/<name> unless "
            "a custom output.results_dir is configured."
        ),
    )
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.name:
        config = apply_name(config, args.name)

    setup_logging()
    train_bert4rec(config)


if __name__ == "__main__":
    main()