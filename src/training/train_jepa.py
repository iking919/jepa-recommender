"""
Training pipeline for JEPA.

Checkpoint saving is enabled by default. Disable it with:
    training.save_checkpoint: false

Training histories, resolved configurations, and run summaries are always
saved. The run summary follows the uniform schema defined in
``src.utils.run_profiling`` (shared with causal JEPA and BERT4Rec), including
detailed GPU/CPU memory measurements.
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
import torch.nn.functional as F
import yaml

from src.data.datasets import JEPADataset
from src.data.gpu_loader import GPUBatchLoader
from src.evaluation.evaluate import evaluate_jepa
from src.models.jepa import JEPA
from src.training.ema import EMA
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.overrides import apply_name, apply_overrides
from src.utils.reproducibility import set_seed
from src.utils.run_profiling import (  # noqa: F401  (count_parameters re-exported)
    MemoryTracker,
    build_run_summary,
    count_parameters,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "JEPA"
DATASET_NAME = "MovieLens-1M"


# ---------------------------------------------------------------------
# Shared helpers (imported by the causal JEPA and BERT4Rec scripts)
# ---------------------------------------------------------------------

def synchronize_cuda(device: torch.device) -> None:
    """Synchronize CUDA before recording GPU-dependent timings."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_peak_gpu_memory_gb(device: torch.device) -> float:
    """Return peak allocated GPU memory in gigabytes."""

    if device.type != "cuda":
        return 0.0

    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def save_epoch_history(history: list[dict], path: Path) -> None:
    """Write epoch-level training results to CSV."""

    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_run_summary(summary: dict, path: Path) -> None:
    """Write the final run summary to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def _resolve_device(config: dict) -> torch.device:
    """Resolve and validate the configured device."""

    configured_device = config.get("device")

    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested in the configuration, but CUDA is not "
            "available in the installed PyTorch environment."
        )

    if configured_device:
        return torch.device(configured_device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------

def train_one_epoch(
    model: JEPA,
    dataloader: GPUBatchLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype = torch.float16,
    forward_model: torch.nn.Module | None = None,
) -> tuple[float, float]:
    """Train JEPA for one epoch and return loss and elapsed time."""

    model.train()
    forward_model = forward_model if forward_model is not None else model
    amp_enabled = use_amp and device.type == "cuda"

    # Accumulate on-device; calling .item() every step forces a GPU sync.
    running_loss = torch.zeros((), device=device)
    epoch_start = time.perf_counter()

    for batch in dataloader:
        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        actual_lens = batch["actual_len"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            z_pred, z_target = forward_model(
                context_sequence=context,
                actual_lens=actual_lens,
                target_item=target,
            )
            loss = (
                1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)
            ).mean()

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        ema.update()
        running_loss += loss.detach().float()

    synchronize_cuda(device)

    elapsed = time.perf_counter() - epoch_start
    return running_loss.item() / len(dataloader), elapsed


@torch.no_grad()
def validate(
    model: JEPA,
    validation_loader: GPUBatchLoader,
    device: torch.device,
    num_items: int,
) -> dict[str, float]:
    """Run JEPA recommendation evaluation on the validation set."""

    model.eval()

    return evaluate_jepa(
        model=model,
        loader=validation_loader,
        num_items=num_items,
        device=device,
    )


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------

def train(config: dict) -> dict:
    """Run a complete JEPA training experiment."""

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

    max_seq_len = model_config["max_seq_len"]
    pad_token = model_config.get("pad_token_id", 0)
    batch_size = training_config["batch_size"]
    configured_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", 3)
    ema_decay = training_config.get("ema_decay", 0.996)
    use_compile = training_config.get("compile", False)
    save_checkpoint_enabled = training_config.get("save_checkpoint", True)

    # Unused now (data lives on the GPU); kept in the summary for schema
    # continuity with older runs.
    num_workers = training_config.get("num_workers", 0)
    trim_padding = training_config.get("trim_padding", False)
    bucket_chunk_batches = training_config.get("length_bucketing", 0)

    # -----------------------------------------------------------------
    # Output paths
    # -----------------------------------------------------------------

    experiment_name = output_config["experiment_name"]

    results_dir = Path(
        output_config.get("results_dir")
        or Path("results") / experiment_name
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = None
    best_checkpoint = None
    if save_checkpoint_enabled:
        checkpoint_dir = Path(
            output_config.get("checkpoint_dir")
            or Path("checkpoints") / experiment_name
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_dir / output_config.get(
            "best_checkpoint_name", f"{experiment_name}_best.pt"
        )

    history_path = results_dir / "training_history.csv"
    summary_path = results_dir / "run_summary.json"
    config_path = results_dir / "config.yaml"

    output_config["checkpoint_dir"] = (
        str(checkpoint_dir) if checkpoint_dir else None
    )
    output_config["results_dir"] = str(results_dir)
    output_config["best_checkpoint_name"] = (
        best_checkpoint.name if best_checkpoint else None
    )
    output_config["save_checkpoint"] = save_checkpoint_enabled

    # -----------------------------------------------------------------
    # Setup (measured)
    # -----------------------------------------------------------------

    tracker = MemoryTracker(device)
    tracker.mark("process_start")

    with tracker.phase("setup"):
        train_df = pd.read_parquet(data_config["train_path"])
        validation_df = pd.read_parquet(data_config["validation_path"])

        train_dataset = JEPADataset(
            contexts=train_df["context_sequence"].tolist(),
            targets=train_df["target_item"].tolist(),
            max_seq_len=max_seq_len,
            pad_token=pad_token,
        )
        validation_dataset = JEPADataset(
            contexts=validation_df["context_sequence"].tolist(),
            targets=validation_df["target_item"].tolist(),
            max_seq_len=max_seq_len,
            pad_token=pad_token,
        )

        train_loader = GPUBatchLoader(
            train_dataset,
            batch_size=batch_size,
            device=device,
            shuffle=True,
            drop_last=True,
            trim_padding=trim_padding,
            bucket_chunk_batches=bucket_chunk_batches,
        )
        validation_loader = GPUBatchLoader(
            validation_dataset,
            batch_size=batch_size,
            device=device,
            shuffle=False,
            drop_last=False,
            trim_padding=trim_padding,
        )
        tracker.mark("after_data")

        model = JEPA(
            vocab_size=model_config["vocab_size"],
            max_seq_len=max_seq_len,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            feed_forward_dim=model_config["feed_forward_dim"],
            dropout=model_config["dropout"],
            pad_token_id=pad_token,
        ).to(device)
        tracker.mark("after_model")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config.get("weight_decay", 0.0),
        )

        ema = EMA(
            target_model=model.target_encoder,
            source_model=model.context_encoder,
            decay=ema_decay,
        )

        use_amp = (
            training_config.get("mixed_precision", True)
            and device.type == "cuda"
        )
        amp_dtype_name = training_config.get("amp_dtype", "float16")
        if amp_dtype_name not in ("float16", "bfloat16"):
            raise ValueError(
                f"amp_dtype must be float16 or bfloat16, "
                f"got {amp_dtype_name}."
            )
        amp_dtype = getattr(torch, amp_dtype_name)
        if (
            use_amp
            and amp_dtype is torch.bfloat16
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "bfloat16 was requested but this GPU does not support it."
            )

        # Loss scaling is only needed for float16; a disabled scaler is a
        # no-op.
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=use_amp and amp_dtype is torch.float16,
        )

        # Compile only the training forward pass. ``model`` stays
        # uncompiled so EMA, validation, and checkpointing are unaffected.
        # dynamic=True avoids a recompile for every trimmed length.
        forward_model = (
            torch.compile(model, dynamic=True) if use_compile else model
        )

    tracker.mark("after_setup")

    total_params, trainable_params = count_parameters(model)

    logger.info("Experiment: %s", experiment_name)
    logger.info("Results directory: %s", results_dir)
    logger.info(
        "Checkpoint saving: %s",
        "ENABLED" if save_checkpoint_enabled else "DISABLED",
    )
    logger.info(
        "Training samples: %d | validation samples: %d",
        len(train_dataset),
        len(validation_dataset),
    )
    logger.info(
        "Training batches per epoch: %d | validation batches: %d",
        len(train_loader),
        len(validation_loader),
    )
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # -----------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------

    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict = {}
    best_epoch_record: dict = {}
    epochs_without_improvement = 0
    early_stopped = False

    epoch_history: list[dict] = []

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------

    for epoch in range(1, configured_epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]

        with tracker.phase("train"):
            train_loss, train_time = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                forward_model=forward_model,
            )

        with tracker.phase("validation"):
            validation_start = time.perf_counter()
            metrics = validate(
                model=model,
                validation_loader=validation_loader,
                device=device,
                num_items=model_config["vocab_size"],
            )
            synchronize_cuda(device)
            validation_time = time.perf_counter() - validation_start

        memory_record = tracker.end_epoch(epoch, train_time, validation_time)

        # Selection is on raw NDCG@10 (selection window of 1).
        selection_score = metrics["NDCG@10"]

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "HR@5": metrics["HR@5"],
            "HR@10": metrics["HR@10"],
            "NDCG@5": metrics["NDCG@5"],
            "NDCG@10": metrics["NDCG@10"],
            "NDCG@10_selection": selection_score,
            "lr": epoch_lr,
            "train_time_seconds": train_time,
            "validation_time_seconds": validation_time,
            "epoch_time_seconds": train_time + validation_time,
            **memory_record,
        }
        epoch_history.append(record)
        save_epoch_history(epoch_history, history_path)

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | HR@10=%.4f | NDCG@10=%.4f | "
            "lr=%.2e | train=%.1fs | val=%.1fs | "
            "peak_alloc=%.2f GB | peak_reserved=%.2f GB",
            epoch,
            configured_epochs,
            train_loss,
            metrics["HR@10"],
            metrics["NDCG@10"],
            epoch_lr,
            train_time,
            validation_time,
            record["gpu_epoch_peak_allocated_gb"],
            record["gpu_epoch_peak_reserved_gb"],
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_epoch_record = dict(record)
            epochs_without_improvement = 0

            if save_checkpoint_enabled:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "metrics": metrics,
                        "config": config,
                    },
                    best_checkpoint,
                )
                logger.info("Saved new best checkpoint: %s", best_checkpoint)
            else:
                logger.info(
                    "New best NDCG@10: %.4f (checkpoint saving disabled)",
                    best_score,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stopped = True
            logger.info(
                "Early stopping after %d epochs without validation "
                "NDCG@10 improvement.",
                patience,
            )
            break

    # -----------------------------------------------------------------
    # Run summary
    # -----------------------------------------------------------------

    synchronize_cuda(device)
    tracker.mark("after_training")

    total_wall_seconds = time.perf_counter() - total_start

    summary = build_run_summary(
        model_name=MODEL_NAME,
        experiment_name=experiment_name,
        dataset=DATASET_NAME,
        device=device,
        seed=seed,
        run_start=run_start,
        run_end=datetime.now().isoformat(),
        total_wall_seconds=total_wall_seconds,
        configured_epochs=configured_epochs,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_epoch_loss=best_epoch_record.get("loss"),
        best_selection_score=best_score if best_epoch else None,
        early_stopped=early_stopped,
        training_configuration={
            "batch_size": batch_size,
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config.get("weight_decay", 0.0),
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name if use_amp else None,
            "compile": use_compile,
            "patience": patience,
            "lr_schedule": "constant",
            "warmup_epochs": 0,
            "min_lr_ratio": 0.0,
            "selection_window": 1,
            "save_checkpoint": save_checkpoint_enabled,
            # Model-specific settings.
            "ema_decay": ema_decay,
            "num_workers": num_workers,
            "trim_padding": trim_padding,
            "length_bucketing": bucket_chunk_batches,
        },
        model_configuration=model_config,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        data_info={
            "training_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "training_batches_per_epoch": len(train_loader),
            "validation_batches": len(validation_loader),
        },
        samples_per_epoch=len(train_loader) * batch_size,
        files={
            "checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "epoch_history": str(history_path),
            "run_summary": str(summary_path),
            "config": str(config_path),
        },
    )

    save_run_summary(summary, summary_path)

    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    memory_gpu = summary["memory"]["gpu"]
    logger.info(
        "Training complete. Best epoch: %d | best NDCG@10: %.4f",
        best_epoch,
        best_metrics.get("NDCG@10", float("nan")),
    )
    logger.info(
        "Memory | peak allocated: %.2f GB | peak reserved: %.2f GB | "
        "CPU peak RSS: %.2f GB",
        memory_gpu["peak_allocated_gb"],
        memory_gpu["peak_reserved_gb"],
        summary["memory"]["cpu"]["peak_rss_gb"],
    )
    logger.info("Results directory: %s", results_dir)

    return summary


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Train a JEPA sequential recommender."
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
    train(config)


if __name__ == "__main__":
    main()