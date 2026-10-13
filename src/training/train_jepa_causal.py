
"""
One-pass (causal) JEPA training.

Checkpoint saving is enabled by default. Disable it with:
    training.save_checkpoint: false

Training histories, resolved configurations, and run summaries are always saved.

Default artifacts:
    checkpoints/<experiment>/<experiment>_best.pt  (if enabled)
    results/<experiment>/config.yaml
    results/<experiment>/training_history.csv
    results/<experiment>/run_summary.json

A custom output.results_dir overrides the default results directory.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data.datasets import JEPADataset
from src.data.gpu_loader import GPUBatchLoader, GPUWindowLoader
from src.data.sequence_windows import (
    build_windows,
    reconstruct_train_sequences,
)
from src.models.jepa import JEPA
from src.training.ema import EMA
from src.training.train_jepa import (
    _resolve_device,
    count_parameters,
    get_peak_gpu_memory_gb,
    save_epoch_history,
    save_run_summary,
    synchronize_cuda,
    validate,
)
from src.utils.checkpointing import save_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.overrides import apply_name, apply_overrides
from src.utils.reproducibility import set_seed
from src.utils.schedules import lr_factor, trailing_mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: JEPA,
    forward_fn,
    dataloader: GPUWindowLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    """Train for one epoch over windows; return mean loss and elapsed time."""

    model.train()

    running_loss = torch.zeros((), device=device)
    epoch_start = time.perf_counter()

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=use_amp and device.type == "cuda",
        ):
            z_pred, z_target = forward_fn(
                batch["input"],
                batch["target"],
            )

        cosine = F.cosine_similarity(
            z_pred.float(),
            z_target.float(),
            dim=-1,
        )

        mask = batch["mask"].float()
        loss = (
            ((1.0 - cosine) * mask).sum()
            / mask.sum().clamp(min=1.0)
        )

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update the target encoder using EMA.
        ema.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().float()

    synchronize_cuda(device)

    elapsed = time.perf_counter() - epoch_start
    return running_loss.item() / len(dataloader), elapsed


# ---------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------

def train(config: dict) -> dict:
    """Run a complete causal one-pass JEPA training experiment."""

    # Reproducibility.
    seed = config.get("seed", 42)
    set_seed(seed)

    # Device.
    device = _resolve_device(config)
    logger.info("Using device: %s", device)

    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    # Configuration.
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    output_config = config.setdefault("output", {})

    if not model_config.get("causal", False):
        raise ValueError(
            "train_jepa_causal requires model.causal: true so that "
            "run_jepa_evaluation builds the same architecture."
        )

    max_seq_len = model_config["max_seq_len"]
    pad_token = model_config.get("pad_token_id", 0)
    stride = training_config.get(
        "window_stride",
        max_seq_len // 2,
    )

    # Checkpoint saving is enabled unless explicitly disabled.
    save_checkpoint_enabled = training_config.get(
        "save_checkpoint", True
    )

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------

    train_df = pd.read_parquet(data_config["train_path"])
    validation_df = pd.read_parquet(data_config["validation_path"])

    sequences = reconstruct_train_sequences(
        train_df,
        max_seq_len=max_seq_len,
    )

    windows = build_windows(
        sequences,
        max_seq_len=max_seq_len,
        stride=stride,
        pad_token=pad_token,
    )

    train_loader = GPUWindowLoader(
        windows,
        batch_size=training_config["batch_size"],
        device=device,
        shuffle=True,
        drop_last=False,
    )

    if train_loader.num_targets != len(train_df):
        raise ValueError(
            f"Window targets ({train_loader.num_targets}) do not match "
            f"the pair file ({len(train_df)}); refusing to train."
        )

    if len(train_loader) == 0:
        raise ValueError(
            "batch_size is larger than the number of training windows "
            f"({train_loader.num_windows})."
        )

    validation_dataset = JEPADataset(
        contexts=validation_df["context_sequence"].tolist(),
        targets=validation_df["target_item"].tolist(),
        max_seq_len=max_seq_len,
        pad_token=pad_token,
    )

    validation_loader = GPUBatchLoader(
        validation_dataset,
        batch_size=training_config["batch_size"],
        device=device,
        shuffle=False,
        drop_last=False,
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = JEPA(
        vocab_size=model_config["vocab_size"],
        max_seq_len=max_seq_len,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=pad_token,
        causal=True,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    logger.info(
        "Sequences: %d | windows: %d (stride %d) | "
        "supervised targets: %d",
        len(sequences),
        train_loader.num_windows,
        stride,
        train_loader.num_targets,
    )

    logger.info(
        "Training batches per epoch: %d | validation samples: %d",
        len(train_loader),
        len(validation_dataset),
    )

    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    # -----------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------

    ema_decay = training_config.get("ema_decay", 0.996)

    ema = EMA(
        target_model=model.target_encoder,
        source_model=model.context_encoder,
        decay=ema_decay,
    )

    # -----------------------------------------------------------------
    # Learning-rate schedule
    # -----------------------------------------------------------------

    lr_schedule = training_config.get("lr_schedule", "constant")
    warmup_epochs = training_config.get("warmup_epochs", 0)
    min_lr_ratio = training_config.get("min_lr_ratio", 0.0)
    selection_window = training_config.get("selection_window", 1)

    steps_per_epoch = len(train_loader)
    total_steps = training_config["num_epochs"] * steps_per_epoch
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(
            step,
            total_steps,
            warmup_steps,
            min_lr_ratio,
            lr_schedule,
        ),
    )

    if (
        lr_schedule == "cosine"
        and training_config.get("patience", 15)
        < training_config["num_epochs"]
    ):
        logger.warning(
            "Cosine schedule with patience < num_epochs can stop training "
            "before the learning rate has decayed. Set patience >= "
            "num_epochs to run the full schedule."
        )

    # -----------------------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------------------

    use_amp = (
        training_config.get("mixed_precision", True)
        and device.type == "cuda"
    )

    amp_dtype_name = training_config.get("amp_dtype", "bfloat16")

    if amp_dtype_name not in ("float16", "bfloat16"):
        raise ValueError(
            f"amp_dtype must be float16 or bfloat16, got {amp_dtype_name}."
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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype is torch.float16,
    )

    # -----------------------------------------------------------------
    # Compilation
    # -----------------------------------------------------------------

    use_compile = training_config.get("compile", False)

    forward_fn = (
        torch.compile(model.forward_sequence)
        if use_compile
        else model.forward_sequence
    )

    # -----------------------------------------------------------------
    # Output directories
    # -----------------------------------------------------------------

    experiment_name = output_config["experiment_name"]

    # Honor a custom results directory when one is provided.
    # This allows grid-search trial results to live under:
    # results/jepa_grid_search/<trial_name>/
    configured_results_dir = output_config.get("results_dir")

    if configured_results_dir:
        results_dir = Path(configured_results_dir)
    else:
        results_dir = Path("results") / experiment_name

    # Always preserve experiment results.
    results_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints remain enabled by default for ordinary training.
    # During grid search, training.save_checkpoint=false prevents even
    # creating a checkpoint directory.
    if save_checkpoint_enabled:
        configured_checkpoint_dir = output_config.get("checkpoint_dir")

        if configured_checkpoint_dir:
            checkpoint_dir = Path(configured_checkpoint_dir)
        else:
            checkpoint_dir = Path("checkpoints") / experiment_name

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_name = f"{experiment_name}_best.pt"
        best_checkpoint = checkpoint_dir / best_checkpoint_name
    else:
        checkpoint_dir = None
        best_checkpoint_name = None
        best_checkpoint = None

    history_path = results_dir / "training_history.csv"
    summary_path = results_dir / "run_summary.json"
    config_path = results_dir / "config.yaml"

    # Record resolved output settings in the saved configuration.
    output_config["checkpoint_dir"] = (
        str(checkpoint_dir) if checkpoint_dir is not None else None
    )
    output_config["results_dir"] = str(results_dir)
    output_config["best_checkpoint_name"] = best_checkpoint_name
    output_config["save_checkpoint"] = save_checkpoint_enabled

    logger.info("Experiment: %s", experiment_name)
    logger.info("Results directory: %s", results_dir)
    logger.info(
        "Checkpoint saving: %s",
        "ENABLED" if save_checkpoint_enabled else "DISABLED",
    )

    if save_checkpoint_enabled:
        logger.info("Checkpoint directory: %s", checkpoint_dir)
        logger.info("Best checkpoint: %s", best_checkpoint)

    # -----------------------------------------------------------------
    # Training state
    # -----------------------------------------------------------------

    best_score = float("-inf")
    best_ndcg10 = float("-inf")
    ndcg_history: list[float] = []
    best_epoch = 0
    best_metrics: dict = {}
    best_epoch_record: dict = {}
    epochs_without_improvement = 0

    epoch_history: list[dict] = []
    total_train_time = 0.0
    total_validation_time = 0.0

    configured_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", 15)

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------

    for epoch in range(1, configured_epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_loss, train_time = train_one_epoch(
            model=model,
            forward_fn=forward_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scheduler=scheduler,
        )

        total_train_time += train_time

        # Validation uses the same procedure as standard JEPA.
        validation_start = time.perf_counter()

        metrics = validate(
            model=model,
            validation_loader=validation_loader,
            device=device,
            num_items=model_config["vocab_size"],
        )

        synchronize_cuda(device)

        validation_time = time.perf_counter() - validation_start
        total_validation_time += validation_time

        epoch_time = train_time + validation_time
        peak_gpu_memory_gb = get_peak_gpu_memory_gb(device)

        ndcg_history.append(metrics["NDCG@10"])
        selection_score = trailing_mean(
            ndcg_history,
            selection_window,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "HR@5": metrics["HR@5"],
            "HR@10": metrics["HR@10"],
            "NDCG@5": metrics["NDCG@5"],
            "NDCG@10": metrics["NDCG@10"],
            "NDCG@10_selection": selection_score,
            "lr": current_lr,
            "train_time_seconds": train_time,
            "validation_time_seconds": validation_time,
            "epoch_time_seconds": epoch_time,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }

        epoch_history.append(record)
        save_epoch_history(epoch_history, history_path)

        logger.info(
            "Epoch %03d/%03d | loss=%.4f | HR@10=%.4f | "
            "NDCG@10=%.4f | sel=%.4f | lr=%.2e | "
            "train=%.1fs | val=%.1fs | peak_gpu=%.2f GB",
            epoch,
            configured_epochs,
            train_loss,
            metrics["HR@10"],
            metrics["NDCG@10"],
            selection_score,
            current_lr,
            train_time,
            validation_time,
            peak_gpu_memory_gb,
        )

        # Track the best epoch whether or not a checkpoint is saved.
        if selection_score > best_score:
            best_score = selection_score
            best_ndcg10 = metrics["NDCG@10"]
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

                logger.info(
                    "Saved new best checkpoint: %s",
                    best_checkpoint,
                )
            else:
                logger.info(
                    "New best selection score: %.4f "
                    "(checkpoint saving disabled)",
                    best_score,
                )
        else:
            epochs_without_improvement += 1

        # Early stopping.
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping after %d epochs without improvement "
                "in the validation selection score.",
                patience,
            )
            break

    # -----------------------------------------------------------------
    # Run summary
    # -----------------------------------------------------------------

    synchronize_cuda(device)

    total_time = time.perf_counter() - total_start
    completed_epochs = len(epoch_history)

    summary = {
        "model": "JEPA (causal one-pass)",
        "experiment_name": experiment_name,
        "dataset": "MovieLens-1M",
        "run_start": run_start,
        "run_end": datetime.now().isoformat(),
        "device": str(device),
        "seed": seed,
        "configured_epochs": configured_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_metrics": {
            name: best_metrics.get(name, 0.0)
            for name in ("HR@5", "HR@10", "NDCG@5", "NDCG@10")
        },
        "best_epoch_loss": best_epoch_record.get("loss"),
        "best_selection_score": (
            best_score if best_epoch else None
        ),
        "timing": {
            "total_training_time_seconds": total_time,
            "total_training_time_minutes": total_time / 60.0,
            "total_train_only_seconds": total_train_time,
            "time_to_best_epoch_seconds": sum(
                record["epoch_time_seconds"]
                for record in epoch_history[:best_epoch]
            ),
            "average_epoch_time_seconds": (
                total_time / completed_epochs if completed_epochs else 0.0
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
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
            "peak_gpu_memory_gb": max(
                (
                    record["peak_gpu_memory_gb"]
                    for record in epoch_history
                ),
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
            "weight_decay": training_config.get("weight_decay", 0.0),
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name,
            "compile": use_compile,
            "ema_decay": ema_decay,
            "patience": patience,
            "window_stride": stride,
            "lr_schedule": lr_schedule,
            "warmup_epochs": warmup_epochs,
            "min_lr_ratio": min_lr_ratio,
            "selection_window": selection_window,
            "save_checkpoint": save_checkpoint_enabled,
        },
        "model_configuration": model_config,
        "data": {
            "sequences": len(sequences),
            "training_windows": train_loader.num_windows,
            "supervised_targets_per_epoch": train_loader.num_targets,
            "training_batches_per_epoch": len(train_loader),
            "validation_samples": len(validation_dataset),
        },
        "files": {
            "checkpoint": (
                str(best_checkpoint)
                if best_checkpoint is not None else None
            ),
            "epoch_history": str(history_path),
            "run_summary": str(summary_path),
            "config": str(config_path),
        },
    }

    save_run_summary(summary, summary_path)

    logger.info(
        "Training complete. Best epoch: %d | "
        "NDCG@10 at that epoch: %.4f | selection score: %.4f",
        best_epoch,
        best_ndcg10,
        best_score,
    )
    logger.info("Checkpoint saving: %s", save_checkpoint_enabled)
    logger.info("Results directory: %s", results_dir)

    return summary


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description="Train a causal one-pass JEPA sequential recommender."
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

    config = apply_overrides(
        load_config(args.config),
        args.overrides,
    )

    if args.name:
        config = apply_name(config, args.name)

    setup_logging()
    train(config)


if __name__ == "__main__":
    main()
