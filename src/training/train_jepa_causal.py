"""
One-pass (causal) JEPA training.

Checkpoint saving is enabled by default. Disable it with:
    training.save_checkpoint: false

Training histories, resolved configurations, and run summaries are always
saved. The run summary follows the uniform schema defined in
``src.utils.run_profiling`` so it can be compared directly with the JEPA and
BERT4Rec summaries, including detailed GPU/CPU memory measurements.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
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
from src.utils.run_profiling import (
    MemoryTracker,
    build_run_summary,
    count_parameters,
)
from src.utils.schedules import lr_factor, trailing_mean

logger = logging.getLogger(__name__)

MODEL_NAME = "JEPA (causal one-pass)"
DATASET_NAME = "MovieLens-1M"


# ---------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------

@dataclass
class DataBundle:
    train_loader: GPUWindowLoader
    validation_loader: GPUBatchLoader
    num_sequences: int
    num_validation_samples: int
    stride: int


@dataclass
class OutputPaths:
    results_dir: Path
    history: Path
    summary: Path
    config: Path
    checkpoint_dir: Path | None
    best_checkpoint: Path | None


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------

def build_data(
    config: dict,
    device: torch.device,
) -> DataBundle:
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    max_seq_len = model_config["max_seq_len"]
    pad_token = model_config.get("pad_token_id", 0)
    stride = training_config.get("window_stride", max_seq_len // 2)
    batch_size = training_config["batch_size"]

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
        batch_size=batch_size,
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
        batch_size=batch_size,
        device=device,
        shuffle=False,
        drop_last=False,
    )

    return DataBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_sequences=len(sequences),
        num_validation_samples=len(validation_dataset),
        stride=stride,
    )


def build_model(model_config: dict, device: torch.device) -> JEPA:
    return JEPA(
        vocab_size=model_config["vocab_size"],
        max_seq_len=model_config["max_seq_len"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        feed_forward_dim=model_config["feed_forward_dim"],
        dropout=model_config["dropout"],
        pad_token_id=model_config.get("pad_token_id", 0),
        causal=True,
    ).to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    lr_schedule = training_config.get("lr_schedule", "constant")
    warmup_epochs = training_config.get("warmup_epochs", 0)
    min_lr_ratio = training_config.get("min_lr_ratio", 0.0)

    total_steps = training_config["num_epochs"] * steps_per_epoch
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))

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

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_factor(
            step,
            total_steps,
            warmup_steps,
            min_lr_ratio,
            lr_schedule,
        ),
    )


def resolve_amp(
    training_config: dict,
    device: torch.device,
) -> tuple[bool, str, torch.dtype, torch.amp.GradScaler]:
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
    return use_amp, amp_dtype_name, amp_dtype, scaler


def resolve_output_paths(
    output_config: dict,
    experiment_name: str,
    save_checkpoint_enabled: bool,
) -> OutputPaths:
    """Resolve and create output directories.

    A custom ``output.results_dir`` lets grid-search trials live under
    results/jepa_grid_search/<trial_name>/. With checkpoint saving disabled,
    no checkpoint directory is created.
    """
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
        best_checkpoint = checkpoint_dir / f"{experiment_name}_best.pt"

    return OutputPaths(
        results_dir=results_dir,
        history=results_dir / "training_history.csv",
        summary=results_dir / "run_summary.json",
        config=results_dir / "config.yaml",
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=best_checkpoint,
    )


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

    amp_enabled = use_amp and device.type == "cuda"
    running_loss = torch.zeros((), device=device)
    epoch_start = time.perf_counter()

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            z_pred, z_target = forward_fn(batch["input"], batch["target"])

        cosine = F.cosine_similarity(
            z_pred.float(),
            z_target.float(),
            dim=-1,
        )
        mask = batch["mask"].float()
        loss = ((1.0 - cosine) * mask).sum() / mask.sum().clamp(min=1.0)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        ema.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().float()

    synchronize_cuda(device)

    elapsed = time.perf_counter() - epoch_start
    return running_loss.item() / len(dataloader), elapsed


def train(config: dict) -> dict:
    """Run a complete causal one-pass JEPA training experiment."""

    seed = config.get("seed", 42)
    set_seed(seed)

    device = _resolve_device(config)
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))

    model_config = config["model"]
    training_config = config["training"]
    output_config = config.setdefault("output", {})

    if not model_config.get("causal", False):
        raise ValueError(
            "train_jepa_causal requires model.causal: true so that "
            "run_jepa_evaluation builds the same architecture."
        )

    save_checkpoint_enabled = training_config.get("save_checkpoint", True)
    configured_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", 15)
    selection_window = training_config.get("selection_window", 1)
    ema_decay = training_config.get("ema_decay", 0.996)
    use_compile = training_config.get("compile", False)

    experiment_name = output_config["experiment_name"]
    paths = resolve_output_paths(
        output_config, experiment_name, save_checkpoint_enabled
    )

    # Record resolved output settings in the saved configuration.
    output_config["checkpoint_dir"] = (
        str(paths.checkpoint_dir) if paths.checkpoint_dir else None
    )
    output_config["results_dir"] = str(paths.results_dir)
    output_config["best_checkpoint_name"] = (
        paths.best_checkpoint.name if paths.best_checkpoint else None
    )
    output_config["save_checkpoint"] = save_checkpoint_enabled

    logger.info("Experiment: %s", experiment_name)
    logger.info("Results directory: %s", paths.results_dir)
    logger.info(
        "Checkpoint saving: %s",
        "ENABLED" if save_checkpoint_enabled else "DISABLED",
    )
    if save_checkpoint_enabled:
        logger.info("Best checkpoint: %s", paths.best_checkpoint)

    # -----------------------------------------------------------------
    # Setup (measured)
    # -----------------------------------------------------------------

    tracker = MemoryTracker(device)
    tracker.mark("process_start")

    with tracker.phase("setup"):
        data = build_data(config, device)
        tracker.mark("after_data")

        model = build_model(model_config, device)
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

        train_loader = data.train_loader
        steps_per_epoch = len(train_loader)
        scheduler = build_scheduler(
            optimizer, training_config, steps_per_epoch
        )

        use_amp, amp_dtype_name, amp_dtype, scaler = resolve_amp(
            training_config, device
        )

        forward_fn = (
            torch.compile(model.forward_sequence)
            if use_compile
            else model.forward_sequence
        )

    tracker.mark("after_setup")

    total_params, trainable_params = count_parameters(model)

    logger.info(
        "Sequences: %d | windows: %d (stride %d) | supervised targets: %d",
        data.num_sequences,
        train_loader.num_windows,
        data.stride,
        train_loader.num_targets,
    )
    logger.info(
        "Training batches per epoch: %d | validation samples: %d",
        steps_per_epoch,
        data.num_validation_samples,
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

    ndcg_history: list[float] = []
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

        with tracker.phase("validation"):
            validation_start = time.perf_counter()
            metrics = validate(
                model=model,
                validation_loader=data.validation_loader,
                device=device,
                num_items=model_config["vocab_size"],
            )
            synchronize_cuda(device)
            validation_time = time.perf_counter() - validation_start

        memory_record = tracker.end_epoch(epoch, train_time, validation_time)

        ndcg_history.append(metrics["NDCG@10"])
        selection_score = trailing_mean(ndcg_history, selection_window)

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
        save_epoch_history(epoch_history, paths.history)

        logger.info(
            "Epoch %03d/%03d | loss=%.4f | HR@10=%.4f | NDCG@10=%.4f | "
            "sel=%.4f | lr=%.2e | train=%.1fs | val=%.1fs | "
            "peak_alloc=%.2f GB | peak_reserved=%.2f GB",
            epoch,
            configured_epochs,
            train_loss,
            metrics["HR@10"],
            metrics["NDCG@10"],
            selection_score,
            epoch_lr,
            train_time,
            validation_time,
            record["gpu_epoch_peak_allocated_gb"],
            record["gpu_epoch_peak_reserved_gb"],
        )

        # Track the best epoch whether or not a checkpoint is saved.
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
                    paths.best_checkpoint,
                )
                logger.info(
                    "Saved new best checkpoint: %s", paths.best_checkpoint
                )
            else:
                logger.info(
                    "New best selection score: %.4f "
                    "(checkpoint saving disabled)",
                    best_score,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            early_stopped = True
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
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config.get("weight_decay", 0.0),
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name,
            "compile": use_compile,
            "patience": patience,
            "lr_schedule": training_config.get("lr_schedule", "constant"),
            "warmup_epochs": training_config.get("warmup_epochs", 0),
            "min_lr_ratio": training_config.get("min_lr_ratio", 0.0),
            "selection_window": selection_window,
            "save_checkpoint": save_checkpoint_enabled,
            # Model-specific settings.
            "ema_decay": ema_decay,
            "window_stride": data.stride,
        },
        model_configuration=model_config,
        model=model,
        optimizer=optimizer,
        tracker=tracker,
        data_info={
            "sequences": data.num_sequences,
            "training_windows": train_loader.num_windows,
            "supervised_targets_per_epoch": train_loader.num_targets,
            "training_batches_per_epoch": steps_per_epoch,
            "validation_samples": data.num_validation_samples,
        },
        samples_per_epoch=train_loader.num_targets,
        files={
            "checkpoint": (
                str(paths.best_checkpoint) if paths.best_checkpoint else None
            ),
            "epoch_history": str(paths.history),
            "run_summary": str(paths.summary),
            "config": str(paths.config),
        },
    )

    save_run_summary(summary, paths.summary)

    # The resolved config (with output settings filled in) is saved with
    # the run so every result directory is self-describing.
    with open(paths.config, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    memory_gpu = summary["memory"]["gpu"]
    logger.info(
        "Training complete. Best epoch: %d | NDCG@10 at that epoch: %.4f | "
        "selection score: %.4f",
        best_epoch,
        best_metrics.get("NDCG@10", float("nan")),
        best_score,
    )
    logger.info(
        "Memory | peak allocated: %.2f GB | peak reserved: %.2f GB | "
        "CPU peak RSS: %.2f GB",
        memory_gpu["peak_allocated_gb"],
        memory_gpu["peak_reserved_gb"],
        summary["memory"]["cpu"]["peak_rss_gb"],
    )
    logger.info("Results directory: %s", paths.results_dir)

    return summary


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def main() -> None:
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

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.name:
        config = apply_name(config, args.name)

    setup_logging()
    train(config)


if __name__ == "__main__":
    main()
