"""Training pipeline for BERT4Rec.

Normal training:
    checkpoints/<dataset>/<experiment>/  best model checkpoint
    results/<dataset>/<experiment>/      history, summary, resolved config

Hyperparameter search:
    ``train_bert4rec(config_dict, save_artifacts=False)`` writes no per-trial
    checkpoint, history, summary or config files.

Model selection uses the raw validation NDCG@10.
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
from src.models.factory import build_bert4rec
from src.training.common import (
    RunPaths,
    build_hardware_summary,
    build_timing_summary,
    count_parameters,
    get_peak_gpu_memory_gb,
    log_hardware,
    prepare_run_paths,
    reset_peak_gpu_memory,
    resolve_device,
    save_epoch_history,
    save_run_summary,
    select_best_metrics,
    synchronize_cuda,
)
from src.utils.checkpointing import save_checkpoint
from src.utils.config import (
    add_config_args,
    load_config,
    load_config_from_args,
    save_resolved_config,
)
from src.utils.logging_utils import setup_logging
from src.utils.paths import resolve_output
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------

def _load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and validation frames."""

    data_config = config["data"]

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

    return train_df, val_df


def _build_train_loader(
    config: dict,
    train_df: pd.DataFrame,
    device: torch.device,
) -> tuple[BERT4RecDataset, DataLoader]:
    """Create the masked-item training dataset and its loader."""

    model_config = config["model"]
    training_config = config["training"]

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

    return train_dataset, DataLoader(train_dataset, **loader_kwargs)


def _build_optimizer(
    model: nn.Module,
    training_config: dict,
    device: torch.device,
) -> tuple[AdamW, bool]:
    """Create AdamW, using the fused CUDA kernel when it is available.

    Returns:
        ``(optimizer, fused_adamw_used)``.
    """

    optimizer_kwargs = {
        "lr": training_config["learning_rate"],
        "weight_decay": training_config["weight_decay"],
    }
    fused_adamw_requested = (
        device.type == "cuda"
        and training_config.get("fused_adamw", True)
    )

    if fused_adamw_requested:
        try:
            optimizer = AdamW(
                model.parameters(), **optimizer_kwargs, fused=True,
            )
            return optimizer, True
        except (TypeError, RuntimeError):
            pass

    return AdamW(model.parameters(), **optimizer_kwargs), False


def _resolve_precision(
    training_config: dict,
    device: torch.device,
) -> tuple[bool, torch.dtype, str]:
    """Select mixed precision from ``training.mixed_precision``.

    Returns:
        ``(use_amp, amp_dtype, amp_dtype_name)`` where the name is one of
        ``"bf16"``, ``"fp16"`` or ``"off"``.
    """

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
    amp_dtype_name = "bf16" if use_bf16 else "fp16" if use_fp16 else "off"

    return use_amp, amp_dtype, amp_dtype_name


def _build_scheduler(
    optimizer: AdamW,
    training_config: dict,
) -> tuple[CosineAnnealingLR | None, str | None]:
    """Create the optional cosine-annealing scheduler.

    Returns:
        ``(scheduler, scheduler_type)``; both ``None`` when disabled.
    """

    if not training_config.get("use_scheduler", False):
        return None, None

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

    return scheduler, scheduler_type


# ---------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------

def _train_one_epoch(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    dataset_size: int,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR | None,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Train for one epoch.

    Returns:
        ``(mean_loss_per_sample, train_time_seconds)``.
    """

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
        supervised = labels.ne(IGNORE_INDEX)
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

    return running_loss / dataset_size, train_time


# ---------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------

def _build_summary(
    *,
    config: dict,
    device: torch.device,
    seed: int,
    run_start: str,
    run_end: str,
    num_epochs: int,
    training_history: list[dict],
    best_epoch: int,
    best_ndcg10: float,
    best_metrics: dict,
    total_time: float,
    total_train_time: float,
    total_validation_time: float,
    parameter_counts: tuple[int, int],
    use_amp: bool,
    amp_dtype_name: str,
    fused_adamw_used: bool,
    scheduler_type: str | None,
    use_scheduler: bool,
    patience: int,
    early_stopping_triggered: bool,
    epochs_without_improvement: int,
    save_checkpoint_enabled: bool,
    paths: RunPaths | None,
    train_dataset: BERT4RecDataset,
    val_df: pd.DataFrame,
    train_loader: DataLoader,
) -> dict:
    """Assemble the ``run_summary.json`` contents."""

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    total_params, trainable_params = parameter_counts

    best_record = training_history[best_epoch - 1] if best_epoch > 0 else {}

    return {
        "model": "BERT4Rec",
        "experiment_name": config["output"]["experiment_name"],
        "dataset": config.get(
            "dataset",
            data_config.get("dataset_name", "unknown"),
        ),
        "run_start": run_start,
        "run_end": run_end,
        "device": str(device),
        "seed": seed,
        "configured_epochs": num_epochs,
        "completed_epochs": len(training_history),
        "best_epoch": best_epoch,
        "best_metrics": select_best_metrics(best_metrics),
        "best_epoch_loss": best_record.get("loss"),
        # BERT4Rec selects on raw NDCG@10, not a smoothed score.
        "best_selection_score": best_ndcg10 if best_epoch else None,
        "timing": build_timing_summary(
            training_history,
            best_epoch=best_epoch,
            total_time=total_time,
            total_train_time=total_train_time,
            total_validation_time=total_validation_time,
        ),
        "hardware": build_hardware_summary(device, training_history),
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "training_configuration": {
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config["weight_decay"],
            "num_workers": training_config.get("num_workers", 0),
            "mixed_precision": use_amp,
            "amp_dtype": amp_dtype_name,
            "fused_adamw": fused_adamw_used,
            "use_scheduler": use_scheduler,
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
                str(paths.best_checkpoint)
                if paths is not None and paths.best_checkpoint is not None
                else None
            ),
            "epoch_history": str(paths.history) if paths is not None else None,
            "run_summary": str(paths.summary) if paths is not None else None,
            "config": str(paths.config) if paths is not None else None,
        },
    }


# ---------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------

def train_bert4rec(
    config_path: str | Path | dict,
    *,
    save_artifacts: bool = True,
) -> dict:
    """Train BERT4Rec from a YAML path or an in-memory configuration.

    ``save_artifacts=False`` disables all per-trial file output for
    hyperparameter search. The caller remains responsible for aggregate
    search reports.
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

    model_config = config["model"]
    training_config = config["training"]
    output_config = config["output"]

    # ----- data, model, optimization -----
    train_df, val_df = _load_data(config)
    train_dataset, train_loader = _build_train_loader(config, train_df, device)

    model = build_bert4rec(model_config).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation users: %d", len(val_df))
    logger.info("Training batches per epoch: %d", len(train_loader))
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    optimizer, fused_adamw_used = _build_optimizer(
        model, training_config, device,
    )

    use_amp, amp_dtype, amp_dtype_name = _resolve_precision(
        training_config, device,
    )
    logger.info("Mixed precision: %s", amp_dtype_name)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scheduler, scheduler_type = _build_scheduler(optimizer, training_config)

    # ----- outputs -----
    experiment_name = output_config["experiment_name"]
    save_checkpoint_enabled = (
        save_artifacts and training_config.get("save_checkpoint", True)
    )

    if save_artifacts:
        paths = prepare_run_paths(
            output_config,
            save_checkpoint=save_checkpoint_enabled,
            experiment_name=experiment_name,
        )
        output_config["checkpoint_dir"] = (
            str(paths.checkpoint_dir)
            if paths.checkpoint_dir is not None else None
        )
        output_config["results_dir"] = str(paths.results_dir)
        output_config["save_checkpoint"] = save_checkpoint_enabled
        save_resolved_config(config, paths.results_dir)
    else:
        paths = None

    best_checkpoint_path = paths.best_checkpoint if paths is not None else None

    # ----- training loop -----
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

        train_loss, train_time = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            dataset_size=len(train_dataset),
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        total_train_time += train_time

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
            save_epoch_history(training_history, paths.history)

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

    summary = _build_summary(
        config=config,
        device=device,
        seed=seed,
        run_start=run_start,
        run_end=run_end,
        num_epochs=num_epochs,
        training_history=training_history,
        best_epoch=best_epoch,
        best_ndcg10=best_ndcg10,
        best_metrics=best_metrics,
        total_time=total_time,
        total_train_time=total_train_time,
        total_validation_time=total_validation_time,
        parameter_counts=(total_params, trainable_params),
        use_amp=use_amp,
        amp_dtype_name=amp_dtype_name,
        fused_adamw_used=fused_adamw_used,
        scheduler_type=scheduler_type,
        use_scheduler=scheduler is not None,
        patience=patience,
        early_stopping_triggered=early_stopping_triggered,
        epochs_without_improvement=epochs_without_improvement,
        save_checkpoint_enabled=save_checkpoint_enabled,
        paths=paths,
        train_dataset=train_dataset,
        val_df=val_df,
        train_loader=train_loader,
    )

    if save_artifacts:
        save_run_summary(summary, paths.summary)

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
        "results_path": str(paths.results_dir) if paths is not None else None,
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
