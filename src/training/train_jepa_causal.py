"""One-pass (causal) JEPA training.

A causal context encoder predicts the next item at every position of a
window in a single forward pass, so each user's history is encoded once per
window instead of once per target. Model selection uses a trailing mean of
the validation NDCG@10 (``training.selection_window``).

Checkpoint saving is enabled by default. Disable it with
``training.save_checkpoint: false``. Training histories, resolved
configurations and run summaries are always saved.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F

from src.data.datasets import JEPADataset
from src.data.gpu_loader import GPUBatchLoader, GPUWindowLoader
from src.data.sequence_windows import build_windows, reconstruct_train_sequences
from src.data.tokens import PAD_TOKEN_ID
from src.evaluation.evaluate import evaluate_jepa
from src.models.factory import build_jepa
from src.models.jepa import JEPA
from src.training.common import (
    AmpSettings,
    RunPaths,
    build_hardware_summary,
    build_timing_summary,
    count_parameters,
    get_peak_gpu_memory_gb,
    log_hardware,
    prepare_run_paths,
    reset_peak_gpu_memory,
    resolve_amp_settings,
    resolve_device,
    save_epoch_history,
    save_run_summary,
    select_best_metrics,
    synchronize_cuda,
)
from src.training.ema import EMA
from src.utils.checkpointing import save_checkpoint
from src.utils.config import (
    add_config_args,
    load_config_from_args,
    save_resolved_config,
)
from src.utils.logging_utils import setup_logging
from src.utils.paths import resolve_output
from src.utils.reproducibility import set_seed
from src.utils.schedules import lr_factor, trailing_mean

logger = logging.getLogger(__name__)


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
            z_pred, z_target = forward_fn(batch["input"], batch["target"])

        cosine = F.cosine_similarity(z_pred.float(), z_target.float(), dim=-1)
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

        ema.update()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach().float()

    synchronize_cuda(device)
    elapsed = time.perf_counter() - epoch_start
    return running_loss.item() / len(dataloader), elapsed


def validate(
    model: JEPA,
    validation_loader: GPUBatchLoader,
    device: torch.device,
    num_items: int,
) -> dict[str, float]:
    """Evaluate causal JEPA using the shared JEPA evaluation protocol."""
    model.eval()
    return evaluate_jepa(
        model=model,
        loader=validation_loader,
        num_items=num_items,
        device=device,
    )


def _build_data(
    config: dict,
    device: torch.device,
    stride: int,
) -> tuple[list, GPUWindowLoader, JEPADataset, GPUBatchLoader]:
    """Build the training windows and the validation loader.

    Returns:
        ``(sequences, train_loader, validation_dataset, validation_loader)``.
    """

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    max_seq_len = model_config["max_seq_len"]
    pad_token = model_config.get("pad_token_id", PAD_TOKEN_ID)

    train_df = pd.read_parquet(data_config["train_path"])
    validation_df = pd.read_parquet(data_config["validation_path"])

    sequences = reconstruct_train_sequences(
        train_df, max_seq_len=max_seq_len,
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

    return sequences, train_loader, validation_dataset, validation_loader


def _build_summary(
    *,
    config: dict,
    device: torch.device,
    seed: int,
    run_start: str,
    run_end: str,
    configured_epochs: int,
    epoch_history: list[dict],
    best_epoch: int,
    best_metrics: dict,
    best_epoch_record: dict,
    best_score: float,
    total_time: float,
    total_train_time: float,
    total_validation_time: float,
    parameter_counts: tuple[int, int],
    amp: AmpSettings,
    use_compile: bool,
    ema_decay: float,
    patience: int,
    stride: int,
    lr_schedule: str,
    warmup_epochs: float,
    min_lr_ratio: float,
    selection_window: int,
    paths: RunPaths,
    save_checkpoint_enabled: bool,
    sequences: list,
    train_loader: GPUWindowLoader,
    validation_dataset: JEPADataset,
) -> dict:
    """Assemble the ``run_summary.json`` contents."""

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    total_params, trainable_params = parameter_counts

    return {
        "model": "JEPA (causal one-pass)",
        "experiment_name": config["output"]["experiment_name"],
        "dataset": data_config.get("dataset_name", "unknown"),
        "run_start": run_start,
        "run_end": run_end,
        "device": str(device),
        "seed": seed,
        "configured_epochs": configured_epochs,
        "completed_epochs": len(epoch_history),
        "best_epoch": best_epoch,
        "best_metrics": select_best_metrics(best_metrics),
        "best_epoch_loss": best_epoch_record.get("loss"),
        "best_selection_score": best_score if best_epoch else None,
        "timing": build_timing_summary(
            epoch_history,
            best_epoch=best_epoch,
            total_time=total_time,
            total_train_time=total_train_time,
            total_validation_time=total_validation_time,
        ),
        "hardware": build_hardware_summary(device, epoch_history),
        "model_parameters": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "training_configuration": {
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "weight_decay": training_config.get("weight_decay", 0.0),
            "mixed_precision": amp.use_amp,
            "amp_dtype": amp.dtype_name,
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
                str(paths.best_checkpoint) if paths.best_checkpoint else None
            ),
            "epoch_history": str(paths.history),
            "run_summary": str(paths.summary),
            "config": str(paths.config),
        },
    }


def train(config: dict) -> dict:
    """Run a complete causal one-pass JEPA training experiment."""
    seed = config.get("seed", 42)
    set_seed(seed)

    device = resolve_device(config)
    log_hardware(device)

    model_config = config["model"]
    training_config = config["training"]
    resolve_output(config)
    output_config = config["output"]

    if not model_config.get("causal", False):
        raise ValueError(
            "train_jepa_causal requires model.causal: true so that "
            "the causal evaluation architecture is used."
        )

    max_seq_len = model_config["max_seq_len"]
    stride = training_config.get("window_stride", max_seq_len // 2)
    save_checkpoint_enabled = training_config.get("save_checkpoint", True)

    # ----- data, model, optimization -----
    sequences, train_loader, validation_dataset, validation_loader = (
        _build_data(config, device, stride)
    )

    model = build_jepa(model_config, causal=True).to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(
        "Sequences: %d | windows: %d (stride %d) | supervised targets: %d",
        len(sequences), train_loader.num_windows, stride, train_loader.num_targets,
    )
    logger.info(
        "Training batches per epoch: %d | validation samples: %d",
        len(train_loader), len(validation_dataset),
    )
    logger.info(
        "Model parameters: %s total | %s trainable",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config.get("weight_decay", 0.0),
    )

    ema_decay = training_config.get("ema_decay", 0.996)
    ema = EMA(
        target_model=model.target_encoder,
        source_model=model.context_encoder,
        decay=ema_decay,
    )

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
            step, total_steps, warmup_steps, min_lr_ratio, lr_schedule,
        ),
    )

    if (
        lr_schedule == "cosine"
        and training_config.get("patience", 15) < training_config["num_epochs"]
    ):
        logger.warning(
            "Cosine schedule with patience < num_epochs can stop training "
            "before the learning rate has decayed. Set patience >= num_epochs "
            "to run the full schedule."
        )

    amp = resolve_amp_settings(training_config, device, "bfloat16")

    use_compile = training_config.get("compile", False)
    forward_fn = (
        torch.compile(model.forward_sequence)
        if use_compile else model.forward_sequence
    )

    # ----- outputs -----
    experiment_name = output_config["experiment_name"]
    paths = prepare_run_paths(
        output_config,
        save_checkpoint=save_checkpoint_enabled,
        experiment_name=experiment_name,
    )

    output_config["checkpoint_dir"] = (
        str(paths.checkpoint_dir) if paths.checkpoint_dir is not None else None
    )
    output_config["results_dir"] = str(paths.results_dir)
    output_config["best_checkpoint_name"] = paths.checkpoint_name
    output_config["save_checkpoint"] = save_checkpoint_enabled
    save_resolved_config(config, paths.results_dir)

    logger.info("Experiment: %s", experiment_name)
    logger.info("Results directory: %s", paths.results_dir)
    logger.info(
        "Checkpoint saving: %s",
        "ENABLED" if save_checkpoint_enabled else "DISABLED",
    )

    # ----- training loop -----
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

    for epoch in range(1, configured_epochs + 1):
        reset_peak_gpu_memory(device)
        synchronize_cuda(device)

        train_loss, train_time = train_one_epoch(
            model=model,
            forward_fn=forward_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=amp.scaler,
            ema=ema,
            device=device,
            use_amp=amp.use_amp,
            amp_dtype=amp.dtype,
            scheduler=scheduler,
        )
        total_train_time += train_time

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
        selection_score = trailing_mean(ndcg_history, selection_window)
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
        save_epoch_history(epoch_history, paths.history)

        logger.info(
            "Epoch %03d/%03d | loss=%.4f | HR@10=%.4f | NDCG@10=%.4f | "
            "sel=%.4f | lr=%.2e | train=%.1fs | val=%.1fs | peak_gpu=%.2f GB",
            epoch, configured_epochs, train_loss, metrics["HR@10"],
            metrics["NDCG@10"], selection_score, current_lr,
            train_time, validation_time, peak_gpu_memory_gb,
        )

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
                        "scaler_state_dict": amp.scaler.state_dict(),
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
                    "New best selection score: %.4f (checkpoint saving disabled)",
                    best_score,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping after %d epochs without improvement "
                "in the validation selection score.", patience,
            )
            break

    synchronize_cuda(device)
    total_time = time.perf_counter() - total_start

    summary = _build_summary(
        config=config,
        device=device,
        seed=seed,
        run_start=run_start,
        run_end=datetime.now().isoformat(),
        configured_epochs=configured_epochs,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_epoch_record=best_epoch_record,
        best_score=best_score,
        total_time=total_time,
        total_train_time=total_train_time,
        total_validation_time=total_validation_time,
        parameter_counts=(total_params, trainable_params),
        amp=amp,
        use_compile=use_compile,
        ema_decay=ema_decay,
        patience=patience,
        stride=stride,
        lr_schedule=lr_schedule,
        warmup_epochs=warmup_epochs,
        min_lr_ratio=min_lr_ratio,
        selection_window=selection_window,
        paths=paths,
        save_checkpoint_enabled=save_checkpoint_enabled,
        sequences=sequences,
        train_loader=train_loader,
        validation_dataset=validation_dataset,
    )

    save_run_summary(summary, paths.summary)
    logger.info(
        "Training complete. Best epoch: %d | NDCG@10 at that epoch: %.4f "
        "| selection score: %.4f",
        best_epoch, best_ndcg10, best_score,
    )
    logger.info("Results directory: %s", paths.results_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a causal one-pass JEPA sequential recommender."
    )
    add_config_args(parser)
    args = parser.parse_args()
    setup_logging()
    train(load_config_from_args(args))


if __name__ == "__main__":
    main()
