"""Training pipeline for standard (non-causal) JEPA.

Each training example is one (context, target) pair, so a user's history is
re-encoded once per target. Model selection uses the raw validation
NDCG@10. See ``train_jepa_causal`` for the one-pass variant.
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
from src.data.gpu_loader import GPUBatchLoader
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

logger = logging.getLogger(__name__)


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

    # Accumulate on-device to avoid synchronizing the GPU on every batch.
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
            enabled=use_amp and device.type == "cuda",
        ):
            z_pred, z_target = forward_model(
                context_sequence=context,
                actual_lens=actual_lens,
                target_item=target,
            )
            loss = (
                1.0
                - F.cosine_similarity(z_pred, z_target, dim=-1)
            ).mean()

        if use_amp and device.type == "cuda":
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
    average_loss = running_loss.item() / len(dataloader)
    return average_loss, elapsed


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


def _build_data(
    config: dict,
    device: torch.device,
) -> tuple[JEPADataset, JEPADataset, GPUBatchLoader, GPUBatchLoader]:
    """Load the pair files and wrap them in GPU-resident loaders."""

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    train_df = pd.read_parquet(data_config["train_path"])
    validation_df = pd.read_parquet(data_config["validation_path"])

    train_dataset = JEPADataset(
        contexts=train_df["context_sequence"].tolist(),
        targets=train_df["target_item"].tolist(),
        max_seq_len=model_config["max_seq_len"],
        pad_token=model_config.get("pad_token_id", PAD_TOKEN_ID),
    )
    validation_dataset = JEPADataset(
        contexts=validation_df["context_sequence"].tolist(),
        targets=validation_df["target_item"].tolist(),
        max_seq_len=model_config["max_seq_len"],
        pad_token=model_config.get("pad_token_id", PAD_TOKEN_ID),
    )

    batch_size = training_config["batch_size"]
    trim_padding = training_config.get("trim_padding", False)
    bucket_chunk_batches = training_config.get("length_bucketing", 0)

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

    return train_dataset, validation_dataset, train_loader, validation_loader


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
    best_ndcg10: float,
    total_time: float,
    total_train_time: float,
    total_validation_time: float,
    parameter_counts: tuple[int, int],
    amp: AmpSettings,
    use_compile: bool,
    ema_decay: float,
    patience: int,
    paths: RunPaths,
    save_checkpoint_enabled: bool,
    train_dataset: JEPADataset,
    validation_dataset: JEPADataset,
    train_loader: GPUBatchLoader,
    validation_loader: GPUBatchLoader,
) -> dict:
    """Assemble the ``run_summary.json`` contents."""

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    total_params, trainable_params = parameter_counts

    return {
        "model": "JEPA",
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
        "best_selection_score": best_ndcg10 if best_epoch else None,
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
            "num_workers": training_config.get("num_workers", 0),
            "mixed_precision": amp.use_amp,
            "amp_dtype": amp.dtype_name,
            "compile": use_compile,
            "trim_padding": training_config.get("trim_padding", False),
            "length_bucketing": training_config.get("length_bucketing", 0),
            "ema_decay": ema_decay,
            "patience": patience,
            "save_checkpoint": save_checkpoint_enabled,
        },
        "model_configuration": model_config,
        "data": {
            "training_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "training_batches_per_epoch": len(train_loader),
            "validation_batches": len(validation_loader),
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
    """Run a complete standard JEPA training experiment."""
    seed = config.get("seed", 42)
    set_seed(seed)

    device = resolve_device(config)
    log_hardware(device)

    model_config = config["model"]
    training_config = config["training"]
    resolve_output(config)
    output_config = config["output"]

    # ----- data, model, optimization -----
    train_dataset, validation_dataset, train_loader, validation_loader = (
        _build_data(config, device)
    )

    model = build_jepa(model_config, causal=False).to(device)

    total_params, trainable_params = count_parameters(model)

    logger.info(
        "Training samples: %d | validation samples: %d",
        len(train_dataset), len(validation_dataset),
    )
    logger.info(
        "Training batches per epoch: %d | validation batches: %d",
        len(train_loader), len(validation_loader),
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

    amp = resolve_amp_settings(training_config, device, "float16")

    use_compile = training_config.get("compile", False)
    forward_model = (
        torch.compile(model, dynamic=True) if use_compile else model
    )

    # ----- outputs -----
    experiment_name = output_config["experiment_name"]
    save_checkpoint_enabled = training_config.get("save_checkpoint", True)
    paths = prepare_run_paths(
        output_config,
        save_checkpoint=save_checkpoint_enabled,
        experiment_name=experiment_name,
    )

    output_config["checkpoint_dir"] = (
        str(paths.checkpoint_dir) if paths.checkpoint_dir is not None else None
    )
    output_config["results_dir"] = str(paths.results_dir)
    output_config["save_checkpoint"] = save_checkpoint_enabled
    save_resolved_config(config, paths.results_dir)

    # ----- training loop -----
    best_ndcg10 = float("-inf")
    best_epoch = 0
    best_metrics: dict = {}
    best_epoch_record: dict = {}
    epochs_without_improvement = 0
    epoch_history: list[dict] = []

    total_train_time = 0.0
    total_validation_time = 0.0
    configured_epochs = training_config["num_epochs"]
    patience = training_config.get("patience", 3)

    run_start = datetime.now().isoformat()
    total_start = time.perf_counter()

    for epoch in range(1, configured_epochs + 1):
        reset_peak_gpu_memory(device)
        logger.info("Starting epoch %d/%d", epoch, configured_epochs)

        synchronize_cuda(device)
        train_loss, train_time = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=amp.scaler,
            ema=ema,
            device=device,
            use_amp=amp.use_amp,
            amp_dtype=amp.dtype,
            forward_model=forward_model,
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

        record = {
            "epoch": epoch,
            "loss": train_loss,
            "HR@5": metrics["HR@5"],
            "HR@10": metrics["HR@10"],
            "NDCG@5": metrics["NDCG@5"],
            "NDCG@10": metrics["NDCG@10"],
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_seconds": train_time,
            "validation_time_seconds": validation_time,
            "epoch_time_seconds": epoch_time,
            "peak_gpu_memory_gb": peak_gpu_memory_gb,
        }
        epoch_history.append(record)
        save_epoch_history(epoch_history, paths.history)

        logger.info(
            "Epoch %02d/%02d | loss=%.4f | HR@5=%.4f | HR@10=%.4f | "
            "NDCG@5=%.4f | NDCG@10=%.4f | train=%.1fs | val=%.1fs | "
            "epoch=%.1fs | peak_gpu=%.2f GB",
            epoch, configured_epochs, train_loss,
            metrics["HR@5"], metrics["HR@10"],
            metrics["NDCG@5"], metrics["NDCG@10"],
            train_time, validation_time, epoch_time, peak_gpu_memory_gb,
        )

        # Standard JEPA retains raw validation NDCG@10 for checkpoint selection.
        if metrics["NDCG@10"] > best_ndcg10:
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
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping after %d epochs without validation "
                "NDCG@10 improvement.", patience,
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
        configured_epochs=configured_epochs,
        epoch_history=epoch_history,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_epoch_record=best_epoch_record,
        best_ndcg10=best_ndcg10,
        total_time=total_time,
        total_train_time=total_train_time,
        total_validation_time=total_validation_time,
        parameter_counts=(total_params, trainable_params),
        amp=amp,
        use_compile=use_compile,
        ema_decay=ema_decay,
        patience=patience,
        paths=paths,
        save_checkpoint_enabled=save_checkpoint_enabled,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )

    save_run_summary(summary, paths.summary)
    logger.info(
        "Training complete. Best epoch: %d | best NDCG@10: %.4f",
        best_epoch, best_ndcg10,
    )
    logger.info("Results directory: %s", paths.results_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a standard JEPA sequential recommender."
    )
    add_config_args(parser)
    args = parser.parse_args()
    setup_logging()
    train(load_config_from_args(args))


if __name__ == "__main__":
    main()
