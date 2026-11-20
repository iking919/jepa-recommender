"""Utilities shared by the BERT4Rec, JEPA and causal JEPA trainers.

Model-specific training logic stays in the individual trainer modules. This
module holds the pieces that are identical across them: device handling,
GPU statistics, mixed-precision setup, output paths, and the run-summary
sections.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

METRIC_NAMES = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")


# ---------------------------------------------------------------------
# Device and GPU statistics
# ---------------------------------------------------------------------


def resolve_device(config: dict) -> torch.device:
    """Resolve and validate the configured training device."""
    configured_device = config.get("device")

    if configured_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested in the configuration, but CUDA is not "
            "available in the installed PyTorch environment."
        )

    if configured_device:
        return torch.device(configured_device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize_cuda(device: torch.device) -> None:
    """Synchronize CUDA before recording GPU-dependent timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_gpu_name(device: torch.device) -> str | None:
    """Return the GPU name, or None when running on CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_name(device)


def reset_peak_gpu_memory(device: torch.device) -> None:
    """Reset peak allocated-memory statistics for the current epoch."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_gpu_memory_gb(device: torch.device) -> float:
    """Return peak allocated GPU memory in GiB."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def release_gpu_memory() -> None:
    """Free cached GPU memory between runs, e.g. between search trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_hardware(device: torch.device) -> None:
    """Log the selected device and GPU name when applicable."""
    logger.info("Using device: %s", device)
    gpu_name = get_gpu_name(device)
    if gpu_name:
        logger.info("GPU: %s", gpu_name)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def save_epoch_history(history: list[dict], path: Path) -> None:
    """Write epoch-level training results to CSV."""
    if not history:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_run_summary(summary: dict, path: Path) -> None:
    """Write the final run summary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


# ---------------------------------------------------------------------
# Mixed precision (JEPA trainers)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class AmpSettings:
    """Resolved automatic-mixed-precision configuration."""

    use_amp: bool
    dtype_name: str
    dtype: torch.dtype
    scaler: torch.amp.GradScaler


def resolve_amp_settings(
    training_config: dict,
    device: torch.device,
    default_dtype_name: str,
) -> AmpSettings:
    """Read ``mixed_precision`` and ``amp_dtype`` from the training config.

    A gradient scaler is only enabled for float16; bfloat16 does not need
    one. ``default_dtype_name`` is the trainer's default ``amp_dtype``.
    """

    use_amp = (
        training_config.get("mixed_precision", True)
        and device.type == "cuda"
    )
    dtype_name = training_config.get("amp_dtype", default_dtype_name)

    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError(
            f"amp_dtype must be float16 or bfloat16, got {dtype_name}."
        )

    dtype = getattr(torch, dtype_name)

    if use_amp and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "bfloat16 was requested but this GPU does not support it."
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and dtype is torch.float16,
    )

    return AmpSettings(
        use_amp=use_amp,
        dtype_name=dtype_name,
        dtype=dtype,
        scaler=scaler,
    )


# ---------------------------------------------------------------------
# Output paths (JEPA trainers)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """Where a run writes its history, summary, config and checkpoint."""

    results_dir: Path
    history: Path
    summary: Path
    config: Path
    checkpoint_dir: Path | None
    checkpoint_name: str | None
    best_checkpoint: Path | None


def prepare_run_paths(
    output_config: dict,
    *,
    save_checkpoint: bool,
    experiment_name: str,
) -> RunPaths:
    """Create the output folders and return the run's file paths.

    The checkpoint folder is only created when checkpoint saving is on. The
    checkpoint file name comes from ``output.best_checkpoint_name``, falling
    back to ``<experiment_name>_best.pt``.
    """

    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if save_checkpoint:
        checkpoint_dir = Path(output_config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = (
            output_config.get("best_checkpoint_name")
            or f"{experiment_name}_best.pt"
        )
        best_checkpoint = checkpoint_dir / checkpoint_name
    else:
        checkpoint_dir = None
        checkpoint_name = None
        best_checkpoint = None

    return RunPaths(
        results_dir=results_dir,
        history=results_dir / "training_history.csv",
        summary=results_dir / "run_summary.json",
        config=results_dir / "config.yaml",
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        best_checkpoint=best_checkpoint,
    )


# ---------------------------------------------------------------------
# Run-summary sections
# ---------------------------------------------------------------------

def select_best_metrics(best_metrics: dict) -> dict:
    """Keep the four reported metrics, defaulting missing ones to 0.0."""

    return {name: best_metrics.get(name, 0.0) for name in METRIC_NAMES}


def build_timing_summary(
    history: list[dict],
    *,
    best_epoch: int,
    total_time: float,
    total_train_time: float,
    total_validation_time: float,
) -> dict:
    """Wall-clock statistics for the ``timing`` section of a run summary."""

    completed_epochs = len(history)

    return {
        "total_training_time_seconds": total_time,
        "total_training_time_minutes": total_time / 60.0,
        "total_train_only_seconds": total_train_time,
        "time_to_best_epoch_seconds": sum(
            row["epoch_time_seconds"] for row in history[:best_epoch]
        ),
        "average_epoch_time_seconds": (
            sum(row["epoch_time_seconds"] for row in history)
            / completed_epochs if completed_epochs else 0.0
        ),
        "average_train_time_seconds": (
            total_train_time / completed_epochs if completed_epochs else 0.0
        ),
        "average_validation_time_seconds": (
            total_validation_time / completed_epochs
            if completed_epochs else 0.0
        ),
    }


def build_hardware_summary(device: torch.device, history: list[dict]) -> dict:
    """Device statistics for the ``hardware`` section of a run summary."""

    return {
        "device": str(device),
        "gpu": get_gpu_name(device),
        "peak_gpu_memory_gb": max(
            (row["peak_gpu_memory_gb"] for row in history),
            default=0.0,
        ),
    }
