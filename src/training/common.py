"""
Shared utilities for JEPA, causal JEPA, and BERT4Rec training.

Keep model-specific training logic in the individual training scripts.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import torch

from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


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


def log_hardware(device: torch.device) -> None:
    """Log the selected device and GPU name when applicable."""
    logger.info("Using device: %s", device)
    gpu_name = get_gpu_name(device)
    if gpu_name:
        logger.info("GPU: %s", gpu_name)
