"""Saving and loading of model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, filepath: str | Path) -> None:
    """Save a training checkpoint, creating its parent directory."""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(filepath: str | Path, map_location=None) -> dict:
    """Load a training checkpoint from disk.

    Checkpoints contain the resolved config and metrics alongside the
    weights, so they are loaded with ``weights_only=False``. Only load
    checkpoints you trust.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    return torch.load(
        filepath,
        map_location=map_location,
        weights_only=False,
    )
