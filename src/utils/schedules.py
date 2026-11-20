"""Learning-rate schedule and model-selection helpers.

Pure Python (no PyTorch) so they can be unit-tested anywhere.
"""

from __future__ import annotations

import math


def lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int = 0,
    min_ratio: float = 0.0,
    kind: str = "constant",
) -> float:
    """Return the multiplier applied to the base learning rate.

    Args:
        step: Current optimizer step (zero-based).
        total_steps: Total number of optimizer steps in the run.
        warmup_steps: Length of the linear warmup, in steps.
        min_ratio: Cosine floor as a fraction of the base learning rate.
        kind: ``"constant"`` holds the base rate after warmup;
            ``"cosine"`` decays it to ``min_ratio * base`` over
            ``total_steps``.
    """

    if kind not in ("constant", "cosine"):
        raise ValueError(
            f"lr_schedule must be 'constant' or 'cosine', got {kind!r}."
        )

    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_ratio}.")

    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps

    if kind == "constant":
        return 1.0

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    return (
        min_ratio
        + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def trailing_mean(values: list[float], window: int) -> float:
    """Return the mean of the last ``window`` values (or all, if fewer)."""

    if not values:
        raise ValueError("values must not be empty.")

    tail = values[-max(1, window):]

    return sum(tail) / len(tail)
