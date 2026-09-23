"""
ema.py

Exponential Moving Average (EMA) utilities for the JEPA Recommender project.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average updater for model parameters.

    The EMA model is updated according to:

        theta_target <- decay * theta_target
                        + (1 - decay) * theta_context

    This is used to maintain the slowly evolving target encoder in JEPA.
    """

    def __init__(
        self,
        target_model: nn.Module,
        source_model: nn.Module,
        decay: float = 0.996,
    ):
        if not 0.0 <= decay < 1.0:
            raise ValueError(
                f"EMA decay must be in [0, 1), got {decay}."
            )

        self.target_model = target_model
        self.source_model = source_model
        self.decay = decay

        self._validate_models()

    def _validate_models(self) -> None:
        """Ensure the source and target models have compatible parameters."""

        source_params = list(self.source_model.parameters())
        target_params = list(self.target_model.parameters())

        if len(source_params) != len(target_params):
            raise ValueError(
                "Source and target models must have the same number "
                "of parameters."
            )

        for source_param, target_param in zip(source_params, target_params):
            if source_param.shape != target_param.shape:
                raise ValueError(
                    "Source and target model parameter shapes do not match."
                )

    @torch.no_grad()
    def update(self) -> None:
        """
        Update target-model parameters using the source model.

        The target encoder does not receive gradients through this operation.
        """

        source_params = list(self.source_model.parameters())
        target_params = list(self.target_model.parameters())

        for target_param, source_param in zip(
            target_params,
            source_params,
        ):
            target_param.data.mul_(self.decay)
            target_param.data.add_(
                source_param.data,
                alpha=1.0 - self.decay,
            )

    def state_dict(self) -> dict:
        """Return EMA configuration as a serializable dictionary."""

        return {
            "decay": self.decay,
        }