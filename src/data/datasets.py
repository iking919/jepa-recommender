"""
PyTorch datasets for sequential recommendation.

This module provides the model-specific dataset wrappers used by both
BERT4Rec and JEPA. Preprocessing remains separate so that the training
pipeline consumes already prepared data.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class BERT4RecDataset(Dataset):
    """Dataset for masked-item training with BERT4Rec."""

    def __init__(
        self,
        sequences: Sequence,
        max_seq_len: int = 200,
        mask_token: int = 1,
        mask_probability: float = 0.15,
        pad_token: int = 0,
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.mask_probability = mask_probability
        self.pad_token = pad_token

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = list(self.sequences[index])[-self.max_seq_len:]

        input_sequence = sequence.copy()
        labels = [-100] * len(sequence)

        for position in range(len(sequence)):
            if torch.rand(1).item() < self.mask_probability:
                labels[position] = sequence[position]
                input_sequence[position] = self.mask_token

        padding_length = self.max_seq_len - len(input_sequence)

        input_sequence = (
            [self.pad_token] * padding_length + input_sequence
        )
        labels = [-100] * padding_length + labels

        return {
            "input_ids": torch.tensor(input_sequence, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class JEPADataset(Dataset):
    """
    Dataset for precomputed JEPA context-target pairs.

    Context sequences are right-padded to ``max_seq_len``. ``actual_len``
    records the number of valid items so the context encoder can select
    the final valid hidden state.
    """

    def __init__(
        self,
        contexts: Sequence,
        targets: Sequence,
        max_seq_len: int = 200,
        pad_token: int = 0,
    ) -> None:
        if len(contexts) != len(targets):
            raise ValueError(
                "contexts and targets must contain the same number of samples. "
                f"Got {len(contexts)} contexts and {len(targets)} targets."
            )

        self.max_seq_len = max_seq_len
        self.pad_token = pad_token

        padded_contexts = []
        actual_lengths = []

        for context in contexts:
            context = list(context)[-max_seq_len:]
            actual_len = len(context)

            padded_contexts.append(
                context + [pad_token] * (max_seq_len - actual_len)
            )
            actual_lengths.append(actual_len)

        self.contexts = torch.tensor(padded_contexts, dtype=torch.long)
        self.actual_lens = torch.tensor(actual_lengths, dtype=torch.long)
        self.targets = torch.tensor(list(targets), dtype=torch.long)

        if self.contexts.ndim != 2:
            raise ValueError(
                "JEPA contexts must have shape "
                f"(num_samples, max_seq_len). Got {self.contexts.shape}."
            )

        if self.contexts.shape[0] != len(targets):
            raise ValueError(
                "Number of context rows does not match the number of targets."
            )

        if self.contexts.shape[1] != max_seq_len:
            raise ValueError(
                f"Expected max_seq_len={max_seq_len}, "
                f"got context shape {self.contexts.shape}."
            )

        if self.actual_lens.shape[0] != len(targets):
            raise ValueError(
                "Number of sequence lengths does not match the number of targets."
            )

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "context": self.contexts[index],
            "actual_len": self.actual_lens[index],
            "target": self.targets[index],
        }
