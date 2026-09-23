"""
PyTorch datasets for sequential recommendation models.
"""

import torch
from torch.utils.data import Dataset


class BERT4RecDataset(Dataset):
    """Dataset for BERT4Rec masked-item training."""

    def __init__(
        self,
        sequences,
        max_seq_len=200,
        mask_token=1,
        mask_probability=0.15,
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = list(self.sequences[index])

        # Keep the most recent interactions.
        sequence = sequence[-self.max_seq_len:]

        input_sequence = sequence.copy()
        labels = [-100] * len(sequence)

        for i in range(len(sequence)):
            if torch.rand(1).item() < self.mask_probability:
                labels[i] = sequence[i]
                input_sequence[i] = self.mask_token

        # Pad to max sequence length.
        padding_length = (
            self.max_seq_len - len(input_sequence)
        )

        input_sequence = (
            [0] * padding_length
            + input_sequence
        )

        labels = (
            [-100] * padding_length
            + labels
        )

        return {
            "input_ids": torch.tensor(
                input_sequence,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


class JEPADataset(Dataset):
    """Dataset for JEPA context-target prediction."""

    def __init__(
        self,
        contexts,
        targets,
        max_seq_len=200,
    ):
        self.contexts = contexts
        self.targets = targets
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        context = list(self.contexts[index])
        target = self.targets[index]

        context = context[-self.max_seq_len:]

        padding_length = (
            self.max_seq_len - len(context)
        )

        context = (
            [0] * padding_length
            + context
        )

        return {
            "context": torch.tensor(
                context,
                dtype=torch.long,
            ),
            "target": torch.tensor(
                target,
                dtype=torch.long,
            ),
        }