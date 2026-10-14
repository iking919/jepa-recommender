"""
GPU-resident batch loader for JEPA context-target pairs.

The JEPA training pairs are already fixed-size tensors, so the usual
``DataLoader`` path (per-sample ``__getitem__`` + Python collate) is pure
overhead. This loader keeps the whole dataset on the target device and
builds each batch with a single index_select.

It yields the same ``{"context", "actual_len", "target"}`` dicts as
``DataLoader(JEPADataset(...))``, so ``evaluate_jepa`` works unchanged.
"""

from __future__ import annotations

import math
from typing import Iterator

import torch

from src.data.datasets import JEPADataset


class GPUBatchLoader:
    """Iterate over a ``JEPADataset`` held entirely on one device.

    Args:
        dataset: Prepared ``JEPADataset`` (right-padded contexts).
        batch_size: Samples per batch.
        device: Device the data is stored on and batches are returned on.
        shuffle: Reshuffle every epoch (training).
        drop_last: Drop the final incomplete batch.
        trim_padding: Slice each batch to its longest valid context
            (rounded up to a multiple of 8 for tensor-core/attention-kernel
            alignment). Contexts are right-padded, so positions and padding
            masks are unchanged and the result is mathematically identical.
            Off by default: with random batches the longest context is
            usually near max_seq_len, so the saving can be small.
        bucket_chunk_batches: If > 1 (and ``shuffle``), sort samples by
            length inside chunks of ``batch_size * bucket_chunk_batches``
            so batches have similar lengths and trim more padding. Batch
            order is reshuffled. 0 disables bucketing.
    """

    def __init__(
        self,
        dataset: JEPADataset,
        batch_size: int,
        device: torch.device,
        shuffle: bool = False,
        drop_last: bool = False,
        trim_padding: bool = False,
        bucket_chunk_batches: int = 0,
    ) -> None:
        if int(dataset.contexts.max()) > torch.iinfo(torch.int16).max:
            raise ValueError("Item IDs exceed int16 range; use int32 storage.")

        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.trim_padding = trim_padding
        self.bucket_chunk_batches = bucket_chunk_batches
        self.num_samples = len(dataset)

        # Contexts dominate memory, so store them as int16 and widen per batch.
        self.contexts = dataset.contexts.to(torch.int16).to(device)
        self.actual_lens = dataset.actual_lens.to(device)
        self.targets = dataset.targets.to(device)

        # CPU copy so per-batch widths never force a GPU sync.
        self._lens_cpu = dataset.actual_lens.clone()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return math.ceil(self.num_samples / self.batch_size)

    def _batches(self) -> list[torch.Tensor]:
        n = self.num_samples

        if self.shuffle:
            order = torch.randperm(n)
        else:
            order = torch.arange(n)

        if self.drop_last:
            order = order[: (n // self.batch_size) * self.batch_size]

        if self.shuffle and self.bucket_chunk_batches > 1:
            chunk = self.batch_size * self.bucket_chunk_batches
            pieces = []
            for start in range(0, order.numel(), chunk):
                idx = order[start : start + chunk]
                pieces.append(idx[torch.argsort(self._lens_cpu[idx])])
            order = torch.cat(pieces)

        batches = list(order.split(self.batch_size))

        if self.shuffle and self.bucket_chunk_batches > 1:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]

        return batches

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for idx_cpu in self._batches():
            idx = idx_cpu.to(self.device, non_blocking=True)

            context = self.contexts.index_select(0, idx)

            if self.trim_padding:
                width = max(1, int(self._lens_cpu[idx_cpu].max()))
                width = min(context.shape[1], -(-width // 8) * 8)
                context = context[:, :width]

            yield {
                "context": context.long(),
                "actual_len": self.actual_lens.index_select(0, idx),
                "target": self.targets.index_select(0, idx),
            }


class GPUWindowLoader:
    """Iterate over causal-training windows held entirely on one device.

    Takes the arrays produced by ``src.data.sequence_windows.build_windows``
    and yields ``{"input", "target", "mask"}`` batches, each of shape
    ``(batch, max_seq_len)``. ``mask`` marks the positions whose target
    contributes to the loss.
    """

    def __init__(
        self,
        windows: dict,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if int(windows["inputs"].max()) > torch.iinfo(torch.int16).max:
            raise ValueError("Item IDs exceed int16 range; use int32 storage.")

        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.inputs = torch.from_numpy(windows["inputs"]).to(torch.int16).to(device)
        self.targets = torch.from_numpy(windows["targets"]).to(torch.int16).to(device)
        self.mask = torch.from_numpy(windows["mask"]).to(device)

        self.num_windows = self.inputs.shape[0]
        self.num_targets = int(self.mask.sum().item())

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_windows // self.batch_size
        return math.ceil(self.num_windows / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        n = self.num_windows
        order = torch.randperm(n) if self.shuffle else torch.arange(n)

        if self.drop_last:
            order = order[: (n // self.batch_size) * self.batch_size]

        for idx_cpu in order.split(self.batch_size):
            idx = idx_cpu.to(self.device, non_blocking=True)

            yield {
                "input": self.inputs.index_select(0, idx).long(),
                "target": self.targets.index_select(0, idx).long(),
                "mask": self.mask.index_select(0, idx),
            }
