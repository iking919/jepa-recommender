"""
Trimming / bucketing must not change which targets are supervised.

Run with:  pytest tests/test_window_loader_trim.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.gpu_loader import GPUWindowLoader
from src.data.sequence_windows import build_windows


def _windows():
    rng = np.random.default_rng(0)
    sequences = [
        rng.integers(2, 50, size=int(n)).tolist()
        for n in rng.integers(2, 40, size=37)
    ]
    return build_windows(sequences, max_seq_len=40, stride=20)


def _collect(loader):
    pairs = []
    for batch in loader:
        inputs, targets, mask = batch["input"], batch["target"], batch["mask"]
        assert inputs.shape == targets.shape == mask.shape
        for row in range(inputs.shape[0]):
            for j in torch.nonzero(mask[row]).flatten().tolist():
                pairs.append((int(inputs[row, : j + 1].sum()), int(targets[row, j])))
    return sorted(pairs)


@pytest.mark.parametrize("bucket", [0, 4])
def test_trim_keeps_every_supervised_target(bucket):
    windows = _windows()
    device = torch.device("cpu")

    plain = GPUWindowLoader(windows, 8, device, shuffle=True, drop_last=False)
    trimmed = GPUWindowLoader(
        windows,
        8,
        device,
        shuffle=True,
        drop_last=False,
        trim_padding=True,
        bucket_chunk_batches=bucket,
    )

    assert plain.num_targets == trimmed.num_targets
    assert _collect(plain) == _collect(trimmed)


def test_trim_actually_shrinks_batches():
    windows = _windows()

    trimmed = GPUWindowLoader(
        windows,
        8,
        torch.device("cpu"),
        shuffle=True,
        drop_last=False,
        trim_padding=True,
        bucket_chunk_batches=4,
    )

    widths = [batch["input"].shape[1] for batch in trimmed]

    assert all(w % 8 == 0 or w == 40 for w in widths)
    assert min(widths) < 40
