"""
The GPU evaluation path must give the same metrics as the original slow
implementation, including when a user has seen almost every candidate.

Run with:  pytest tests/test_fast_eval.py
"""

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.evaluate import evaluate_jepa, evaluate_jepa_reference
from src.models.jepa import JEPA

METRICS = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")


def _batches(num_items, max_len, hist_lens, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)

    n = len(hist_lens)
    contexts = torch.zeros(n, max_len, dtype=torch.long)

    for row, length in enumerate(hist_lens):
        contexts[row, :length] = torch.randint(
            2, num_items, (length,), generator=generator
        )

    lens = torch.tensor(hist_lens, dtype=torch.long)
    targets = torch.randint(2, num_items, (n,), generator=generator)

    return [
        {
            "context": contexts[i : i + batch_size],
            "actual_len": lens[i : i + batch_size],
            "target": targets[i : i + batch_size],
        }
        for i in range(0, n, batch_size)
    ]


def _model(num_items, max_len, causal, scoring):
    torch.manual_seed(0)

    model = JEPA(
        vocab_size=num_items,
        max_seq_len=max_len,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        feed_forward_dim=32,
        dropout=0.0,
        causal=causal,
    )
    model.scoring = scoring
    return model.eval()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("scoring", ["ema_cosine", "tied_cosine"])
@pytest.mark.parametrize("exclude_seen", [True, False])
def test_fast_matches_reference(causal, scoring, exclude_seen):
    num_items, max_len = 60, 24
    hist_lens = [1, 2, 5, 12, 24, 24, 3, 8, 17, 1, 24, 6]

    batches = _batches(num_items, max_len, hist_lens, batch_size=5, seed=1)
    model = _model(num_items, max_len, causal, scoring)
    device = torch.device("cpu")

    kwargs = dict(
        model=model,
        loader=batches,
        num_items=num_items,
        device=device,
        exclude_seen=exclude_seen,
    )

    expected = evaluate_jepa_reference(**kwargs)
    actual = evaluate_jepa(**kwargs)

    for name in METRICS:
        assert actual[name] == pytest.approx(expected[name], abs=1e-6), name


def test_users_who_have_seen_almost_everything():
    # 10 candidates (ids 2..11); histories cover most of them, so fewer
    # than k unseen items remain and the gap must be filled with -1.
    num_items, max_len = 12, 24
    hist_lens = [24, 24, 24, 24, 24, 20, 18, 3]

    batches = _batches(num_items, max_len, hist_lens, batch_size=4, seed=3)
    model = _model(num_items, max_len, causal=True, scoring="ema_cosine")
    device = torch.device("cpu")

    kwargs = dict(
        model=model,
        loader=batches,
        num_items=num_items,
        device=device,
    )

    expected = evaluate_jepa_reference(**kwargs)
    actual = evaluate_jepa(**kwargs)

    for name in METRICS:
        assert actual[name] == pytest.approx(expected[name], abs=1e-6), name


def test_fewer_candidates_than_k():
    # 4 candidates but k_values asks for 10: must not crash.
    num_items, max_len = 6, 8
    hist_lens = [1, 2, 3, 8]

    batches = _batches(num_items, max_len, hist_lens, batch_size=2, seed=4)
    model = _model(num_items, max_len, causal=True, scoring="ema_cosine")
    device = torch.device("cpu")

    kwargs = dict(
        model=model,
        loader=batches,
        num_items=num_items,
        device=device,
    )

    expected = evaluate_jepa_reference(**kwargs)
    actual = evaluate_jepa(**kwargs)

    for name in METRICS:
        assert actual[name] == pytest.approx(expected[name], abs=1e-6), name
