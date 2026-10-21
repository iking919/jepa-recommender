"""Tests for window construction. Run: python -m tests.test_sequence_windows"""

import numpy as np
import pandas as pd

from src.data.sequence_windows import build_windows, reconstruct_train_sequences

W = 200


def make_pairs(sequences, max_seq_len=W):
    """Mirror scripts/preprocess_ml1m.py::generate_jepa_datasets."""
    rows = []
    for user_id, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            rows.append(
                {
                    "user_id": user_id,
                    "context_sequence": np.array(seq[:i][-max_seq_len:]),
                    "target_item": seq[i],
                }
            )
    return pd.DataFrame(rows)


def random_sequences(rng, lengths):
    return [list(rng.integers(2, 3418, size=n)) for n in lengths]


def test_reconstruction_roundtrip():
    rng = np.random.default_rng(0)
    seqs = random_sequences(rng, [2, 5, 199, 200, 201, 202, 450, 1000])
    out = reconstruct_train_sequences(make_pairs(seqs))
    assert [list(map(int, s)) for s in seqs] == out


def test_every_target_supervised_exactly_once():
    rng = np.random.default_rng(1)
    lengths = [2, 3, 50, 199, 200, 201, 202, 250, 299, 300, 301, 450, 1000, 2314]
    seqs = random_sequences(rng, lengths)

    for stride in (1, 50, 100, 150, 200):
        win = build_windows(seqs, max_seq_len=W, stride=stride)
        starts_seen = {}

        for row in range(len(win["inputs"])):
            user = win["user_index"][row]
            seq = np.asarray(seqs[user])
            # window start = the position of its first input within the sequence
            k = starts_seen.get(user, 0)
            start = k * stride
            starts_seen[user] = k + 1

            n_in = min(W, len(seq) - start)
            assert (win["inputs"][row, :n_in] == seq[start: start + n_in]).all()
            assert (win["inputs"][row, n_in:] == 0).all()

            for j in np.flatnonzero(win["mask"][row]):
                # target is the item right after the input at position j
                assert win["targets"][row, j] == seq[start + j + 1]

        for user, seq in enumerate(seqs):
            covered = []
            for row in np.flatnonzero(win["user_index"] == user):
                k = len(covered_rows := [r for r in np.flatnonzero(win["user_index"] == user) if r < row])
                start = k * stride
                for j in np.flatnonzero(win["mask"][row]):
                    covered.append(start + j + 1)
            assert sorted(covered) == list(range(1, len(seq))), (stride, len(seq))


def test_total_targets_match_pair_file():
    rng = np.random.default_rng(2)
    seqs = random_sequences(rng, rng.integers(2, 700, size=300))
    pairs = make_pairs(seqs)
    win = build_windows(reconstruct_train_sequences(pairs), stride=100)
    assert int(win["mask"].sum()) == len(pairs)


def test_minimum_context_for_later_windows():
    rng = np.random.default_rng(3)
    seqs = random_sequences(rng, [1000])
    win = build_windows(seqs, stride=100)
    # Later windows only supervise positions with >= 101 items of context.
    for row in range(1, len(win["inputs"])):
        assert np.flatnonzero(win["mask"][row]).min() >= 100


def test_window_count_is_small():
    rng = np.random.default_rng(4)
    seqs = random_sequences(rng, [150] * 100 + [1000] * 10)
    win = build_windows(seqs, stride=100)
    assert len(win["inputs"]) == 100 * 1 + 10 * 9


def test_shuffled_pairs_are_rejected():
    rng = np.random.default_rng(5)
    pairs = make_pairs(random_sequences(rng, [30, 40]))
    try:
        reconstruct_train_sequences(pairs.sample(frac=1.0, random_state=0))
    except ValueError:
        return
    raise AssertionError("shuffled pair file should have been rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
