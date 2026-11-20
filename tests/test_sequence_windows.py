"""Tests for window construction used by one-pass (causal) JEPA training."""

import numpy as np
import pandas as pd
import pytest

from src.data.sequence_windows import build_windows, reconstruct_train_sequences

MAX_SEQ_LEN = 200


def make_pairs(sequences, max_seq_len=MAX_SEQ_LEN):
    """Build a pair file the way ``generate_jepa_datasets`` does."""

    rows = []

    for user_id, sequence in enumerate(sequences):
        if len(sequence) < 2:
            continue

        for i in range(1, len(sequence)):
            rows.append(
                {
                    "user_id": user_id,
                    "context_sequence": np.array(sequence[:i][-max_seq_len:]),
                    "target_item": sequence[i],
                }
            )

    return pd.DataFrame(rows)


def random_sequences(rng, lengths):
    return [list(rng.integers(2, 3418, size=n)) for n in lengths]


def test_reconstruction_roundtrip():
    rng = np.random.default_rng(0)
    sequences = random_sequences(rng, [2, 5, 199, 200, 201, 202, 450, 1000])

    reconstructed = reconstruct_train_sequences(make_pairs(sequences))

    assert [list(map(int, s)) for s in sequences] == reconstructed


def test_every_target_is_supervised_exactly_once():
    rng = np.random.default_rng(1)
    lengths = [2, 3, 50, 199, 200, 201, 202, 250, 299, 300, 301, 450, 1000, 2314]
    sequences = random_sequences(rng, lengths)

    for stride in (1, 50, 100, 150, 200):
        windows = build_windows(sequences, max_seq_len=MAX_SEQ_LEN, stride=stride)

        windows_seen = {}
        covered = {user: [] for user in range(len(sequences))}

        for row in range(len(windows["inputs"])):
            user = windows["user_index"][row]
            sequence = np.asarray(sequences[user])

            # Windows of one user appear in order; window k starts at k * stride.
            k = windows_seen.get(user, 0)
            start = k * stride
            windows_seen[user] = k + 1

            # Inputs are the sequence slice, right-padded with zeros.
            n_inputs = min(MAX_SEQ_LEN, len(sequence) - start)
            assert (
                windows["inputs"][row, :n_inputs]
                == sequence[start:start + n_inputs]
            ).all()
            assert (windows["inputs"][row, n_inputs:] == 0).all()

            # Each target is the item right after its input position.
            for j in np.flatnonzero(windows["mask"][row]):
                assert windows["targets"][row, j] == sequence[start + j + 1]
                covered[user].append(start + j + 1)

        for user, sequence in enumerate(sequences):
            assert sorted(covered[user]) == list(range(1, len(sequence))), (
                stride,
                len(sequence),
            )


def test_total_targets_match_the_pair_file():
    rng = np.random.default_rng(2)
    sequences = random_sequences(rng, rng.integers(2, 700, size=300))
    pairs = make_pairs(sequences)

    windows = build_windows(reconstruct_train_sequences(pairs), stride=100)

    assert int(windows["mask"].sum()) == len(pairs)


def test_later_windows_only_supervise_positions_with_enough_context():
    rng = np.random.default_rng(3)
    sequences = random_sequences(rng, [1000])

    windows = build_windows(sequences, stride=100)

    for row in range(1, len(windows["inputs"])):
        assert np.flatnonzero(windows["mask"][row]).min() >= 100


def test_window_count_is_small():
    rng = np.random.default_rng(4)
    sequences = random_sequences(rng, [150] * 100 + [1000] * 10)

    windows = build_windows(sequences, stride=100)

    assert len(windows["inputs"]) == 100 * 1 + 10 * 9


def test_shuffled_pair_file_is_rejected():
    rng = np.random.default_rng(5)
    pairs = make_pairs(random_sequences(rng, [30, 40]))

    with pytest.raises(ValueError):
        reconstruct_train_sequences(pairs.sample(frac=1.0, random_state=0))
