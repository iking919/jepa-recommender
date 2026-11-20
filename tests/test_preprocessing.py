"""Tests for the shared preprocessing steps."""

import pandas as pd
import pytest

from src.data.preprocessing import (
    apply_item_id_map,
    build_item_id_map,
    build_user_sequences,
    filter_rare_items,
    preprocess_ratings,
)
from src.data.splits import leave_last_two_out
from src.data.tokens import FIRST_ITEM_ID


def make_ratings(rows):
    """Build a ratings frame from ``(user_id, item_id)`` pairs, in order."""

    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def test_filter_rare_items_keeps_items_at_or_above_the_threshold():
    ratings = make_ratings(
        [("u1", "a"), ("u2", "a"), ("u3", "a"), ("u1", "b"), ("u2", "b")]
    )

    filtered = filter_rare_items(ratings, "item_id", min_item_interactions=3)

    assert set(filtered["item_id"]) == {"a"}
    assert len(filtered) == 3


def test_filter_rare_items_preserves_row_order():
    ratings = make_ratings(
        [("u1", "a"), ("u1", "rare"), ("u1", "a"), ("u2", "a")]
    )

    filtered = filter_rare_items(ratings, "item_id", min_item_interactions=2)

    assert filtered["item_id"].tolist() == ["a", "a", "a"]


def test_user_sequences_follow_input_order_and_keep_the_most_recent_items():
    ratings = make_ratings(
        [("u1", 1), ("u1", 2), ("u1", 3), ("u1", 4), ("u2", 9)]
    )

    sequences = build_user_sequences(ratings, "item_id", max_seq_len=3)

    assert sequences["user_id"].tolist() == ["u1", "u2"]
    assert sequences["sequence"].tolist() == [[2, 3, 4], [9]]


def test_leave_last_two_out_splits_and_drops_short_users():
    user_sequences = pd.DataFrame(
        {
            "user_id": ["long", "short"],
            "sequence": [[1, 2, 3, 4, 5], [1, 2]],
        }
    )

    split = leave_last_two_out(user_sequences)

    assert split["user_id"].tolist() == ["long"]
    assert split["train_sequence"].tolist() == [[1, 2, 3]]
    assert split["validation_item"].tolist() == [4]
    assert split["test_item"].tolist() == [5]


def test_leave_last_two_out_returns_the_expected_columns_when_empty():
    user_sequences = pd.DataFrame({"user_id": ["u"], "sequence": [[1]]})

    split = leave_last_two_out(user_sequences)

    assert split.empty
    assert list(split.columns) == [
        "user_id",
        "train_sequence",
        "validation_item",
        "test_item",
    ]


def test_item_ids_are_contiguous_sorted_and_start_after_special_tokens():
    split = pd.DataFrame(
        {
            "user_id": ["u"],
            "train_sequence": [["c", "a"]],
            "validation_item": ["b"],
            "test_item": ["z"],
        }
    )

    item_map = build_item_id_map(split)

    assert item_map == {
        "a": FIRST_ITEM_ID,
        "b": FIRST_ITEM_ID + 1,
        "c": FIRST_ITEM_ID + 2,
        "z": FIRST_ITEM_ID + 3,
    }


def test_apply_item_id_map_rewrites_every_column_without_mutating_input():
    split = pd.DataFrame(
        {
            "user_id": ["u"],
            "train_sequence": [["a", "b"]],
            "validation_item": ["b"],
            "test_item": ["a"],
        }
    )
    item_map = {"a": 2, "b": 3}

    mapped = apply_item_id_map(split, item_map)

    assert mapped["train_sequence"].tolist() == [[2, 3]]
    assert mapped["validation_item"].tolist() == [3]
    assert mapped["test_item"].tolist() == [2]
    assert split["train_sequence"].tolist() == [["a", "b"]]


def test_preprocess_ratings_runs_filter_sequence_and_split():
    rows = [("u1", item) for item in "abcde"]
    rows += [("u2", item) for item in "abcde"]
    rows += [("u3", "a"), ("u3", "b")]
    ratings = make_ratings(rows)

    split = preprocess_ratings(
        ratings,
        item_column="item_id",
        max_seq_len=200,
        min_item_interactions=2,
    )

    # u3 has too few interactions to be split.
    assert split["user_id"].tolist() == ["u1", "u2"]
    assert split["train_sequence"].tolist() == [["a", "b", "c"]] * 2
    assert split["validation_item"].tolist() == ["d", "d"]
    assert split["test_item"].tolist() == ["e", "e"]


def test_preprocess_ratings_raises_when_no_user_can_be_split():
    ratings = make_ratings([("u1", "a"), ("u1", "b"), ("u2", "a"), ("u2", "b")])

    with pytest.raises(RuntimeError):
        preprocess_ratings(
            ratings,
            item_column="item_id",
            min_item_interactions=1,
        )
