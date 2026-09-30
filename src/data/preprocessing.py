"""
Reusable preprocessing utilities for sequential recommendation.

The preprocessing pipeline loads MovieLens-1M, orders interactions
chronologically, removes rare items, builds user histories, and creates
a contiguous item-ID mapping.

Special tokens:
    0 = PAD
    1 = MASK
    2+ = actual movie IDs
"""

from __future__ import annotations

import pandas as pd


PAD_TOKEN = 0
MASK_TOKEN = 1
ITEM_ID_OFFSET = 2


def load_movielens(raw_dir: str):
    """Load the MovieLens-1M ratings, users, and movie metadata files."""

    ratings = pd.read_csv(
        f"{raw_dir}/ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    users = pd.read_csv(
        f"{raw_dir}/users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )

    movies = pd.read_csv(
        f"{raw_dir}/movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, users, movies


def sort_ratings_chronologically(ratings: pd.DataFrame) -> pd.DataFrame:
    """Sort interactions chronologically within each user."""

    ratings = ratings.copy()
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    return ratings.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)


def filter_rare_items(
    ratings: pd.DataFrame,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Remove items with fewer than ``min_item_interactions`` ratings."""

    item_counts = ratings.groupby("movie_id").size()
    valid_items = item_counts[item_counts >= min_item_interactions].index

    return ratings[ratings["movie_id"].isin(valid_items)].copy()


def build_user_sequences(
    ratings: pd.DataFrame,
    max_seq_len: int = 200,
) -> pd.DataFrame:
    """Build chronological item sequences for each user."""

    sequences = (
        ratings.groupby("user_id")["movie_id"]
        .apply(list)
        .reset_index(name="sequence")
    )

    sequences["sequence"] = sequences["sequence"].apply(
        lambda sequence: sequence[-max_seq_len:]
    )
    sequences["sequence_length"] = sequences["sequence"].apply(len)

    return sequences


def create_movie_mapping(
    df_split: pd.DataFrame,
    movies: pd.DataFrame,
):
    """
    Create contiguous model IDs for all items appearing in the split.

    IDs 0 and 1 remain reserved for PAD and MASK.
    """

    all_movies = set()

    for sequence in df_split["train_sequence"]:
        all_movies.update(sequence)

    all_movies.update(df_split["validation_item"].tolist())
    all_movies.update(df_split["test_item"].tolist())

    unique_movie_ids = sorted(all_movies)

    movie_map = {
        original_id: index + ITEM_ID_OFFSET
        for index, original_id in enumerate(unique_movie_ids)
    }

    mapping = pd.DataFrame(
        [
            {
                "original_movie_id": original_id,
                "mapped_movie_id": mapped_id,
            }
            for original_id, mapped_id in movie_map.items()
        ]
    )

    mapping = mapping.merge(
        movies[["movie_id", "title"]],
        left_on="original_movie_id",
        right_on="movie_id",
        how="left",
    ).drop(columns=["movie_id"])

    return movie_map, mapping


def apply_movie_mapping(
    df_split: pd.DataFrame,
    movie_map: dict[int, int],
) -> pd.DataFrame:
    """Convert original movie IDs to contiguous model IDs."""

    df_split = df_split.copy()

    df_split["train_sequence"] = df_split["train_sequence"].apply(
        lambda sequence: [movie_map[movie_id] for movie_id in sequence]
    )
    df_split["validation_item"] = df_split["validation_item"].map(movie_map)
    df_split["test_item"] = df_split["test_item"].map(movie_map)

    return df_split
