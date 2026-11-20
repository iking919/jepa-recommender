"""Preprocess MovieLens-1M into model-ready datasets.

Usage (from the repository root):
    python -m scripts.preprocess_ml1m

Reads:
    data/raw/ml-1m/

Writes:
    data/interim/ml-1m/                       shared intermediate files
    data/processed/ml-1m/bert4rec/            BERT4Rec train/validation/test
    data/processed/ml-1m/jepa/                JEPA context-target pairs

Steps: load ratings, sort chronologically per user, then the shared pipeline
in ``src.data.preprocessing`` (filter rare movies, build sequences,
leave-last-two-out split, contiguous item IDs, model-specific files).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.preprocessing import (
    MAX_SEQ_LEN,
    MIN_ITEM_INTERACTIONS,
    apply_item_id_map,
    build_item_id_map,
    preprocess_ratings,
    write_datasets,
)
from src.utils.logging_utils import setup_logging
from src.utils.paths import (
    INTERIM_DATA_ROOT,
    PROCESSED_DATA_ROOT,
    RAW_DATA_ROOT,
)

logger = logging.getLogger("preprocess_ml1m")

DATASET = "ml-1m"
RAW_DIR = RAW_DATA_ROOT / DATASET


def load_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens-1M ratings, users and movie metadata files."""

    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    users = pd.read_csv(
        raw_dir / "users.dat",
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )

    movies = pd.read_csv(
        raw_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    return ratings, users, movies


def sort_ratings_chronologically(ratings: pd.DataFrame) -> pd.DataFrame:
    """Sort interactions by user, then by timestamp."""

    return ratings.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)


def build_movie_mapping_table(
    item_map: dict[int, int],
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Tabulate original and mapped movie IDs, with titles."""

    mapping = pd.DataFrame(
        [
            {"original_movie_id": original_id, "mapped_movie_id": mapped_id}
            for original_id, mapped_id in item_map.items()
        ]
    )

    return mapping.merge(
        movies[["movie_id", "title"]],
        left_on="original_movie_id",
        right_on="movie_id",
        how="left",
    ).drop(columns=["movie_id"])


def preprocess_pipeline(
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DATA_ROOT,
    processed_dir: Path = PROCESSED_DATA_ROOT,
    max_seq_len: int = MAX_SEQ_LEN,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> None:
    """Run the complete MovieLens-1M preprocessing pipeline."""

    logger.info("Raw data directory: %s", raw_dir)
    logger.info("Interim data directory: %s", interim_dir)
    logger.info("Processed data directory: %s", processed_dir)

    ratings, users, movies = load_movielens(Path(raw_dir))
    logger.info(
        "Loaded %s ratings, %s users and %s movies",
        f"{len(ratings):,}",
        f"{len(users):,}",
        f"{len(movies):,}",
    )

    ratings = sort_ratings_chronologically(ratings)
    df_split = preprocess_ratings(
        ratings,
        item_column="movie_id",
        max_seq_len=max_seq_len,
        min_item_interactions=min_item_interactions,
    )

    item_map = build_item_id_map(df_split)
    movie_mapping = build_movie_mapping_table(item_map, movies)
    df_split = apply_item_id_map(df_split, item_map)

    write_datasets(
        df_split,
        movie_mapping,
        dataset=DATASET,
        mapping_filename="movie_id_mapping.csv",
        interim_root=interim_dir,
        processed_root=processed_dir,
        max_seq_len=max_seq_len,
    )

    logger.info("MovieLens-1M preprocessing completed.")


def main() -> None:
    setup_logging()
    preprocess_pipeline()


if __name__ == "__main__":
    main()
