"""Preprocess the Amazon Beauty 5-core dataset into model-ready datasets.

Source: Stanford SNAP Amazon product reviews, Beauty 5-core subset
(https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
reviews_Beauty_5.json.gz). Fetch it with ``scripts.download_amazon_beauty``.

Usage (from the repository root):
    python -m scripts.preprocess_amazon_beauty

Reads:
    data/raw/amazon_beauty/reviews_Beauty_5.json

Writes:
    data/interim/amazon_beauty/               shared intermediate files
    data/processed/amazon_beauty/bert4rec/    BERT4Rec train/validation/test
    data/processed/amazon_beauty/jepa/        JEPA context-target pairs

Steps: load reviews (``reviewerID``, ``asin``, ``unixReviewTime``), sort
chronologically per user, then the shared pipeline in
``src.data.preprocessing`` (filter rare products, build sequences,
leave-last-two-out split, contiguous item IDs, model-specific files).
"""

from __future__ import annotations

import json
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

logger = logging.getLogger("preprocess_amazon_beauty")

DATASET = "amazon_beauty"
RAW_FILE = RAW_DATA_ROOT / DATASET / "reviews_Beauty_5.json"

PROGRESS_INTERVAL = 100_000


def load_amazon_beauty(raw_file: Path) -> pd.DataFrame:
    """Load the Beauty 5-core reviews (one JSON record per line).

    Field mapping: ``reviewerID`` -> ``user_id``, ``asin`` -> ``item_id``,
    ``unixReviewTime`` -> ``timestamp`` (Unix seconds). Malformed or
    incomplete records are skipped. ``_row_id`` records the line number and
    is used later to break timestamp ties deterministically.
    """

    raw_file = Path(raw_file)

    if not raw_file.is_file():
        raise FileNotFoundError(
            f"Amazon Beauty 5-core dataset not found: {raw_file}"
        )

    records = []
    invalid_records = 0
    total_lines = 0

    with raw_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total_lines += 1

            try:
                review = json.loads(line)

                user_id = review.get("reviewerID")
                item_id = review.get("asin")
                timestamp = review.get("unixReviewTime")

                if user_id is None or item_id is None or timestamp is None:
                    invalid_records += 1
                    continue

                records.append(
                    {
                        "user_id": str(user_id),
                        "item_id": str(item_id),
                        "timestamp": int(timestamp),
                        "_row_id": line_number,
                    }
                )

            except (json.JSONDecodeError, TypeError, ValueError):
                invalid_records += 1

            if total_lines % PROGRESS_INTERVAL == 0:
                logger.info("Processed %s JSON records", f"{total_lines:,}")

    ratings = pd.DataFrame(records)

    if ratings.empty:
        raise RuntimeError("No valid Amazon Beauty interactions were loaded.")

    logger.info("Total JSON records: %s", f"{total_lines:,}")
    logger.info("Valid interactions: %s", f"{len(ratings):,}")
    logger.info("Invalid or incomplete records: %s", f"{invalid_records:,}")
    logger.info("Unique users: %s", f"{ratings['user_id'].nunique():,}")
    logger.info("Unique products: %s", f"{ratings['item_id'].nunique():,}")
    logger.info(
        "Timestamp range: %s - %s",
        ratings["timestamp"].min(),
        ratings["timestamp"].max(),
    )

    return ratings


def sort_ratings_chronologically(ratings: pd.DataFrame) -> pd.DataFrame:
    """Sort by user, then timestamp, then original line order.

    The line order makes the result deterministic when timestamps tie.
    """

    return ratings.sort_values(
        by=["user_id", "timestamp", "_row_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_item_mapping_table(item_map: dict[str, int]) -> pd.DataFrame:
    """Tabulate original (ASIN) and mapped item IDs."""

    original_ids = list(item_map)

    return pd.DataFrame(
        {
            "original_item_id": original_ids,
            "item_id": [item_map[item] for item in original_ids],
        }
    )


def preprocess_pipeline(
    raw_file: Path = RAW_FILE,
    interim_dir: Path = INTERIM_DATA_ROOT,
    processed_dir: Path = PROCESSED_DATA_ROOT,
    max_seq_len: int = MAX_SEQ_LEN,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> None:
    """Run the complete Amazon Beauty preprocessing pipeline."""

    logger.info("Raw data file: %s", raw_file)
    logger.info("Interim data directory: %s", interim_dir)
    logger.info("Processed data directory: %s", processed_dir)

    ratings = load_amazon_beauty(raw_file)
    ratings = sort_ratings_chronologically(ratings)

    df_split = preprocess_ratings(
        ratings,
        item_column="item_id",
        max_seq_len=max_seq_len,
        min_item_interactions=min_item_interactions,
    )

    item_map = build_item_id_map(df_split)
    item_mapping = build_item_mapping_table(item_map)
    df_split = apply_item_id_map(df_split, item_map)

    write_datasets(
        df_split,
        item_mapping,
        dataset=DATASET,
        mapping_filename="item_id_mapping.csv",
        interim_root=interim_dir,
        processed_root=processed_dir,
        max_seq_len=max_seq_len,
    )

    logger.info("Amazon Beauty preprocessing completed.")
    logger.info(
        "Users: %s | products: %s | mean training sequence length: %.2f",
        f"{len(df_split):,}",
        f"{len(item_map):,}",
        df_split["train_sequence"].apply(len).mean(),
    )


def main() -> None:
    setup_logging()
    preprocess_pipeline()


if __name__ == "__main__":
    main()
