"""
preprocess_amazon_beauty.py

Preprocesses the Amazon Reviews 2023 All Beauty dataset
for the JEPA Recommender project.

This script orchestrates the preprocessing pipeline.

Pipeline:
1. Load raw Amazon Beauty reviews from JSONL.
2. Extract user IDs, parent ASINs, and timestamps.
3. Sort interactions chronologically by user.
4. Filter products with fewer than the minimum number of interactions.
5. Build chronological user sequences and truncate to max_seq_len.
6. Perform a leave-last-two-out train/validation/test split.
7. Create contiguous product IDs:
       PAD = 0
       MASK = 1
       Products start at ID = 2
8. Save shared intermediate preprocessing artifacts.
9. Generate BERT4Rec-specific datasets.
10. Generate JEPA-specific context-target pairs.
11. Save model-specific datasets under data/processed/amazon_beauty/.

Usage:
    python scripts/preprocess_amazon_beauty.py

Expected raw data location:
    data/raw/amazon_beauty/All_Beauty.jsonl

Expected intermediate output:
    data/interim/amazon_beauty/

Expected processed output:
    data/processed/amazon_beauty/
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Project Path Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "amazon_beauty",
    "All_Beauty.jsonl",
)

INTERIM_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "interim",
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
)

MAX_SEQ_LEN = 200
MIN_ITEM_INTERACTIONS = 5

PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
FIRST_ITEM_ID = 2


# ---------------------------------------------------------------------------
# Load Raw Amazon Beauty Dataset
# ---------------------------------------------------------------------------

def load_amazon_beauty(raw_file):
    """
    Load Amazon Beauty reviews from JSONL.

    Only the fields required for sequential recommendation
    are retained.

    parent_asin is used as the product identifier so that
    product variations are grouped under their parent product.

    Timestamps are retained in milliseconds.
    """

    print("\n=== Loading Amazon Beauty Reviews ===")

    if not os.path.isfile(raw_file):
        raise FileNotFoundError(
            f"Amazon Beauty dataset not found: {raw_file}"
        )

    records = []

    invalid_records = 0
    total_lines = 0

    with open(
        raw_file,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            total_lines += 1

            try:
                review = json.loads(line)

                user_id = review.get("user_id")
                item_id = review.get("parent_asin")
                timestamp = review.get("timestamp")

                if (
                    user_id is None
                    or item_id is None
                    or timestamp is None
                ):
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

            if total_lines % 100_000 == 0:
                print(
                    f"Processed {total_lines:,} JSONL records..."
                )

    ratings = pd.DataFrame(records)

    if ratings.empty:
        raise RuntimeError(
            "No valid Amazon Beauty interactions were loaded."
        )

    print(
        f"\nTotal JSONL records: {total_lines:,}"
    )

    print(
        f"Valid interactions: {len(ratings):,}"
    )

    print(
        f"Invalid or incomplete records: {invalid_records:,}"
    )

    print(
        f"Unique users: {ratings['user_id'].nunique():,}"
    )

    print(
        f"Unique products: {ratings['item_id'].nunique():,}"
    )

    return ratings


# ---------------------------------------------------------------------------
# Chronological Sorting
# ---------------------------------------------------------------------------

def sort_ratings_chronologically(ratings):
    """
    Sort interactions chronologically within each user.

    Original JSONL row order is used as a deterministic
    tie-breaker when timestamps are identical.
    """

    print("\n=== Sorting Interactions Chronologically ===")

    ratings = ratings.sort_values(
        by=[
            "user_id",
            "timestamp",
            "_row_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    return ratings


# ---------------------------------------------------------------------------
# Filter Rare Items
# ---------------------------------------------------------------------------

def filter_rare_items(
    ratings,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
):
    """
    Remove products with fewer than the minimum number
    of interactions.

    Filtering is performed using parent ASIN.
    """

    print("\n=== Filtering Rare Products ===")

    item_counts = ratings["item_id"].value_counts()

    valid_items = item_counts[
        item_counts >= min_item_interactions
    ].index

    ratings_filtered = ratings[
        ratings["item_id"].isin(valid_items)
    ].copy()

    print(
        f"Minimum product interactions: "
        f"{min_item_interactions}"
    )

    print(
        f"Products before filtering: "
        f"{ratings['item_id'].nunique():,}"
    )

    print(
        f"Products after filtering: "
        f"{ratings_filtered['item_id'].nunique():,}"
    )

    print(
        f"Interactions before filtering: "
        f"{len(ratings):,}"
    )

    print(
        f"Interactions after filtering: "
        f"{len(ratings_filtered):,}"
    )

    print(
        f"Removed interactions: "
        f"{len(ratings) - len(ratings_filtered):,}"
    )

    return ratings_filtered


# ---------------------------------------------------------------------------
# Build Chronological User Sequences
# ---------------------------------------------------------------------------

def build_user_sequences(
    ratings,
    max_seq_len=MAX_SEQ_LEN,
):
    """
    Generate chronological interaction sequences for each user.

    Sequences are truncated to the most recent max_seq_len
    interactions.

    Users with fewer than three interactions are excluded
    during the subsequent leave-last-two-out split.
    """

    print("\n=== Building User Sequences ===")

    user_sequences = (
        ratings.groupby(
            "user_id",
            sort=False,
        )["item_id"]
        .agg(list)
        .reset_index(name="sequence")
    )

    user_sequences["sequence"] = (
        user_sequences["sequence"].apply(
            lambda sequence: sequence[-max_seq_len:]
        )
    )

    print(
        f"Users before minimum sequence filtering: "
        f"{len(user_sequences):,}"
    )

    print(
        f"Maximum sequence length: {max_seq_len}"
    )

    print(
        f"Average sequence length: "
        f"{user_sequences['sequence'].apply(len).mean():.2f}"
    )

    return user_sequences


# ---------------------------------------------------------------------------
# Leave-Last-Two-Out Split
# ---------------------------------------------------------------------------

def leave_last_two_out(user_sequences):
    """
    Split each user sequence chronologically.

        train_sequence = all interactions except last two
        validation_item = second-to-last interaction
        test_item = last interaction

    Users must have at least three interactions.
    """

    print("\n=== Creating Leave-Last-Two-Out Split ===")

    split_data = []

    for _, row in user_sequences.iterrows():

        sequence = row["sequence"]

        if len(sequence) < 3:
            continue

        split_data.append(
            {
                "user_id": row["user_id"],
                "train_sequence": sequence[:-2],
                "validation_item": sequence[-2],
                "test_item": sequence[-1],
            }
        )

    df_split = pd.DataFrame(split_data)

    print(
        f"Retained users with sequence length >= 3: "
        f"{len(df_split):,}"
    )

    print(
        f"Excluded users: "
        f"{len(user_sequences) - len(df_split):,}"
    )

    return df_split


# ---------------------------------------------------------------------------
# Contiguous Product ID Mapping
# ---------------------------------------------------------------------------

def create_item_mapping(df_split):
    """
    Create contiguous product IDs.

        PAD = 0
        MASK = 1
        Products start at ID = 2

    Mapping is constructed from all retained train,
    validation, and test interactions.

    This ensures validation and test targets have valid IDs.
    """

    print("\n=== Creating Contiguous Product ID Mapping ===")

    unique_items = sorted(
        set(
            item
            for sequence in df_split["train_sequence"]
            for item in sequence
        )
        | set(df_split["validation_item"])
        | set(df_split["test_item"])
    )

    item_map = {
        item: index + FIRST_ITEM_ID
        for index, item in enumerate(unique_items)
    }

    item_mapping_df = pd.DataFrame(
        {
            "original_item_id": unique_items,
            "item_id": [
                item_map[item]
                for item in unique_items
            ],
        }
    )

    print(
        f"Unique products: {len(item_map):,}"
    )

    print(
        f"PAD token: {PAD_TOKEN_ID}"
    )

    print(
        f"MASK token: {MASK_TOKEN_ID}"
    )

    print(
        f"First product ID: {FIRST_ITEM_ID}"
    )

    print(
        f"Vocabulary size: {len(item_map) + FIRST_ITEM_ID:,}"
    )

    return item_map, item_mapping_df


def apply_item_mapping(df_split, item_map):
    """
    Convert original parent ASINs into contiguous integer IDs.
    """

    print("\n=== Applying Product ID Mapping ===")

    df_split = df_split.copy()

    df_split["train_sequence"] = (
        df_split["train_sequence"].apply(
            lambda sequence: [
                item_map[item]
                for item in sequence
            ]
        )
    )

    df_split["validation_item"] = (
        df_split["validation_item"].map(item_map)
    )

    df_split["test_item"] = (
        df_split["test_item"].map(item_map)
    )

    return df_split


# ---------------------------------------------------------------------------
# Save Interim Dataset
# ---------------------------------------------------------------------------

def save_interim_dataset(
    df_split,
    item_mapping_df,
    interim_dir,
):
    """
    Save shared preprocessing artifacts.
    """

    print("\n=== Saving Interim Dataset ===")

    interim_path = os.path.join(
        interim_dir,
        "amazon_beauty",
    )

    os.makedirs(
        interim_path,
        exist_ok=True,
    )

    df_split.to_parquet(
        os.path.join(
            interim_path,
            "interim_sequences.parquet",
        ),
        index=False,
    )

    item_mapping_df.to_csv(
        os.path.join(
            interim_path,
            "item_id_mapping.csv",
        ),
        index=False,
    )

    print(
        f"Interim dataset saved to: {interim_path}"
    )


# ---------------------------------------------------------------------------
# BERT4Rec Dataset Generation
# ---------------------------------------------------------------------------

def generate_bert4rec_datasets(
    df_split,
    processed_dir,
    max_seq_len=MAX_SEQ_LEN,
):
    """
    Generate BERT4Rec-specific training, validation,
    and test datasets.
    """

    print(
        "\n=== Generating BERT4Rec-Specific Datasets ==="
    )

    # Training data.
    bert_train = df_split[
        [
            "user_id",
            "train_sequence",
        ]
    ].copy()

    # Validation data.
    bert_val = df_split[
        [
            "user_id",
            "train_sequence",
            "validation_item",
        ]
    ].copy()

    # Test data includes the validation item as context.
    bert_test = df_split[
        [
            "user_id",
            "train_sequence",
            "validation_item",
            "test_item",
        ]
    ].copy()

    bert_test["test_input"] = bert_test.apply(
        lambda row: (
            list(row["train_sequence"])
            + [row["validation_item"]]
        )[-max_seq_len:],
        axis=1,
    )

    bert_test = bert_test[
        [
            "user_id",
            "test_input",
            "test_item",
        ]
    ]

    bert_path = os.path.join(
        processed_dir,
        "amazon_beauty",
        "bert4rec",
    )

    os.makedirs(
        bert_path,
        exist_ok=True,
    )

    bert_train.to_parquet(
        os.path.join(
            bert_path,
            "train_sequences.parquet",
        ),
        index=False,
    )

    bert_val.to_parquet(
        os.path.join(
            bert_path,
            "validation.parquet",
        ),
        index=False,
    )

    bert_test.to_parquet(
        os.path.join(
            bert_path,
            "test.parquet",
        ),
        index=False,
    )

    print(
        f"BERT4Rec training users: {len(bert_train):,}"
    )

    print(
        f"BERT4Rec validation users: {len(bert_val):,}"
    )

    print(
        f"BERT4Rec test users: {len(bert_test):,}"
    )

    print(
        f"BERT4Rec dataset saved to: {bert_path}"
    )


# ---------------------------------------------------------------------------
# JEPA Dataset Generation
# ---------------------------------------------------------------------------

def generate_jepa_datasets(
    df_split,
    processed_dir,
    max_seq_len=MAX_SEQ_LEN,
):
    """
    Generate JEPA-specific context-target datasets.

    Training pairs:

        context_sequence = interactions before position i
        target_item      = interaction at position i

    Validation:
        Predict the validation item using the complete
        training sequence.

    Test:
        Predict the test item after observing the
        validation item.
    """

    print(
        "\n=== Generating JEPA-Specific Datasets ==="
    )

    jepa_train_data = []

    for _, row in df_split.iterrows():

        user_id = row["user_id"]
        sequence = row["train_sequence"]

        if len(sequence) < 2:
            continue

        for index in range(1, len(sequence)):

            context = sequence[:index][-max_seq_len:]
            target = sequence[index]

            jepa_train_data.append(
                {
                    "user_id": user_id,
                    "context_sequence": context,
                    "target_item": target,
                }
            )

    jepa_train = pd.DataFrame(
        jepa_train_data,
        columns=[
            "user_id",
            "context_sequence",
            "target_item",
        ],
    )

    # Validation.
    jepa_val = df_split[
        [
            "user_id",
            "train_sequence",
            "validation_item",
        ]
    ].copy()

    jepa_val.columns = [
        "user_id",
        "context_sequence",
        "target_item",
    ]

    jepa_val["context_sequence"] = (
        jepa_val["context_sequence"].apply(
            lambda sequence: sequence[-max_seq_len:]
        )
    )

    # Test.
    jepa_test = df_split[
        [
            "user_id",
            "train_sequence",
            "validation_item",
            "test_item",
        ]
    ].copy()

    jepa_test["context_sequence"] = jepa_test.apply(
        lambda row: (
            list(row["train_sequence"])
            + [row["validation_item"]]
        )[-max_seq_len:],
        axis=1,
    )

    jepa_test = jepa_test[
        [
            "user_id",
            "context_sequence",
            "test_item",
        ]
    ]

    jepa_test.columns = [
        "user_id",
        "context_sequence",
        "target_item",
    ]

    jepa_path = os.path.join(
        processed_dir,
        "amazon_beauty",
        "jepa",
    )

    os.makedirs(
        jepa_path,
        exist_ok=True,
    )

    jepa_train.to_parquet(
        os.path.join(
            jepa_path,
            "train_context_target.parquet",
        ),
        index=False,
    )

    jepa_val.to_parquet(
        os.path.join(
            jepa_path,
            "validation.parquet",
        ),
        index=False,
    )

    jepa_test.to_parquet(
        os.path.join(
            jepa_path,
            "test.parquet",
        ),
        index=False,
    )

    print(
        f"JEPA training context-target pairs: "
        f"{len(jepa_train):,}"
    )

    print(
        f"JEPA validation users: {len(jepa_val):,}"
    )

    print(
        f"JEPA test users: {len(jepa_test):,}"
    )

    print(
        f"JEPA dataset saved to: {jepa_path}"
    )


# ---------------------------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    raw_file=RAW_FILE,
    interim_dir=INTERIM_DIR,
    processed_dir=PROCESSED_DIR,
    max_seq_len=MAX_SEQ_LEN,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
):
    """
    Run the complete Amazon Beauty preprocessing pipeline.
    """

    print("=" * 72)
    print("Amazon Reviews 2023 - All Beauty Preprocessing")
    print("=" * 72)

    print(f"Raw data file: {raw_file}")
    print(f"Interim data directory: {interim_dir}")
    print(f"Processed data directory: {processed_dir}")

    print(f"Maximum sequence length: {max_seq_len}")

    print(
        f"Minimum product interactions: "
        f"{min_item_interactions}"
    )

    # ------------------------------------------------------------------
    # Step 1: Load raw data.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 1: Loading Raw Amazon Beauty Dataset ==="
    )

    ratings = load_amazon_beauty(raw_file)

    print(
        f"Loaded {len(ratings):,} interactions."
    )

    # ------------------------------------------------------------------
    # Step 2: Sort and filter.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 2: Sorting and Filtering Dataset ==="
    )

    ratings = sort_ratings_chronologically(
        ratings
    )

    ratings_filtered = filter_rare_items(
        ratings,
        min_item_interactions,
    )

    # ------------------------------------------------------------------
    # Step 3: Build user sequences.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 3: Generating Chronological User "
        "Sequences and Truncating ==="
    )

    user_sequences = build_user_sequences(
        ratings_filtered,
        max_seq_len,
    )

    # ------------------------------------------------------------------
    # Step 4: Train / validation / test split.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 4: Creating Train/Validation/Test "
        "Split (Leave-Last-Two-Out) ==="
    )

    df_split = leave_last_two_out(
        user_sequences
    )

    # ------------------------------------------------------------------
    # Step 5: Contiguous product ID mapping.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 5: Creating Contiguous Product ID Mapping ==="
    )

    item_map, item_mapping_df = create_item_mapping(
        df_split
    )

    df_split = apply_item_mapping(
        df_split,
        item_map,
    )

    # ------------------------------------------------------------------
    # Save shared intermediate artifacts.
    # ------------------------------------------------------------------

    save_interim_dataset(
        df_split,
        item_mapping_df,
        interim_dir,
    )

    # ------------------------------------------------------------------
    # Step 6: BERT4Rec datasets.
    # ------------------------------------------------------------------

    generate_bert4rec_datasets(
        df_split,
        processed_dir,
        max_seq_len,
    )

    # ------------------------------------------------------------------
    # Step 7: JEPA datasets.
    # ------------------------------------------------------------------

    generate_jepa_datasets(
        df_split,
        processed_dir,
        max_seq_len,
    )

    print("\n" + "=" * 72)
    print(
        "Amazon Beauty preprocessing completed successfully."
    )
    print("=" * 72)


if __name__ == "__main__":
    preprocess_pipeline()
