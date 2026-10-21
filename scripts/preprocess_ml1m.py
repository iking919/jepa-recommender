"""
preprocess_ml1m.py

Preprocesses the MovieLens-1M dataset for the JEPA Recommender project.

This script orchestrates the preprocessing pipeline using reusable
functions from src/data/.

Pipeline:
1. Load raw MovieLens-1M ratings, users, and movies.
2. Sort ratings chronologically by user.
3. Filter movies with fewer than the minimum number of interactions.
4. Build chronological user sequences and truncate to max_seq_len.
5. Perform a leave-last-two-out train/validation/test split.
6. Create contiguous movie IDs:
       PAD = 0
       MASK = 1
       Movies start at ID = 2
7. Save shared intermediate preprocessing artifacts.
8. Generate BERT4Rec-specific datasets.
9. Generate JEPA-specific context-target pairs.
10. Save model-specific datasets under data/processed/ml-1m/.

Usage:
    python scripts/preprocess_ml1m.py

Expected raw data location:
    data/raw/ml-1m/

Expected intermediate output:
    data/interim/ml-1m/

Expected processed output:
    data/processed/ml-1m/
"""

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

# Allow imports from the project root.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.data.preprocessing import (
    load_movielens,
    sort_ratings_chronologically,
    filter_rare_items,
    build_user_sequences,
    create_movie_mapping,
    apply_movie_mapping,
)

from src.data.splits import (
    leave_last_two_out,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "ml-1m",
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


# ---------------------------------------------------------------------------
# Save Interim Dataset
# ---------------------------------------------------------------------------

def save_interim_dataset(
    df_split,
    movie_mapping_df,
    interim_dir,
):
    """Save shared preprocessing artifacts."""

    print("\n=== Saving Interim Dataset ===")

    interim_path = os.path.join(
        interim_dir,
        "ml-1m",
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

    movie_mapping_df.to_csv(
        os.path.join(
            interim_path,
            "movie_id_mapping.csv",
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
    Generate BERT4Rec-specific training, validation, and test datasets.
    """

    print(
        "\n=== Step 6: Generating BERT4Rec-Specific Datasets ==="
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

    # Test data uses the validation item as the final observed item.
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
        "ml-1m",
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
        f"BERT4Rec training users: "
        f"{len(bert_train):,}"
    )

    print(
        f"BERT4Rec validation users: "
        f"{len(bert_val):,}"
    )

    print(
        f"BERT4Rec test users: "
        f"{len(bert_test):,}"
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

    Training pairs are generated from the training sequence:

        context_sequence = interactions before position i
        target_item      = interaction at position i
    """

    print(
        "\n=== Step 7: Generating JEPA-Specific Datasets ==="
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
        jepa_train_data
    )

    # Validation:
    # Predict the validation item using the complete training sequence.
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

    # Test:
    # Predict the test item after observing the validation item.
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
        "ml-1m",
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
        f"JEPA validation users: "
        f"{len(jepa_val):,}"
    )

    print(
        f"JEPA test users: "
        f"{len(jepa_test):,}"
    )

    print(
        f"JEPA dataset saved to: {jepa_path}"
    )


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    raw_dir=RAW_DIR,
    interim_dir=INTERIM_DIR,
    processed_dir=PROCESSED_DIR,
    max_seq_len=MAX_SEQ_LEN,
    min_item_interactions=MIN_ITEM_INTERACTIONS,
):
    """Run the complete MovieLens-1M preprocessing pipeline."""

    print("=" * 72)
    print("MovieLens-1M Preprocessing Pipeline")
    print("=" * 72)

    print(
        f"Raw data directory: {raw_dir}"
    )

    print(
        f"Interim data directory: {interim_dir}"
    )

    print(
        f"Processed data directory: {processed_dir}"
    )

    print(
        f"Maximum sequence length: {max_seq_len}"
    )

    print(
        f"Minimum item interactions: "
        f"{min_item_interactions}"
    )

    # ------------------------------------------------------------------
    # Step 1: Load raw data.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 1: Loading Raw MovieLens-1M Dataset ==="
    )

    ratings, users, movies = load_movielens(
        raw_dir
    )

    print(
        f"Loaded {len(ratings):,} ratings, "
        f"{len(users):,} users, and "
        f"{len(movies):,} movies."
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

    print(
        f"Minimum movie interactions: "
        f"{min_item_interactions}"
    )

    print(
        f"Remaining interactions: "
        f"{len(ratings_filtered):,}"
    )

    print(
        f"Removed interactions: "
        f"{len(ratings) - len(ratings_filtered):,}"
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

    print(
        f"Users before minimum sequence filtering: "
        f"{len(user_sequences):,}"
    )

    print(
        f"Maximum sequence length: "
        f"{max_seq_len}"
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

    print(
        f"Retained users with sequence length >= 3: "
        f"{len(df_split):,}"
    )

    # ------------------------------------------------------------------
    # Step 5: Contiguous movie ID mapping.
    # ------------------------------------------------------------------

    print(
        "\n=== Step 5: Creating Contiguous Movie ID Mapping ==="
    )

    movie_map, movie_mapping_df = create_movie_mapping(
        df_split,
        movies,
    )

    df_split = apply_movie_mapping(
        df_split,
        movie_map,
    )

    print(
        f"Unique movies: "
        f"{len(movie_map):,}"
    )

    print(
        "PAD token: 0"
    )

    print(
        "MASK token: 1"
    )

    print(
        f"Vocabulary size: "
        f"{len(movie_map) + 2:,}"
    )

    # ------------------------------------------------------------------
    # Save shared intermediate artifacts.
    # ------------------------------------------------------------------

    save_interim_dataset(
        df_split,
        movie_mapping_df,
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
        "Preprocessing pipeline completed successfully."
    )
    print("=" * 72)


if __name__ == "__main__":
    preprocess_pipeline()