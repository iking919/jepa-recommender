"""Dataset-agnostic preprocessing steps.

The dataset-specific scripts (``scripts/preprocess_<dataset>.py``) load and
sort the raw interactions, then use the steps below to produce the shared
intermediate files and the model-specific training files.

Pipeline, after loading and chronological sorting:

    1. ``preprocess_ratings``      filter, build sequences, split
    2. ``build_item_id_map``       original item IDs -> contiguous model IDs
    3. ``apply_item_id_map``       rewrite the split with model IDs
    4. ``write_datasets``          interim, BERT4Rec and JEPA parquet files
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.splits import leave_last_two_out
from src.data.tokens import FIRST_ITEM_ID, MASK_TOKEN_ID, PAD_TOKEN_ID

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 200
MIN_ITEM_INTERACTIONS = 5


# ---------------------------------------------------------------------
# Filtering, sequences and splits
# ---------------------------------------------------------------------

def filter_rare_items(
    ratings: pd.DataFrame,
    item_column: str,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> pd.DataFrame:
    """Drop items with fewer than ``min_item_interactions`` interactions."""

    item_counts = ratings[item_column].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index

    filtered = ratings[ratings[item_column].isin(valid_items)].copy()

    logger.info("Minimum item interactions: %d", min_item_interactions)
    logger.info(
        "Items: %s -> %s",
        f"{ratings[item_column].nunique():,}",
        f"{filtered[item_column].nunique():,}",
    )
    logger.info(
        "Interactions: %s -> %s (removed %s)",
        f"{len(ratings):,}",
        f"{len(filtered):,}",
        f"{len(ratings) - len(filtered):,}",
    )

    return filtered


def build_user_sequences(
    ratings: pd.DataFrame,
    item_column: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> pd.DataFrame:
    """Build each user's chronological item sequence.

    ``ratings`` must already be sorted chronologically within each user.
    Sequences keep only the most recent ``max_seq_len`` interactions.
    """

    user_sequences = (
        ratings.groupby("user_id", sort=False)[item_column]
        .agg(list)
        .reset_index(name="sequence")
    )

    user_sequences["sequence"] = user_sequences["sequence"].apply(
        lambda sequence: sequence[-max_seq_len:]
    )

    lengths = user_sequences["sequence"].apply(len)

    logger.info("Users before sequence-length filtering: %s", f"{len(lengths):,}")
    logger.info("Maximum sequence length: %d", max_seq_len)
    logger.info(
        "Sequence length: mean %.2f | median %.2f | min %d | max %d",
        lengths.mean(),
        lengths.median(),
        lengths.min(),
        lengths.max(),
    )

    return user_sequences


def preprocess_ratings(
    ratings: pd.DataFrame,
    item_column: str,
    max_seq_len: int = MAX_SEQ_LEN,
    min_item_interactions: int = MIN_ITEM_INTERACTIONS,
) -> pd.DataFrame:
    """Filter, build sequences and split chronologically.

    Args:
        ratings: Interactions with ``user_id`` and ``item_column`` columns,
            sorted chronologically within each user.
        item_column: Name of the column holding the original item IDs.
        max_seq_len: Maximum sequence length kept per user.
        min_item_interactions: Items below this count are removed.

    Returns:
        The leave-last-two-out split, still using the original item IDs.
    """

    logger.info("Filtering rare items")
    filtered = filter_rare_items(ratings, item_column, min_item_interactions)

    logger.info("Building user sequences")
    user_sequences = build_user_sequences(filtered, item_column, max_seq_len)

    logger.info("Creating leave-last-two-out split")
    df_split = leave_last_two_out(user_sequences)

    if df_split.empty:
        raise RuntimeError("No users remain after leave-last-two-out splitting.")

    logger.info("Users with at least 3 interactions: %s", f"{len(df_split):,}")
    logger.info(
        "Excluded users: %s", f"{len(user_sequences) - len(df_split):,}"
    )

    return df_split


# ---------------------------------------------------------------------
# Item ID mapping
# ---------------------------------------------------------------------

def build_item_id_map(df_split: pd.DataFrame) -> dict:
    """Map original item IDs to contiguous model IDs.

    The map covers every item in the train, validation and test columns, so
    all targets have a valid ID. IDs start at ``FIRST_ITEM_ID`` (0 and 1 are
    reserved for padding and masking) and follow the sorted original IDs.
    """

    unique_items = sorted(
        set(item for sequence in df_split["train_sequence"] for item in sequence)
        | set(df_split["validation_item"])
        | set(df_split["test_item"])
    )

    item_map = {
        item: index + FIRST_ITEM_ID for index, item in enumerate(unique_items)
    }

    logger.info("Unique items: %s", f"{len(item_map):,}")
    logger.info(
        "PAD id: %d | MASK id: %d | first item id: %d",
        PAD_TOKEN_ID,
        MASK_TOKEN_ID,
        FIRST_ITEM_ID,
    )
    logger.info("Vocabulary size: %s", f"{len(item_map) + FIRST_ITEM_ID:,}")

    return item_map


def apply_item_id_map(df_split: pd.DataFrame, item_map: dict) -> pd.DataFrame:
    """Return a copy of the split with original item IDs replaced."""

    df_split = df_split.copy()

    df_split["train_sequence"] = df_split["train_sequence"].apply(
        lambda sequence: [item_map[item] for item in sequence]
    )
    df_split["validation_item"] = df_split["validation_item"].map(item_map)
    df_split["test_item"] = df_split["test_item"].map(item_map)

    return df_split


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------

def _save_parquet(frame: pd.DataFrame, directory: Path, filename: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / filename, index=False)


def _append_validation_item(row: pd.Series, max_seq_len: int) -> list[int]:
    """Test context: the training sequence plus the validation item."""

    return (list(row["train_sequence"]) + [row["validation_item"]])[
        -max_seq_len:
    ]


def save_interim_dataset(
    df_split: pd.DataFrame,
    item_mapping: pd.DataFrame,
    interim_dir: Path,
    mapping_filename: str,
) -> None:
    """Save the mapped split and the item-ID mapping table."""

    _save_parquet(df_split, interim_dir, "interim_sequences.parquet")
    item_mapping.to_csv(interim_dir / mapping_filename, index=False)

    logger.info("Interim dataset saved to: %s", interim_dir)


def generate_bert4rec_datasets(
    df_split: pd.DataFrame,
    output_dir: Path,
    max_seq_len: int = MAX_SEQ_LEN,
) -> None:
    """Write the BERT4Rec train, validation and test files.

    The test input is the training sequence followed by the validation item.
    """

    bert_train = df_split[["user_id", "train_sequence"]].copy()
    bert_val = df_split[["user_id", "train_sequence", "validation_item"]].copy()

    bert_test = df_split[
        ["user_id", "train_sequence", "validation_item", "test_item"]
    ].copy()
    bert_test["test_input"] = bert_test.apply(
        _append_validation_item, axis=1, max_seq_len=max_seq_len
    )
    bert_test = bert_test[["user_id", "test_input", "test_item"]]

    _save_parquet(bert_train, output_dir, "train_sequences.parquet")
    _save_parquet(bert_val, output_dir, "validation.parquet")
    _save_parquet(bert_test, output_dir, "test.parquet")

    logger.info("BERT4Rec training users: %s", f"{len(bert_train):,}")
    logger.info("BERT4Rec validation users: %s", f"{len(bert_val):,}")
    logger.info("BERT4Rec test users: %s", f"{len(bert_test):,}")
    logger.info("BERT4Rec dataset saved to: %s", output_dir)


def generate_jepa_datasets(
    df_split: pd.DataFrame,
    output_dir: Path,
    max_seq_len: int = MAX_SEQ_LEN,
) -> None:
    """Write the JEPA context-target train, validation and test files.

    Training pairs: ``context_sequence`` holds the interactions before
    position ``i`` and ``target_item`` the interaction at position ``i``.
    Validation predicts the validation item from the full training
    sequence; test predicts the test item after also seeing the validation
    item.
    """

    pair_columns = ["user_id", "context_sequence", "target_item"]

    train_rows = []
    for _, row in df_split.iterrows():
        sequence = row["train_sequence"]

        if len(sequence) < 2:
            continue

        for index in range(1, len(sequence)):
            train_rows.append(
                {
                    "user_id": row["user_id"],
                    "context_sequence": sequence[:index][-max_seq_len:],
                    "target_item": sequence[index],
                }
            )

    jepa_train = pd.DataFrame(train_rows, columns=pair_columns)

    jepa_val = df_split[["user_id", "train_sequence", "validation_item"]].copy()
    jepa_val.columns = pair_columns
    jepa_val["context_sequence"] = jepa_val["context_sequence"].apply(
        lambda sequence: sequence[-max_seq_len:]
    )

    jepa_test = df_split[
        ["user_id", "train_sequence", "validation_item", "test_item"]
    ].copy()
    jepa_test["context_sequence"] = jepa_test.apply(
        _append_validation_item, axis=1, max_seq_len=max_seq_len
    )
    jepa_test = jepa_test[["user_id", "context_sequence", "test_item"]]
    jepa_test.columns = pair_columns

    _save_parquet(jepa_train, output_dir, "train_context_target.parquet")
    _save_parquet(jepa_val, output_dir, "validation.parquet")
    _save_parquet(jepa_test, output_dir, "test.parquet")

    logger.info("JEPA training pairs: %s", f"{len(jepa_train):,}")
    logger.info("JEPA validation users: %s", f"{len(jepa_val):,}")
    logger.info("JEPA test users: %s", f"{len(jepa_test):,}")
    logger.info("JEPA dataset saved to: %s", output_dir)


def write_datasets(
    df_split: pd.DataFrame,
    item_mapping: pd.DataFrame,
    *,
    dataset: str,
    mapping_filename: str,
    interim_root: Path,
    processed_root: Path,
    max_seq_len: int = MAX_SEQ_LEN,
) -> None:
    """Write the interim files and both model-specific dataset folders.

    Files go to ``<interim_root>/<dataset>/`` and
    ``<processed_root>/<dataset>/{bert4rec,jepa}/``.
    """

    save_interim_dataset(
        df_split,
        item_mapping,
        Path(interim_root) / dataset,
        mapping_filename,
    )

    processed_dir = Path(processed_root) / dataset
    generate_bert4rec_datasets(df_split, processed_dir / "bert4rec", max_seq_len)
    generate_jepa_datasets(df_split, processed_dir / "jepa", max_seq_len)
