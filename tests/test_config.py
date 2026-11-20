"""Tests for configuration loading, using the repository's own configs."""

from pathlib import Path

import pytest

from src.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CAUSAL_CONFIG = CONFIG_DIR / "ml-1m" / "jepa_causal_tuned.yaml"


def test_dataset_is_inferred_from_the_config_folder():
    config = load_config(CAUSAL_CONFIG)

    assert config["dataset"] == "ml-1m"


def test_dataset_facts_are_merged_into_the_model_section():
    config = load_config(CAUSAL_CONFIG)

    assert config["model"]["vocab_size"] == 3418
    assert config["model"]["max_seq_len"] == 200


def test_data_paths_expand_the_dataset_placeholder():
    config = load_config(CAUSAL_CONFIG)

    assert config["data"]["train_path"] == (
        "data/processed/ml-1m/jepa/train_context_target.parquet"
    )


def test_output_paths_are_derived_from_dataset_and_experiment():
    config = load_config(CAUSAL_CONFIG)
    output = config["output"]

    assert output["experiment_name"] == "jepa_causal_tuned"
    assert output["results_dir"] == "results/ml-1m/jepa_causal_tuned"
    assert output["checkpoint_dir"] == "checkpoints/ml-1m/jepa_causal_tuned"
    assert output["best_checkpoint_name"] == "jepa_causal_tuned_best.pt"


def test_name_and_overrides_are_applied():
    config = load_config(
        CAUSAL_CONFIG,
        overrides=["seed=7"],
        name="my_run",
    )

    assert config["seed"] == 7
    assert config["output"]["results_dir"] == "results/ml-1m/my_run"


def test_changing_the_dataset_with_set_is_rejected():
    with pytest.raises(ValueError):
        load_config(CAUSAL_CONFIG, overrides=["dataset=amazon_beauty"])


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(CONFIG_DIR / "ml-1m" / "does_not_exist.yaml")
