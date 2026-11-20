"""Tests for command-line config overrides."""

import pytest

from src.utils.overrides import apply_name, apply_overrides

BASE_CONFIG = {
    "seed": 42,
    "dataset": "ml-1m",
    "training": {
        "learning_rate": 0.0005,
        "compile": False,
        "num_epochs": 150,
    },
    "output": {
        "experiment_name": "a",
        "checkpoint_dir": "x",
        "results_dir": "y",
        "best_checkpoint_name": "z.pt",
    },
}


def test_overrides_parse_types_and_do_not_mutate_the_input():
    result = apply_overrides(
        BASE_CONFIG,
        ["training.learning_rate=0.001", "training.compile=true", "seed=7"],
    )

    assert result["training"]["learning_rate"] == 0.001
    assert result["training"]["compile"] is True
    assert result["seed"] == 7

    assert BASE_CONFIG["training"]["learning_rate"] == 0.0005
    assert BASE_CONFIG["seed"] == 42


def test_overrides_reject_unknown_keys_and_malformed_arguments():
    with pytest.raises(ValueError):
        apply_overrides(BASE_CONFIG, ["training.learning_rat=1"])

    with pytest.raises(ValueError):
        apply_overrides(BASE_CONFIG, ["trainng.learning_rate=1"])

    with pytest.raises(ValueError):
        apply_overrides(BASE_CONFIG, ["no_equals_sign"])


def test_plus_prefix_adds_a_new_key():
    result = apply_overrides(BASE_CONFIG, ["+training.new_flag=3"])

    assert result["training"]["new_flag"] == 3


def test_name_rederives_every_output_path():
    output = apply_name(BASE_CONFIG, "run1")["output"]

    assert output == {
        "experiment_name": "run1",
        "checkpoint_dir": "checkpoints/ml-1m/run1",
        "results_dir": "results/ml-1m/run1",
        "best_checkpoint_name": "run1_best.pt",
    }
