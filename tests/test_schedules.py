"""Tests for the learning-rate schedule and model-selection helpers."""

import pytest

from src.utils.schedules import lr_factor, trailing_mean


def test_constant_schedule_is_one():
    assert all(lr_factor(step, 1000) == 1.0 for step in (0, 10, 999))


def test_warmup_ramps_linearly_to_one():
    factors = [lr_factor(step, 1000, warmup_steps=10) for step in range(12)]

    assert factors[0] == 0.1
    assert factors[9] == 1.0
    assert factors[10] == 1.0
    assert all(a <= b for a, b in zip(factors, factors[1:]))


def test_cosine_schedule_shape():
    total = 1000

    assert lr_factor(0, total, kind="cosine") == 1.0
    assert lr_factor(500, total, kind="cosine") == pytest.approx(0.5)
    assert lr_factor(total, total, kind="cosine") == pytest.approx(0.0)
    assert lr_factor(
        total, total, min_ratio=0.1, kind="cosine"
    ) == pytest.approx(0.1)


def test_cosine_schedule_never_goes_negative_past_the_end():
    assert lr_factor(10_000, 1000, kind="cosine") >= 0.0


def test_cosine_starts_at_one_after_warmup():
    assert lr_factor(100, 1000, warmup_steps=100, kind="cosine") == 1.0
    # Last warmup step.
    assert lr_factor(99, 1000, warmup_steps=100, kind="cosine") == 1.0


def test_invalid_schedule_arguments_are_rejected():
    with pytest.raises(ValueError):
        lr_factor(0, 10, kind="linear")

    with pytest.raises(ValueError):
        lr_factor(0, 10, min_ratio=2.0, kind="cosine")


def test_trailing_mean_uses_the_last_window_values():
    assert trailing_mean([1.0], 3) == 1.0
    assert trailing_mean([1.0, 2.0, 3.0, 4.0], 2) == 3.5
    assert trailing_mean([1.0, 2.0], 1) == 2.0


def test_trailing_mean_rejects_empty_input():
    with pytest.raises(ValueError):
        trailing_mean([], 2)
