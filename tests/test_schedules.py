"""Run: python -m tests.test_schedules"""

import math

from src.utils.schedules import lr_factor, trailing_mean
from src.utils.overrides import apply_name, apply_overrides


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_constant_is_one():
    assert all(lr_factor(s, 1000) == 1.0 for s in (0, 10, 999))


def test_warmup_ramps_to_one():
    f = [lr_factor(s, 1000, warmup_steps=10) for s in range(12)]
    assert f[0] == 0.1 and f[9] == 1.0 and f[10] == 1.0
    assert all(a <= b for a, b in zip(f, f[1:]))


def test_cosine_shape():
    total = 1000
    assert lr_factor(0, total, kind="cosine") == 1.0
    assert abs(lr_factor(500, total, kind="cosine") - 0.5) < 1e-9
    assert abs(lr_factor(total, total, kind="cosine")) < 1e-9
    assert abs(lr_factor(total, total, min_ratio=0.1, kind="cosine") - 0.1) < 1e-9
    # never goes negative or past the end
    assert lr_factor(10 * total, total, kind="cosine") >= 0.0


def test_cosine_after_warmup_starts_at_one():
    assert lr_factor(100, 1000, warmup_steps=100, kind="cosine") == 1.0
    assert lr_factor(99, 1000, warmup_steps=100, kind="cosine") == 1.0  # end of ramp


def test_bad_schedule_rejected():
    assert raises(lambda: lr_factor(0, 10, kind="linear"))
    assert raises(lambda: lr_factor(0, 10, min_ratio=2.0, kind="cosine"))


def test_trailing_mean():
    assert trailing_mean([1.0], 3) == 1.0
    assert trailing_mean([1.0, 2.0, 3.0, 4.0], 2) == 3.5
    assert trailing_mean([1.0, 2.0], 1) == 2.0
    assert raises(lambda: trailing_mean([], 2))


CFG = {
    "seed": 42,
    "training": {"learning_rate": 0.0005, "compile": False, "num_epochs": 150},
    "output": {"experiment_name": "a", "checkpoint_dir": "x", "results_dir": "y",
               "best_checkpoint_name": "z.pt"},
}


def test_overrides_parse_types_and_do_not_mutate():
    out = apply_overrides(CFG, ["training.learning_rate=0.001", "training.compile=true", "seed=7"])
    assert out["training"]["learning_rate"] == 0.001 and out["training"]["compile"] is True
    assert out["seed"] == 7
    assert CFG["training"]["learning_rate"] == 0.0005 and CFG["seed"] == 42


def test_overrides_reject_typos_but_allow_plus():
    assert raises(lambda: apply_overrides(CFG, ["training.learning_rat=1"]))
    assert raises(lambda: apply_overrides(CFG, ["trainng.learning_rate=1"]))
    assert raises(lambda: apply_overrides(CFG, ["no_equals_sign"]))
    out = apply_overrides(CFG, ["+training.new_flag=3"])
    assert out["training"]["new_flag"] == 3


def test_name_sets_all_output_paths():
    out = apply_name(CFG, "run1")["output"]
    assert out == {"experiment_name": "run1", "checkpoint_dir": "checkpoints/run1",
                   "results_dir": "results/run1", "best_checkpoint_name": "run1_best.pt"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
