"""
Staged hyperparameter search for causal one-pass JEPA (v2).

Changes from the first search script:
  * ema_decay and model dropout are real search dimensions (both were
    missing/unwired before) and every value is applied to the trial config.
  * Multiple seeds per configuration; results are aggregated per
    configuration (mean/std) so the winner is not a single lucky run.
  * Defaults reflect the first grid: the best learning rate sat at the top
    edge of the grid, weight decay and warmup barely mattered, and the EMA
    decay was inherited from a regime with ~100x more steps per epoch.
  * Cosine trials automatically get patience >= num_epochs so the schedule
    actually runs to completion instead of being cut off early.
  * Resumable: a trial whose run_summary.json exists is loaded, not retrained.
  * Never overwrites the old search's report (separate output dir by default,
    and a guard against an incompatible report in the chosen dir).
  * --dry-run lists the trials without training; --top-k-from re-runs the
    best configurations of an earlier search across several seeds.

Stage 1 (single seed, ~18 trials with the defaults):
    python -m scripts.run_jepa_hyperparameter_search_v2 \
        --config configs/jepa_causal_baseline.yaml

Stage 2 (best 3 configurations from stage 1, 3 seeds each):
    python -m scripts.run_jepa_hyperparameter_search_v2 \
        --config configs/jepa_causal_baseline.yaml \
        --top-k-from results/jepa_grid_search_v2/grid_search_report.json \
        --top-k 3 --seeds 42 7 123 \
        --output-dir results/jepa_grid_search_v2_seeds

Checkpoints are disabled; histories, configs, summaries and the aggregate
CSV/JSON files are kept.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import logging
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("jepa_causal_grid_search_v2")

# Parameters that define a configuration (seed is separate).
PARAM_KEYS = (
    "lr_schedule",
    "learning_rate",
    "weight_decay",
    "warmup_epochs",
    "ema_decay",
    "dropout",
)

# Stage-1 defaults, informed by the first grid:
#   * lr: best trials were at the 1e-3 edge -> try 1e-3 and 2e-3
#   * weight decay / warmup: no measurable effect -> pinned
#   * schedule: constant (cosine needs a full-length run to be tested fairly;
#     pass --schedule constant cosine to include it)
#   * ema_decay: 0.996 was set for ~10k steps/epoch, causal has ~95
#     -> search faster-moving targets as well
#   * dropout: best runs peak around epoch 30-35 then decline -> more
#     regularization is the likeliest unexplored lever
DEFAULT_SPACE = {
    "lr_schedule": ["constant"],
    "learning_rate": [1e-3, 2e-3],
    "weight_decay": [0.05],
    "warmup_epochs": [5],
    "ema_decay": [0.9, 0.97, 0.996],
    "dropout": [0.2, 0.3, 0.4],
}


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------
# JSON / number helpers
# ---------------------------------------------------------------------

def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(to_json_safe(payload), file, indent=2, allow_nan=False)
        file.write("\n")


def param_key(params: dict[str, Any]) -> tuple:
    return tuple(params[key] for key in PARAM_KEYS)


# ---------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------

def summarize_configurations(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group successful trials by configuration and aggregate over seeds."""

    groups: dict[tuple, list[dict[str, Any]]] = {}

    for trial in trials:
        if trial.get("status") != "SUCCESS":
            continue
        if not is_finite_number(trial.get("best_selection_score")):
            continue
        groups.setdefault(param_key(trial), []).append(trial)

    summaries = []

    for key, members in groups.items():
        scores = [float(m["best_selection_score"]) for m in members]
        summaries.append(
            {
                **dict(zip(PARAM_KEYS, key)),
                "n_seeds": len(scores),
                "seeds": [m["seed"] for m in members],
                "mean_score": statistics.fmean(scores),
                "std_score": (
                    statistics.stdev(scores) if len(scores) > 1 else None
                ),
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_best_epoch": statistics.fmean(
                    m["best_epoch"] for m in members
                    if m.get("best_epoch") is not None
                ) if any(m.get("best_epoch") is not None for m in members)
                else None,
            }
        )

    summaries.sort(key=lambda s: s["mean_score"], reverse=True)
    return summaries


def write_search_outputs(
    output_dir: Path,
    space: dict[str, list[Any]],
    seeds: list[int],
    planned: int,
    trials: list[dict[str, Any]],
    started_at: str,
    completed: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten best_metrics into val_* columns for the CSV.
    rows = []
    for trial in trials:
        row = {k: v for k, v in trial.items() if k != "best_metrics"}
        for name, value in (trial.get("best_metrics") or {}).items():
            row[f"val_{name}"] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "search_summary.csv", index=False)

    configurations = summarize_configurations(trials)

    pd.DataFrame(
        [
            {k: v for k, v in c.items() if k != "seeds"}
            for c in configurations
        ]
    ).to_csv(output_dir / "config_summary.csv", index=False)

    successful = [t for t in trials if t.get("status") == "SUCCESS"]
    failed = [t for t in trials if t.get("status") == "FAILED"]

    report = {
        "experiment": "Causal one-pass JEPA hyperparameter search (v2)",
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed": completed,
        "total_trials_planned": planned,
        "trials_recorded": len(trials),
        "successful_trials": len(successful),
        "failed_trials": len(failed),
        "selection_metric": "best_selection_score (validation NDCG@10)",
        "search_space": space,
        "seeds": seeds,
        "best_configuration_by_mean": (
            configurations[0] if configurations else None
        ),
        "configurations_ranked": configurations,
        "trials": trials,
    }

    write_json(output_dir / "grid_search_report.json", report)


# ---------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML dictionary in: {config_path}")

    for section in ("model", "training", "output"):
        config.setdefault(section, {})

    return config


def build_trial_name(params: dict[str, Any], seed: int) -> str:
    return (
        f"jepa_causal_{params['lr_schedule']}"
        f"_lr{params['learning_rate']:g}"
        f"_wd{params['weight_decay']:g}"
        f"_warm{params['warmup_epochs']}"
        f"_ema{params['ema_decay']:g}"
        f"_do{params['dropout']:g}"
        f"_s{seed}"
    )


def make_trial_config(
    base_config: dict[str, Any],
    experiment_name: str,
    results_dir: Path,
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)

    config["seed"] = seed

    config["model"]["dropout"] = params["dropout"]

    training = config["training"]
    training["lr_schedule"] = params["lr_schedule"]
    training["learning_rate"] = params["learning_rate"]
    training["weight_decay"] = params["weight_decay"]
    training["warmup_epochs"] = params["warmup_epochs"]
    training["ema_decay"] = params["ema_decay"]
    training["save_checkpoint"] = False

    # A cosine schedule only decays if training is allowed to run its full
    # length; otherwise early stopping ends it near the peak learning rate.
    if params["lr_schedule"] == "cosine":
        training["patience"] = max(
            training.get("patience", 0), training["num_epochs"]
        )

    output = config["output"]
    output["experiment_name"] = experiment_name
    output["results_dir"] = str(results_dir)
    output["checkpoint_dir"] = None
    output["best_checkpoint_name"] = None

    return config


# ---------------------------------------------------------------------
# Trial planning
# ---------------------------------------------------------------------

def plan_from_space(space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [
        dict(zip(PARAM_KEYS, values))
        for values in itertools.product(*(space[k] for k in PARAM_KEYS))
    ]


def plan_from_report(report_path: Path, top_k: int) -> list[dict[str, Any]]:
    """Take the top-k configurations (by mean score) from an earlier report."""

    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    trials = report.get("trials", [])

    missing = [
        key for key in PARAM_KEYS
        if trials and any(key not in t for t in trials)
    ]
    if missing:
        raise SystemExit(
            f"{report_path} lacks {missing} on its trials; it was not "
            "produced by this script. Use a v2 report."
        )

    ranked = summarize_configurations(trials)

    if not ranked:
        raise SystemExit(f"No successful trials in {report_path}.")

    return [
        {key: config[key] for key in PARAM_KEYS} for config in ranked[:top_k]
    ]


def guard_output_dir(output_dir: Path) -> None:
    """Refuse to write over a report from the old (v1) search script."""

    report_path = output_dir / "grid_search_report.json"

    if not report_path.is_file():
        return

    try:
        with report_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
    except (OSError, json.JSONDecodeError):
        return

    space = existing.get("search_space", {})

    if "ema_decay" not in space or "dropout" not in space:
        raise SystemExit(
            f"{report_path} comes from the previous search script and would "
            "be overwritten. Choose a different --output-dir (default: "
            "results/jepa_grid_search_v2) or move the old folder first."
        )


def fill_record_from_summary(
    record: dict[str, Any], summary: dict[str, Any]
) -> None:
    metrics = summary.get("best_metrics", {})
    timing = summary.get("timing", {})

    record["best_metrics"] = to_json_safe(metrics) if isinstance(
        metrics, dict
    ) else {}
    score = summary.get("best_selection_score")
    record["best_selection_score"] = (
        float(score) if is_finite_number(score) else None
    )
    record["best_epoch"] = summary.get("best_epoch")
    record["completed_epochs"] = summary.get("completed_epochs")
    record["time_to_best_epoch_seconds"] = timing.get(
        "time_to_best_epoch_seconds"
    )
    record["total_training_time_seconds"] = timing.get(
        "total_training_time_seconds"
    )


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------

def run_search(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    base_config = load_config(config_path)

    if not base_config["model"].get("causal", False):
        raise SystemExit(
            f"{config_path} has model.causal != true. Use a causal JEPA "
            "config such as configs/jepa_causal_baseline.yaml."
        )

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    space = {
        "lr_schedule": args.schedule,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup,
        "ema_decay": args.ema,
        "dropout": args.dropout,
    }

    if args.top_k_from is not None:
        param_sets = plan_from_report(args.top_k_from, args.top_k)
        # Record the configurations actually run as the search space.
        space = {
            key: sorted({p[key] for p in param_sets}, key=str)
            for key in PARAM_KEYS
        }
    else:
        param_sets = plan_from_space(space)

    seeds = list(dict.fromkeys(args.seeds))  # de-duplicate, keep order

    plan = [
        (params, seed) for params in param_sets for seed in seeds
    ]

    LOGGER.info("Base configuration: %s", config_path)
    LOGGER.info("Output directory: %s", output_dir)
    LOGGER.info(
        "Configurations: %d | seeds: %s | total trials: %d",
        len(param_sets), seeds, len(plan),
    )

    if args.dry_run:
        for number, (params, seed) in enumerate(plan, start=1):
            print(f"{number:>3}  {build_trial_name(params, seed)}")
        print(f"\n{len(plan)} trials (dry run, nothing was trained).")
        return 0

    guard_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imported here so --dry-run works without the training stack.
    from src.training.train_jepa_causal import train

    started_at = datetime.now().astimezone().isoformat()
    all_trials: list[dict[str, Any]] = []

    def save(completed: bool) -> None:
        write_search_outputs(
            output_dir=output_dir,
            space=space,
            seeds=seeds,
            planned=len(plan),
            trials=all_trials,
            started_at=started_at,
            completed=completed,
        )

    save(completed=False)

    for number, (params, seed) in enumerate(plan, start=1):
        name = build_trial_name(params, seed)
        trial_dir = output_dir / name
        summary_path = trial_dir / "run_summary.json"

        LOGGER.info("=" * 80)
        LOGGER.info("Trial %d/%d: %s", number, len(plan), name)

        record: dict[str, Any] = {
            "trial_number": number,
            "experiment_name": name,
            "trial_results_dir": str(trial_dir),
            **params,
            "seed": seed,
            "save_checkpoint": False,
            "status": "RUNNING",
            "resumed": False,
            "best_selection_score": None,
            "best_epoch": None,
            "completed_epochs": None,
            "time_to_best_epoch_seconds": None,
            "total_training_time_seconds": None,
            "best_metrics": {},
            "error": None,
        }
        all_trials.append(record)

        # The trainer writes run_summary.json only when a run finishes, so
        # its presence means this trial is already complete.
        if not args.no_resume and summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary = json.load(file)
                fill_record_from_summary(record, summary)
                record["status"] = "SUCCESS"
                record["resumed"] = True
                LOGGER.info(
                    "Already complete; loaded from disk (score=%s).",
                    record["best_selection_score"],
                )
                save(completed=False)
                continue
            except (OSError, json.JSONDecodeError) as error:
                LOGGER.warning(
                    "Could not reuse %s (%s); retraining.", summary_path, error
                )

        save(completed=False)

        try:
            trial_dir.mkdir(parents=True, exist_ok=True)

            trial_config = make_trial_config(
                base_config=base_config,
                experiment_name=name,
                results_dir=trial_dir,
                params=params,
                seed=seed,
            )

            with (trial_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(trial_config, f, sort_keys=False)

            summary = train(trial_config)

            if not isinstance(summary, dict):
                raise TypeError(
                    "train(config) must return a dict summary; got "
                    f"{type(summary).__name__}."
                )

            fill_record_from_summary(record, summary)

            if record["best_selection_score"] is None:
                raise ValueError("Trial produced no finite selection score.")

            record["status"] = "SUCCESS"
            LOGGER.info(
                "Trial completed: score=%.4f best_epoch=%s epochs=%s",
                record["best_selection_score"],
                record["best_epoch"],
                record["completed_epochs"],
            )

        except Exception as error:  # noqa: BLE001 - one bad trial must not stop the search
            LOGGER.exception("Trial failed: %s", name)
            record["status"] = "FAILED"
            record["error"] = f"{type(error).__name__}: {error}"

        finally:
            release_gpu_memory()
            save(completed=False)

    save(completed=True)

    configurations = summarize_configurations(all_trials)

    LOGGER.info("=" * 80)
    LOGGER.info("Search finished.")
    LOGGER.info(
        "Successful trials: %d | failed: %d",
        sum(t["status"] == "SUCCESS" for t in all_trials),
        sum(t["status"] == "FAILED" for t in all_trials),
    )

    for rank, config in enumerate(configurations[:5], start=1):
        std = config["std_score"]
        LOGGER.info(
            "#%d mean=%.4f%s | lr=%g ema=%g dropout=%g wd=%g warmup=%d %s "
            "(n=%d, mean best epoch=%s)",
            rank,
            config["mean_score"],
            f" +/- {std:.4f}" if std is not None else "",
            config["learning_rate"],
            config["ema_decay"],
            config["dropout"],
            config["weight_decay"],
            config["warmup_epochs"],
            config["lr_schedule"],
            config["n_seeds"],
            (
                f"{config['mean_best_epoch']:.0f}"
                if config["mean_best_epoch"] is not None else "n/a"
            ),
        )

    if configurations and args.top_k_from is None:
        # A winner at the edge of the explored range means the range should
        # be widened before trusting it.
        best = configurations[0]
        for key in ("learning_rate", "ema_decay", "dropout"):
            values = sorted(space[key])
            if len(values) > 1 and best[key] in (values[0], values[-1]):
                LOGGER.warning(
                    "Best %s (%g) is at the edge of the searched range %s; "
                    "consider extending it.",
                    key, best[key], values,
                )

    LOGGER.info("Report: %s", output_dir / "grid_search_report.json")

    return 0 if configurations else 1


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for causal one-pass JEPA (v2)."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jepa_causal_baseline.yaml"),
        help="Causal JEPA base config (model.causal must be true).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/jepa_grid_search_v2"),
        help="Where trial folders and aggregate reports are written.",
    )

    space = parser.add_argument_group(
        "search space (each takes one or more values)"
    )
    space.add_argument(
        "--schedule", nargs="+", default=DEFAULT_SPACE["lr_schedule"],
        choices=("constant", "cosine"),
    )
    space.add_argument(
        "--lr", nargs="+", type=float, default=DEFAULT_SPACE["learning_rate"],
    )
    space.add_argument(
        "--weight-decay", nargs="+", type=float,
        default=DEFAULT_SPACE["weight_decay"],
    )
    space.add_argument(
        "--warmup", nargs="+", type=int, default=DEFAULT_SPACE["warmup_epochs"],
    )
    space.add_argument(
        "--ema", nargs="+", type=float, default=DEFAULT_SPACE["ema_decay"],
    )
    space.add_argument(
        "--dropout", nargs="+", type=float, default=DEFAULT_SPACE["dropout"],
    )

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42],
        help="Run every configuration once per seed (default: 42).",
    )
    parser.add_argument(
        "--top-k-from", type=Path, default=None, metavar="REPORT.json",
        help=(
            "Instead of a grid, re-run the best configurations from an "
            "earlier v2 report (ranked by mean score)."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="How many configurations to take with --top-k-from.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Retrain trials even if their run_summary.json already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the planned trials and exit without training.",
    )

    return parser.parse_args()


def main() -> int:
    configure_logging()
    return run_search(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
