"""Staged grid search for BERT4Rec using this repository's training pipeline.

Run from the repository root:
    python scripts/run_bert4rec_hyperparameter_search.py \
        --config configs/bert4rec_tuned.yaml

Each trial gets its own config, checkpoint, training history, and run summary.
The aggregate CSV and JSON report are updated after each trial. Trials are
ranked by best validation NDCG@10; this script never evaluates the test set.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_bert4rec import train_bert4rec

LOGGER = logging.getLogger("bert4rec_hyperparameter_search")

# 24-trial first-pass grid. Keep architecture fixed initially; tune the main
# optimization and masking variables before spending compute on model size.
SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [3e-4, 5e-4, 1e-3],
    "weight_decay": [0.01, 0.05],
    "mask_probability": [0.15, 0.30],
    "batch_size": [128, 256],
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_outputs(output_dir: Path, trials: list[dict[str, Any]],
                  started_at: str, completed: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(output_dir / "search_summary.csv", index=False)

    successful = [
        t for t in trials
        if t.get("status") == "SUCCESS" and finite_number(t.get("best_validation_ndcg10"))
    ]
    successful.sort(key=lambda t: float(t["best_validation_ndcg10"]), reverse=True)
    report = {
        "experiment": "BERT4Rec hyperparameter search",
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed": completed,
        "selection_metric": "validation NDCG@10",
        "total_trials_planned": math.prod(len(v) for v in SEARCH_SPACE.values()),
        "trials_recorded": len(trials),
        "successful_trials": sum(t.get("status") == "SUCCESS" for t in trials),
        "failed_trials": sum(t.get("status") == "FAILED" for t in trials),
        "search_space": SEARCH_SPACE,
        "best_trial_overall": successful[0] if successful else None,
        "trials": trials,
    }
    with (output_dir / "grid_search_report.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")
    LOGGER.info("Updated %s", output_dir / "search_summary.csv")
    LOGGER.info("Updated %s", output_dir / "grid_search_report.json")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for key in ("data", "model", "training", "output"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Base config must contain a '{key}' mapping")
    return config


def trial_name(params: dict[str, Any]) -> str:
    return (
        f"bert4rec_lr{params['learning_rate']:g}"
        f"_wd{params['weight_decay']:g}"
        f"_mask{params['mask_probability']:g}"
        f"_bs{params['batch_size']}"
    ).replace(".", "p")


def best_history_row(history: Any) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        raise ValueError("Trainer did not return a non-empty training_history list")
    valid = [r for r in history if isinstance(r, dict) and finite_number(r.get("NDCG@10"))]
    if not valid:
        raise ValueError("No finite validation NDCG@10 was found in training history")
    return max(valid, key=lambda r: float(r["NDCG@10"]))


def run_search(config_path: Path, max_epochs: int | None = None,
               limit_trials: int | None = None) -> int:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Base configuration not found: {config_path}")
    base_config = load_config(config_path)

    output_root = PROJECT_ROOT / "results" / "bert4rec_grid_search"
    checkpoint_root = PROJECT_ROOT / "checkpoints" / "bert4rec_grid_search"
    config_root = output_root / "trial_configs"
    output_root.mkdir(parents=True, exist_ok=True)

    keys = list(SEARCH_SPACE)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*(SEARCH_SPACE[k] for k in keys))]
    if limit_trials is not None:
        if limit_trials < 1:
            raise ValueError("--limit-trials must be at least 1")
        combinations = combinations[:limit_trials]

    started_at = datetime.now().astimezone().isoformat()
    trials: list[dict[str, Any]] = []
    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Base config: %s", config_path)
    LOGGER.info("Planned trials this run: %d", len(combinations))
    LOGGER.info("Selection metric: validation NDCG@10; test set is not evaluated")
    write_outputs(output_root, trials, started_at, completed=False)

    for number, params in enumerate(combinations, start=1):
        name = trial_name(params)
        trial_results = output_root / name
        trial_checkpoints = checkpoint_root / name
        trial_results.mkdir(parents=True, exist_ok=True)
        trial_checkpoints.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(base_config)
        config["training"].update(params)
        if max_epochs is not None:
            config["training"]["num_epochs"] = max_epochs
        config["output"].update({
            "experiment_name": name,
            "results_dir": str(trial_results),
            "checkpoint_dir": str(trial_checkpoints),
            "best_checkpoint_name": f"{name}_best.pt",
        })
        config["experiment_name"] = name

        config_path_for_trial = config_root / f"{name}.yaml"
        config_path_for_trial.parent.mkdir(parents=True, exist_ok=True)
        with config_path_for_trial.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        record: dict[str, Any] = {
            "trial_number": number,
            "experiment_name": name,
            **params,
            "num_epochs": config["training"]["num_epochs"],
            "seed": config.get("seed", 42),
            "config_path": str(config_path_for_trial),
            "results_dir": str(trial_results),
            "checkpoint_dir": str(trial_checkpoints),
            "status": "RUNNING",
            "best_validation_ndcg10": None,
            "best_epoch": None,
            "best_validation_metrics": {},
            "error": None,
        }
        trials.append(record)
        write_outputs(output_root, trials, started_at, completed=False)
        LOGGER.info("=" * 78)
        LOGGER.info("Trial %d/%d: %s | params=%s", number, len(combinations), name, params)

        try:
            # The repository trainer accepts a YAML path (not a dict).
            trainer_result = train_bert4rec(str(config_path_for_trial))
            row = best_history_row(trainer_result.get("training_history"))
            record["best_validation_ndcg10"] = float(row["NDCG@10"])
            record["best_epoch"] = int(row["epoch"])
            record["best_validation_metrics"] = {
                key: row.get(key)
                for key in ("HR@5", "HR@10", "NDCG@5", "NDCG@10", "loss", "epoch_time_seconds")
                if key in row
            }
            record["best_checkpoint_path"] = trainer_result.get("best_model_path")
            record["training_history_path"] = str(trial_results / "training_history.csv")
            record["run_summary_path"] = str(trial_results / "run_summary.json")
            record["status"] = "SUCCESS"
            LOGGER.info("Trial complete: best val NDCG@10=%.5f at epoch %s",
                        record["best_validation_ndcg10"], record["best_epoch"])
        except Exception as exc:
            LOGGER.exception("Trial failed: %s", name)
            record["status"] = "FAILED"
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            write_outputs(output_root, trials, started_at, completed=False)

    write_outputs(output_root, trials, started_at, completed=True)
    successful = [t for t in trials if t.get("status") == "SUCCESS" and finite_number(t.get("best_validation_ndcg10"))]
    successful.sort(key=lambda t: float(t["best_validation_ndcg10"]), reverse=True)
    LOGGER.info("=" * 78)
    LOGGER.info("Search complete: %d successful / %d attempted", len(successful), len(trials))
    LOGGER.info("Summary CSV: %s", output_root / "search_summary.csv")
    LOGGER.info("Summary JSON: %s", output_root / "grid_search_report.json")
    if successful:
        LOGGER.info("Best trial: %s (validation NDCG@10=%.5f)",
                    successful[0]["experiment_name"], successful[0]["best_validation_ndcg10"])
    return 0 if successful else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BERT4Rec hyperparameter grid search.")
    parser.add_argument("--config", type=Path, default=Path("configs/bert4rec_tuned.yaml"),
                        help="Base BERT4Rec YAML config (default: configs/bert4rec_tuned.yaml).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count for every trial; default uses the base config.")
    parser.add_argument("--limit-trials", type=int, default=None,
                        help="Run only the first N combinations (useful for a smoke test).")
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    return run_search(args.config, max_epochs=args.epochs, limit_trials=args.limit_trials)


if __name__ == "__main__":
    raise SystemExit(main())
