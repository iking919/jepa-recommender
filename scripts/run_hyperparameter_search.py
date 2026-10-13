
"""
Staged grid search for causal one-pass JEPA.

All experiment outputs are stored under:
    results/jepa_grid_search/

Each trial receives its own subdirectory. The master CSV and JSON report
are updated after every completed or failed trial.

Checkpoints are disabled during the search. Training histories, configs,
run summaries, and aggregate search results are retained.
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

from src.training.train_jepa_causal import train


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOGGER = logging.getLogger("jepa_causal_grid_search")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


# ---------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------

SEARCH_SPACE = {
    "lr_schedule": ["constant", "cosine"],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "weight_decay": [0.01, 0.05],
    "warmup_epochs": [0, 5],
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def is_finite_number(value: Any) -> bool:
    """Return True if value can be interpreted as a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def to_json_safe(value: Any) -> Any:
    """Convert common NumPy/Pandas values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Handle NumPy scalar values without requiring NumPy as an import.
    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except (ValueError, TypeError):
            pass

    # Handle paths and other objects conservatively.
    if isinstance(value, Path):
        return str(value)

    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report, rejecting non-finite JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = to_json_safe(payload)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            safe_payload,
            file,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def write_search_outputs(
    output_dir: Path,
    trials: list[dict[str, Any]],
    started_at: str,
    completed: bool,
) -> None:
    """Persist aggregate CSV and JSON after each trial."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "search_summary.csv"
    json_path = output_dir / "grid_search_report.json"

    dataframe = pd.DataFrame(trials)
    dataframe.to_csv(csv_path, index=False)

    successful_trials = [
        trial
        for trial in trials
        if trial.get("status") == "SUCCESS"
        and is_finite_number(trial.get("best_selection_score"))
    ]

    ranked_trials = sorted(
        successful_trials,
        key=lambda trial: float(trial["best_selection_score"]),
        reverse=True,
    )

    best_by_schedule: dict[str, dict[str, Any] | None] = {}

    for schedule in SEARCH_SPACE["lr_schedule"]:
        schedule_trials = [
            trial
            for trial in ranked_trials
            if trial.get("lr_schedule") == schedule
        ]
        best_by_schedule[schedule] = (
            schedule_trials[0] if schedule_trials else None
        )

    report = {
        "experiment": "Causal one-pass JEPA hyperparameter search",
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed": completed,
        "total_trials_planned": math.prod(
            len(values) for values in SEARCH_SPACE.values()
        ),
        "trials_recorded": len(trials),
        "successful_trials": sum(
            trial.get("status") == "SUCCESS" for trial in trials
        ),
        "failed_trials": sum(
            trial.get("status") == "FAILED" for trial in trials
        ),
        "selection_metric": "best_selection_score",
        "search_space": SEARCH_SPACE,
        "best_trial_overall": ranked_trials[0] if ranked_trials else None,
        "best_trial_by_schedule": best_by_schedule,
        "trials": trials,
    }

    write_json(json_path, report)

    LOGGER.info("Updated master CSV: %s", csv_path)
    LOGGER.info("Updated master JSON: %s", json_path)


def build_trial_name(
    schedule: str,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
) -> str:
    """Build a readable and filesystem-safe trial name."""
    lr_label = f"{learning_rate:.4f}"
    wd_label = f"{weight_decay:.2f}"

    return (
        f"jepa_causal_{schedule}"
        f"_lr{lr_label}"
        f"_wd{wd_label}"
        f"_warm{warmup_epochs}"
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration into a dictionary."""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML dictionary in configuration: {config_path}"
        )

    config.setdefault("training", {})
    config.setdefault("output", {})

    return config


def make_trial_config(
    base_config: dict[str, Any],
    experiment_name: str,
    results_dir: Path,
    schedule: str,
    learning_rate: float,
    weight_decay: float,
    warmup_epochs: int,
) -> dict[str, Any]:
    """Copy the base config and apply one trial's parameters."""
    config = copy.deepcopy(base_config)

    config.setdefault("training", {})
    config.setdefault("output", {})

    config["training"]["lr_schedule"] = schedule
    config["training"]["learning_rate"] = learning_rate
    config["training"]["weight_decay"] = weight_decay
    config["training"]["warmup_epochs"] = warmup_epochs

    # No model checkpoints should be written during hyperparameter search.
    config["training"]["save_checkpoint"] = False

    # Each trial writes all its outputs into its own search subdirectory.
    config["output"]["results_dir"] = str(results_dir)

    # Keep checkpoint paths unset when checkpoint saving is disabled.
    config["output"]["checkpoint_dir"] = None
    config["output"]["best_checkpoint_name"] = None

    # Preserve the trial name in the config if the trainer uses it.
    config["experiment_name"] = experiment_name

    return config


def extract_best_metrics(
    summary: Any,
) -> tuple[dict[str, Any], Any]:
    """
    Extract metrics from the trainer's returned summary.

    Expected trainer output includes best_metrics and best_selection_score.
    The alternate lookups make this helper tolerant of common summary shapes.
    """
    if not isinstance(summary, dict):
        raise TypeError(
            "train(config) must return a dictionary summary; "
            f"received {type(summary).__name__}"
        )

    metrics = summary.get("best_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    selection_score = summary.get("best_selection_score")

    if selection_score is None:
        selection_score = summary.get("best_score")

    if selection_score is None:
        selection_score = metrics.get("ndcg@10")

    if selection_score is None:
        selection_score = metrics.get("ndcg10")

    return metrics, selection_score


# ---------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------

def run_search(config_path: Path) -> int:
    config_path = config_path.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    base_config = load_config(config_path)

    # All master files and trial subdirectories live here.
    search_results_dir = PROJECT_ROOT / "results" / "jepa_grid_search"
    search_results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = search_results_dir / "search_summary.csv"
    json_path = search_results_dir / "grid_search_report.json"

    combinations = list(
        itertools.product(
            SEARCH_SPACE["lr_schedule"],
            SEARCH_SPACE["learning_rate"],
            SEARCH_SPACE["weight_decay"],
            SEARCH_SPACE["warmup_epochs"],
        )
    )

    started_at = datetime.now().astimezone().isoformat()
    all_trials: list[dict[str, Any]] = []

    LOGGER.info("Project root: %s", PROJECT_ROOT)
    LOGGER.info("Base configuration: %s", config_path)
    LOGGER.info("Search output directory: %s", search_results_dir)
    LOGGER.info("Total trials: %d", len(combinations))
    LOGGER.info("Checkpoint saving: disabled for every trial")

    # Initialize the aggregate files before the first trial.
    write_search_outputs(
        output_dir=search_results_dir,
        trials=all_trials,
        started_at=started_at,
        completed=False,
    )

    for trial_number, parameters in enumerate(combinations, start=1):
        (
            schedule,
            learning_rate,
            weight_decay,
            warmup_epochs,
        ) = parameters

        experiment_name = build_trial_name(
            schedule=schedule,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
        )

        # Crucially, the trial directory is nested under the master directory.
        trial_results_dir = search_results_dir / experiment_name
        trial_results_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("=" * 80)
        LOGGER.info("Trial %d/%d: %s", trial_number, len(combinations), experiment_name)
        LOGGER.info(
            "Parameters: schedule=%s, lr=%g, weight_decay=%g, warmup=%d",
            schedule,
            learning_rate,
            weight_decay,
            warmup_epochs,
        )
        LOGGER.info("Trial output: %s", trial_results_dir)

        trial_record: dict[str, Any] = {
            "trial_number": trial_number,
            "experiment_name": experiment_name,
            "trial_results_dir": str(trial_results_dir),
            "lr_schedule": schedule,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "save_checkpoint": False,
            "status": "RUNNING",
            "best_selection_score": None,
            "best_metrics": {},
            "error": None,
        }

        # Record the running trial so earlier results remain available.
        all_trials.append(trial_record)
        write_search_outputs(
            output_dir=search_results_dir,
            trials=all_trials,
            started_at=started_at,
            completed=False,
        )

        try:
            trial_config = make_trial_config(
                base_config=base_config,
                experiment_name=experiment_name,
                results_dir=trial_results_dir,
                schedule=schedule,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
            )

            # Save the exact configuration used for this trial.
            config_output_path = trial_results_dir / "config.yaml"
            with config_output_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(
                    trial_config,
                    file,
                    sort_keys=False,
                )

            summary = train(trial_config)
            best_metrics, selection_score = extract_best_metrics(summary)

            trial_record["best_metrics"] = to_json_safe(best_metrics)
            trial_record["best_selection_score"] = (
                float(selection_score)
                if is_finite_number(selection_score)
                else None
            )
            trial_record["status"] = "SUCCESS"

            # Retain additional trainer summary information when available.
            trial_record["run_summary"] = to_json_safe(summary)

            LOGGER.info(
                "Trial completed: %s; selection score=%s",
                experiment_name,
                trial_record["best_selection_score"],
            )

        except Exception as exc:
            LOGGER.exception("Trial failed: %s", experiment_name)

            trial_record["status"] = "FAILED"
            trial_record["error"] = f"{type(exc).__name__}: {exc}"

        finally:
            # Update the master files even if the current trial failed.
            write_search_outputs(
                output_dir=search_results_dir,
                trials=all_trials,
                started_at=started_at,
                completed=False,
            )

    # Final report marks the full search as complete.
    write_search_outputs(
        output_dir=search_results_dir,
        trials=all_trials,
        started_at=started_at,
        completed=True,
    )

    successful_trials = [
        trial
        for trial in all_trials
        if trial.get("status") == "SUCCESS"
        and is_finite_number(trial.get("best_selection_score"))
    ]
    successful_trials.sort(
        key=lambda trial: float(trial["best_selection_score"]),
        reverse=True,
    )

    LOGGER.info("=" * 80)
    LOGGER.info("Grid search finished.")
    LOGGER.info("Successful trials: %d", len(successful_trials))
    LOGGER.info("Failed trials: %d", sum(
        trial.get("status") == "FAILED" for trial in all_trials
    ))
    LOGGER.info("Master CSV: %s", csv_path)
    LOGGER.info("Master JSON: %s", json_path)

    if successful_trials:
        best = successful_trials[0]
        LOGGER.info("Best trial: %s", best["experiment_name"])
        LOGGER.info(
            "Best selection score: %s",
            best["best_selection_score"],
        )
    else:
        LOGGER.warning(
            "No successful trial produced a finite selection score."
        )

    return 0 if successful_trials else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter grid search for causal JEPA."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/jepa_tuned.yaml"),
        help="Base YAML configuration file.",
    )

    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()
    return run_search(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
