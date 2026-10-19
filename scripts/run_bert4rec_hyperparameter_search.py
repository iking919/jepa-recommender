
"""
Compact BERT4Rec hyperparameter grid search.

The search writes only aggregate reports to:
    results/bert4rec_grid_search/
        search_summary.csv
        config_summary.csv
        grid_search_report.json

Individual trial checkpoints, YAML files, histories, and run summaries
are disabled. The trainer returns per-epoch history in memory.

Resume behavior:
    Completed trial IDs in grid_search_report.json are skipped by default.
    Use --restart to start a fresh search and overwrite the aggregate reports.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.training.train_bert4rec import train_bert4rec
from src.utils.config import load_config
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


SEARCH_SPACE: dict[str, list[Any]] = {
    "learning_rate": [3e-4, 5e-4, 1e-3],
    "weight_decay": [0.01, 0.05],
    "mask_probability": [0.15, 0.30],
    "batch_size": [128, 256],
}


def build_trials() -> list[dict[str, Any]]:
    """Build all combinations in a deterministic order."""

    names = list(SEARCH_SPACE)
    values = [SEARCH_SPACE[name] for name in names]

    return [
        dict(zip(names, combination))
        for combination in itertools.product(*values)
    ]


def write_csv(records: list[dict], path: Path) -> None:
    """Write aggregate records to a CSV file."""

    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep a consistent column order across every row.
    fieldnames = list(records[0].keys())

    for record in records[1:]:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def write_json(data: dict, path: Path) -> None:
    """Write JSON through a temporary file to reduce corruption risk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    temporary_path.replace(path)


def save_reports(
    completed_trials: list[dict],
    output_dir: Path,
    base_config_path: str,
) -> None:
    """Save the three aggregate reports, with no per-trial directories."""

    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_trials = sorted(
        completed_trials,
        key=lambda row: row.get("best_ndcg_10", float("-inf")),
        reverse=True,
    )

    summary_fields = [
        "trial_id",
        "learning_rate",
        "weight_decay",
        "mask_probability",
        "batch_size",
        "best_epoch",
        "best_ndcg_10",
        "HR@5",
        "HR@10",
        "NDCG@5",
        "NDCG@10",
        "completed_epochs",
        "total_training_time_seconds",
        "average_epoch_time_seconds",
        "status",
    ]

    summary_rows = [
        {key: trial.get(key) for key in summary_fields}
        for trial in ranked_trials
    ]

    write_csv(
        summary_rows,
        output_dir / "search_summary.csv",
    )

    config_fields = [
        "trial_id",
        "learning_rate",
        "weight_decay",
        "mask_probability",
        "batch_size",
        "best_epoch",
        "best_ndcg_10",
        "status",
    ]

    config_rows = [
        {key: trial.get(key) for key in config_fields}
        for trial in ranked_trials
    ]

    write_csv(
        config_rows,
        output_dir / "config_summary.csv",
    )

    report = {
        "model": "BERT4Rec",
        "dataset": "MovieLens-1M",
        "base_config": base_config_path,
        "search_space": SEARCH_SPACE,
        "total_possible_trials": len(build_trials()),
        "completed_trials": len(completed_trials),
        "successful_trials": sum(
            trial.get("status") == "completed"
            for trial in completed_trials
        ),
        "failed_trials": sum(
            trial.get("status") == "failed"
            for trial in completed_trials
        ),
        "ranking_metric": "validation NDCG@10",
        "updated_at": datetime.now().isoformat(),
        "trials": completed_trials,
        "ranked_trials": ranked_trials,
        "best_trial": ranked_trials[0] if ranked_trials else None,
    }

    write_json(
        report,
        output_dir / "grid_search_report.json",
    )


def run_search(
    config_path: str,
    output_dir: Path,
    *,
    restart: bool = False,
) -> None:
    """Run the grid search, resuming completed trials unless restarted."""

    base_config = load_config(config_path)
    trials = build_trials()

    report_path = output_dir / "grid_search_report.json"

    completed_trials: list[dict] = []

    if report_path.exists() and not restart:
        try:
            with report_path.open("r", encoding="utf-8") as file:
                previous_report = json.load(file)

            previous_trials = previous_report.get("trials", [])

            # Resume only completed trials with IDs in the current grid.
            valid_ids = {
                f"trial_{index:03d}"
                for index in range(1, len(trials) + 1)
            }

            completed_trials = [
                trial
                for trial in previous_trials
                if (
                    trial.get("trial_id") in valid_ids
                    and trial.get("status") == "completed"
                )
            ]

            logger.info(
                "Resuming search: %d completed trials found.",
                len(completed_trials),
            )

        except (json.JSONDecodeError, OSError) as error:
            logger.warning(
                "Could not read previous report (%s). Starting fresh.",
                error,
            )
            completed_trials = []

    completed_ids = {
        trial["trial_id"]
        for trial in completed_trials
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Base configuration: %s", config_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Total grid combinations: %d", len(trials))
    logger.info("Completed trials to skip: %d", len(completed_ids))

    for index, parameters in enumerate(trials, start=1):
        trial_id = f"trial_{index:03d}"

        if trial_id in completed_ids:
            logger.info(
                "[%d/%d] Skipping %s; already completed.",
                index,
                len(trials),
                trial_id,
            )
            continue

        logger.info(
            "[%d/%d] Starting %s with parameters: %s",
            index,
            len(trials),
            trial_id,
            parameters,
        )

        trial_config = deepcopy(base_config)
        training_config = trial_config["training"]

        # Apply this trial's hyperparameters.
        training_config["learning_rate"] = parameters["learning_rate"]
        training_config["weight_decay"] = parameters["weight_decay"]
        training_config["mask_probability"] = parameters["mask_probability"]
        training_config["batch_size"] = parameters["batch_size"]

        start_time = time.perf_counter()

        try:
            # Crucially, this writes no trial-specific artifacts.
            result = train_bert4rec(
                trial_config,
                save_artifacts=False,
            )

            elapsed_seconds = time.perf_counter() - start_time
            history = result.get("training_history", [])

            best_epoch = result.get("best_epoch", 0)

            best_record = (
                history[best_epoch - 1]
                if best_epoch > 0 and best_epoch <= len(history)
                else {}
            )

            # The trainer's best NDCG@10 and the best history row should
            # agree; use the returned metric as the primary value.
            best_ndcg10 = result.get(
                "best_ndcg_10",
                best_record.get("NDCG@10"),
            )

            trial_result = {
                "trial_id": trial_id,
                **parameters,
                "best_epoch": best_epoch,
                "best_ndcg_10": best_ndcg10,
                "HR@5": best_record.get("HR@5"),
                "HR@10": best_record.get("HR@10"),
                "NDCG@5": best_record.get("NDCG@5"),
                "NDCG@10": best_record.get("NDCG@10"),
                "completed_epochs": len(history),
                "total_training_time_seconds": elapsed_seconds,
                "average_epoch_time_seconds": (
                    elapsed_seconds / len(history)
                    if history
                    else None
                ),
                "status": "completed",
            }

            completed_trials.append(trial_result)

            logger.info(
                "%s completed | best NDCG@10=%.5f | epoch=%d | "
                "time=%.1f minutes",
                trial_id,
                best_ndcg10 if best_ndcg10 is not None else float("nan"),
                best_epoch,
                elapsed_seconds / 60.0,
            )

        except Exception as error:
            elapsed_seconds = time.perf_counter() - start_time

            logger.exception("%s failed.", trial_id)

            # Save a failed-trial record so the issue is visible.
            # Failed trials are intentionally retried on the next run.
            failed_trial = {
                "trial_id": trial_id,
                **parameters,
                "best_epoch": None,
                "best_ndcg_10": None,
                "HR@5": None,
                "HR@10": None,
                "NDCG@5": None,
                "NDCG@10": None,
                "completed_epochs": 0,
                "total_training_time_seconds": elapsed_seconds,
                "average_epoch_time_seconds": None,
                "status": "failed",
                "error": str(error),
            }

            # Keep the failure in the report but replace it if the trial
            # succeeds on a later resume.
            completed_trials = [
                trial
                for trial in completed_trials
                if trial.get("trial_id") != trial_id
            ]
            completed_trials.append(failed_trial)

        # Persist progress after each trial so interruption loses
        # at most the currently running trial.
        save_reports(
            completed_trials,
            output_dir,
            config_path,
        )

    save_reports(
        completed_trials,
        output_dir,
        config_path,
    )

    successful = [
        trial
        for trial in completed_trials
        if trial.get("status") == "completed"
    ]

    if successful:
        best_trial = max(
            successful,
            key=lambda trial: trial["best_ndcg_10"],
        )

        logger.info("Grid search finished.")
        logger.info("Successful trials: %d/%d", len(successful), len(trials))
        logger.info("Best trial: %s", best_trial["trial_id"])
        logger.info("Best parameters: %s", {
            key: best_trial[key]
            for key in SEARCH_SPACE
        })
        logger.info(
            "Best validation NDCG@10: %.5f",
            best_trial["best_ndcg_10"],
        )
    else:
        logger.warning(
            "No successful trials. Check the error logs and report."
        )

    logger.info("Aggregate reports saved to: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a compact BERT4Rec hyperparameter grid search."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bert4rec_tuned.yaml",
        help="Path to the base BERT4Rec YAML configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/bert4rec_grid_search",
        help="Directory for aggregate search reports.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard previous search progress and start fresh.",
    )

    args = parser.parse_args()

    setup_logging()

    run_search(
        config_path=args.config,
        output_dir=Path(args.output_dir),
        restart=args.restart,
    )


if __name__ == "__main__":
    main()
