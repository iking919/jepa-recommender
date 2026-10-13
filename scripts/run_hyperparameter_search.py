"""
Comprehensive Hyperparameter Grid Search for Causal JEPA

Searches across:
- lr_schedule: ["constant", "cosine"]
- learning_rate: [1e-4, 3e-4, 5e-4, 1e-3]
- weight_decay: [0.01, 0.05]
- warmup_epochs: [0, 5]

Logs all trial results to a master CSV and JSON summary, and preserves 
the best checkpoint.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Import the training function and config utilities from your codebase
from src.training.train_jepa_causal import train
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_grid_search(base_config_path: str) -> None:
    setup_logging()
    base_config = load_config(base_config_path)

    # Define the hyperparameter search space
    search_space = {
        "lr_schedule": ["constant", "cosine"],
        "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
        "weight_decay": [0.01, 0.05],
        "warmup_epochs": [0, 5],
    }

    # Generate all combinations (Cartesian product)
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    logger.info("Starting hyperparameter grid search over %d configurations.", len(combinations))

    search_results_dir = Path("results/jepa_grid_search")
    search_results_dir.mkdir(parents=True, exist_ok=True)

    all_trials = []
    best_overall_score = float("-inf")
    best_trial_info = {}

    search_start_time = datetime.now()

    for idx, params in enumerate(combinations, start=1):
        logger.info("=" * 80)
        logger.info("Running Trial %d/%d with parameters: %s", idx, len(combinations), params)
        logger.info("=" * 80)

        # Deepcopy base config to avoid mutating shared state
        config = copy.deepcopy(base_config)

        # Apply hyperparameters
        config["training"]["lr_schedule"] = params["lr_schedule"]
        config["training"]["learning_rate"] = params["learning_rate"]
        config["training"]["weight_decay"] = params["weight_decay"]
        config["training"]["warmup_epochs"] = params["warmup_epochs"]

        # Construct a unique experiment name for artifact isolation
        exp_name = (
            f"jepa_causal_{params['lr_schedule']}_"
            f"lr{params['learning_rate']}_"
            f"wd{params['weight_decay']}_"
            f"warm{params['warmup_epochs']}"
        )
        config["output"]["experiment_name"] = exp_name
        config["output"]["checkpoint_dir"] = f"checkpoints/{exp_name}"
        config["output"]["results_dir"] = f"results/{exp_name}"
        config["output"]["best_checkpoint_name"] = f"{exp_name}_best.pt"

        try:
            # Execute training run and capture run summary dictionary
            summary = train(config)

            best_sel_score = summary.get("best_selection_score", 0.0)
            best_metrics = summary.get("best_metrics", {})
            best_epoch = summary.get("best_epoch", 0)

            trial_record = {
                "trial_id": idx,
                "experiment_name": exp_name,
                "lr_schedule": params["lr_schedule"],
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "warmup_epochs": params["warmup_epochs"],
                "best_epoch": best_epoch,
                "selection_score": best_sel_score,
                "ndcg_10": best_metrics.get("NDCG@10", 0.0),
                "hr_10": best_metrics.get("HR@10", 0.0),
                "ndcg_5": best_metrics.get("NDCG@5", 0.0),
                "hr_5": best_metrics.get("HR@5", 0.0),
                "status": "SUCCESS",
            }

            if best_sel_score > best_overall_score:
                best_overall_score = best_sel_score
                best_trial_info = trial_record.copy()
                logger.info(">>> New best model found! Selection Score: %.4f", best_overall_score)

        except Exception as e:
            logger.exception("Trial %d failed with error: %s", idx, e)
            trial_record = {
                "trial_id": idx,
                "experiment_name": exp_name,
                "lr_schedule": params["lr_schedule"],
                "learning_rate": params["learning_rate"],
                "weight_decay": params["weight_decay"],
                "warmup_epochs": params["warmup_epochs"],
                "best_epoch": None,
                "selection_score": 0.0,
                "ndcg_10": 0.0,
                "hr_10": 0.0,
                "ndcg_5": 0.0,
                "hr_5": 0.0,
                "status": f"FAILED: {str(e)}",
            }

        all_trials.append(trial_record)

        # Incrementally update the master CSV log after each trial
        df_trials = pd.DataFrame(all_trials)
        df_trials.to_csv(search_results_dir / "search_summary.csv", index=False)

    # Save final master search summary JSON
    search_summary = {
        "search_start": search_start_time.isoformat(),
        "search_end": datetime.now().isoformat(),
        "total_trials": len(combinations),
        "best_trial": best_trial_info,
        "all_trials": all_trials,
    }

    with open(search_results_dir / "grid_search_report.json", "w") as f:
        json.dump(search_summary, f, indent=4)

    logger.info("=" * 80)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("Best Trial Experiment: %s", best_trial_info.get("experiment_name"))
    logger.info("Best Validation NDCG@10: %.4f", best_trial_info.get("ndcg_10", 0.0))
    logger.info("Master logs saved to: %s", search_results_dir / "search_summary.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run causal JEPA hyperparameter grid search.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML configuration file.",
    )
    args = parser.parse_args()

    run_grid_search(args.config)