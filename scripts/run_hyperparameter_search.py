"""
Comprehensive Hyperparameter Grid Search for Causal JEPA
- Searches across constant and cosine schedules, learning rates, weight decays, and warmups.
- Saves disk space by deleting intermediate checkpoints, keeping only the best 
  checkpoint for 'constant' and the best checkpoint for 'cosine'.
- Maintains a clean master CSV table of all trial performances.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so 'src' can be found when running from scripts/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import copy
import itertools
import json
import logging
from datetime import datetime

import pandas as pd
import yaml

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
    
    # Track the best trial per schedule to know which checkpoints to keep
    best_by_schedule = {
        "constant": {"score": float("-inf"), "trial_record": None, "exp_name": None},
        "cosine": {"score": float("-inf"), "trial_record": None, "exp_name": None},
    }
    
    overall_best_score = float("-inf")
    overall_best_trial = {}

    search_start_time = datetime.now()

    for idx, params in enumerate(combinations, start=1):
        logger.info("=" * 80)
        logger.info("Running Trial %d/%d with parameters: %s", idx, len(combinations), params)
        logger.info("=" * 80)

        config = copy.deepcopy(base_config)

        # Apply hyperparameters
        config["training"]["lr_schedule"] = params["lr_schedule"]
        config["training"]["learning_rate"] = params["learning_rate"]
        config["training"]["weight_decay"] = params["weight_decay"]
        config["training"]["warmup_epochs"] = params["warmup_epochs"]

        # Construct a unique experiment name
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

        checkpoint_path = Path(config["output"]["checkpoint_dir"]) / config["output"]["best_checkpoint_name"]

        try:
            # Execute training run
            summary = train(config)

            best_sel_score = summary.get("best_selection_score", 0.0)
            best_metrics = summary.get("best_metrics", {})
            best_epoch = summary.get("best_epoch", 0)
            schedule = params["lr_schedule"]

            trial_record = {
                "trial_id": idx,
                "experiment_name": exp_name,
                "lr_schedule": schedule,
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

            # Update schedule-specific best
            if best_sel_score > best_by_schedule[schedule]["score"]:
                # If there was a previous best for this schedule, clean up its checkpoint file to save space
                prev_exp = best_by_schedule[schedule]["exp_name"]
                if prev_exp:
                    prev_ckpt = Path(f"checkpoints/{prev_exp}/{prev_exp}_best.pt")
                    if prev_ckpt.exists():
                        prev_ckpt.unlink()
                        logger.info("Removed non-optimal checkpoint: %s", prev_ckpt)

                best_by_schedule[schedule]["score"] = best_sel_score
                best_by_schedule[schedule]["trial_record"] = trial_record.copy()
                best_by_schedule[schedule]["exp_name"] = exp_name
                logger.info(">>> New best %s model found! Selection Score: %.4f", schedule, best_sel_score)
            else:
                # Not the best for this schedule; remove its checkpoint immediately to save space
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info("Removed intermediate checkpoint to save space: %s", checkpoint_path)

            # Update overall best
            if best_sel_score > overall_best_score:
                overall_best_score = best_sel_score
                overall_best_trial = trial_record.copy()

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
            # Remove checkpoint if it exists from partial failure
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        all_trials.append(trial_record)

        # Incrementally update the master CSV table after each trial
        df_trials = pd.DataFrame(all_trials)
        df_trials.to_csv(search_results_dir / "search_summary.csv", index=False)

    # Save final JSON summary report containing best models per schedule and overall
    search_summary = {
        "search_start": search_start_time.isoformat(),
        "search_end": datetime.now().isoformat(),
        "total_trials": len(combinations),
        "overall_best_trial": overall_best_trial,
        "best_by_schedule": {
            sched: data["trial_record"] for sched, data in best_by_schedule.items()
        },
        "all_trials": all_trials,
    }

    with open(search_results_dir / "grid_search_report.json", "w") as f:
        json.dump(search_summary, f, indent=4)

    logger.info("=" * 80)
    logger.info("GRID SEARCH COMPLETE")
    logger.info("Overall Best Experiment: %s", overall_best_trial.get("experiment_name"))
    logger.info("Overall Best Validation NDCG@10: %.4f", overall_best_trial.get("ndcg_10", 0.0))
    logger.info("Saved Checkpoints Retained Only For Best Models:")
    for sched, data in best_by_schedule.items():
        if data["exp_name"]:
            logger.info("  - %s schedule best: checkpoints/%s/%s_best.pt", sched, data["exp_name"], data["exp_name"])
    logger.info("Master table log saved to: %s", search_results_dir / "search_summary.csv")
    logger.info("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run causal JEPA hyperparameter grid search with optimized storage.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base YAML configuration file.",
    )
    args = parser.parse_args()

    run_grid_search(args.config)