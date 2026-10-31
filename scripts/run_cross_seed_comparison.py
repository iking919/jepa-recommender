
"""
Cross-seed comparison for causal JEPA and BERT4Rec.

Outputs:
    seed_results.csv
    metric_summary.csv
    cross_seed_report.json

The original YAML configuration is never modified. Each run receives a
deep copy with only its random seed changed.

JEPA:
    Uses src.training.train_jepa_causal.train(config)

BERT4Rec:
    Uses src.training.train_bert4rec.train_bert4rec(
        config, save_artifacts=False
    )

Reports go to results/<dataset>/<model>_seed_search/ unless --output-dir is
given. JEPA's per-seed trainer artifacts go to a reused ``_scratch/`` folder
inside that directory (no checkpoints), so this script never overwrites the
real experiment under results/<dataset>/<experiment>/.

Example:
    python -m scripts.run_cross_seed_comparison \
        --config configs/ml-1m/jepa_causal_tuned.yaml \
        --dataset amazon_beauty \
        --allow-existing

    python -m scripts.run_cross_seed_comparison \
        --config configs/ml-1m/bert4rec_tuned.yaml \
        --allow-existing
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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.paths import scratch_dir, search_dir  # noqa: E402

LOGGER = logging.getLogger("cross_seed_comparison")

DEFAULT_SEEDS = [42, 123, 456, 789, 2026]

METRICS = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def json_safe(value: Any) -> Any:
    """Convert common Python/numpy values to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write atomically to reduce the risk of corrupting progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, allow_nan=False)
        file.write("\n")

    temporary_path.replace(path)


def detect_model(config: dict[str, Any], requested: str) -> str:
    """Detect the trainer from the configuration, or honor --model."""
    if requested != "auto":
        return requested

    model = config.get("model", {})
    training = config.get("training", {})

    if not isinstance(model, dict):
        model = {}

    if not isinstance(training, dict):
        training = {}

    # Causal JEPA explicitly enables this setting in its YAML.
    if model.get("causal") is True:
        return "jepa"

    # BERT4Rec uses masked-item training.
    if "mask_probability" in training:
        return "bert4rec"

    # Also accept explicit model names in common configuration locations.
    candidates = [
        config.get("model_name"),
        config.get("model_type"),
        model.get("name"),
        model.get("type"),
        model.get("model_name"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str):
            name = candidate.lower().replace("-", "").replace("_", "")
            if "bert4rec" in name:
                return "bert4rec"
            if "jepa" in name:
                return "jepa"

    raise ValueError(
        "Could not identify the model from the YAML. "
        "Pass --model jepa or --model bert4rec explicitly."
    )


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def get_metric(metrics: dict[str, Any], name: str) -> float | None:
    """Read metric names from common trainer output conventions."""
    aliases = {
        "HR@5": ["HR@5", "hr@5", "hr@5.0", "hr5", "hr_5", "hr@5"],
        "HR@10": ["HR@10", "hr@10", "hr10", "hr_10"],
        "NDCG@5": ["NDCG@5", "ndcg@5", "ndcg5", "ndcg_5"],
        "NDCG@10": [
            "NDCG@10", "ndcg@10", "ndcg10", "ndcg_10"
        ],
    }

    normalized = {
        str(key).strip().lower(): value
        for key, value in metrics.items()
    }

    for alias in aliases[name]:
        value = normalized.get(alias.lower())
        if finite_number(value):
            return float(value)

    return None


def extract_bert_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics at the best validation epoch."""
    history = result.get("training_history", [])
    best_epoch = result.get("best_epoch", 0)

    best_record: dict[str, Any] = {}

    if (
        isinstance(history, list)
        and isinstance(best_epoch, int)
        and 0 < best_epoch <= len(history)
        and isinstance(history[best_epoch - 1], dict)
    ):
        best_record = history[best_epoch - 1]

    # Prefer the trainer's explicitly returned best NDCG@10.
    metrics = dict(best_record)

    best_ndcg = result.get("best_ndcg_10")
    if finite_number(best_ndcg):
        metrics["NDCG@10"] = float(best_ndcg)

    return {
        "best_epoch": best_epoch,
        "completed_epochs": len(history) if isinstance(history, list) else None,
        **{metric: get_metric(metrics, metric) for metric in METRICS},
    }


def extract_jepa_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the best validation metrics from the JEPA trainer summary."""
    best_metrics = result.get("best_metrics", {})

    if not isinstance(best_metrics, dict):
        best_metrics = {}

    best_epoch = result.get("best_epoch")
    completed_epochs = result.get("completed_epochs")

    return {
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        **{
            metric: get_metric(best_metrics, metric)
            for metric in METRICS
        },
    }


def train_one_seed(
    model_type: str,
    base_config: dict[str, Any],
    seed: int,
    scratch: Path,
) -> dict[str, Any]:
    """Train one model with a copied config and a different seed."""
    trial_config = copy.deepcopy(base_config)
    trial_config["seed"] = seed

    if model_type == "jepa":
        # The JEPA trainer always writes history/summary/config files.
        # Send them to a scratch folder (and skip checkpoints) so seed runs
        # cannot overwrite the real experiment's results or checkpoint.
        trial_config["training"]["save_checkpoint"] = False
        trial_config["output"]["results_dir"] = scratch.as_posix()
        trial_config["output"]["checkpoint_dir"] = None
        trial_config["output"]["best_checkpoint_name"] = None

    start = time.perf_counter()

    if model_type == "jepa":
        from src.training.train_jepa_causal import train

        result = train(trial_config)

        if not isinstance(result, dict):
            raise TypeError(
                "JEPA train(config) must return a dictionary."
            )

        extracted = extract_jepa_metrics(result)

        score = result.get(
            "best_selection_score",
            result.get("best_score"),
        )

        if not finite_number(score):
            score = extracted.get("NDCG@10")

        extracted["best_selection_score"] = (
            float(score) if finite_number(score) else None
        )

        timing = result.get("timing", {})
        if not isinstance(timing, dict):
            timing = {}

    else:
        from src.training.train_bert4rec import train_bert4rec

        result = train_bert4rec(
            trial_config,
            save_artifacts=False,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "train_bert4rec() must return a dictionary."
            )

        extracted = extract_bert_metrics(result)
        extracted["best_selection_score"] = extracted["NDCG@10"]
        timing = {}

    elapsed = time.perf_counter() - start

    return {
        "seed": seed,
        "status": "SUCCESS",
        **extracted,
        "elapsed_seconds": elapsed,
        "average_epoch_seconds": (
            elapsed / extracted["completed_epochs"]
            if isinstance(extracted.get("completed_epochs"), int)
            and extracted["completed_epochs"] > 0
            else None
        ),
        "trainer_timing": json_safe(timing),
        "error": None,
    }


def aggregate_results(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate descriptive statistics over successful runs."""
    successful = [
        row for row in records
        if row.get("status") == "SUCCESS"
    ]

    summary = []

    fields = [
        *METRICS,
        "best_selection_score",
        "best_epoch",
        "completed_epochs",
        "elapsed_seconds",
        "average_epoch_seconds",
    ]

    for metric in fields:
        values = [
            float(row[metric])
            for row in successful
            if finite_number(row.get(metric))
        ]

        if not values:
            continue

        summary.append({
            "metric": metric,
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": (
                statistics.stdev(values)
                if len(values) >= 2
                else None
            ),
            "min": min(values),
            "max": max(values),
        })

    return summary


def save_reports(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    model_type: str,
    config_path: Path,
    seeds: list[int],
    started_at: str,
    completed: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep one row per seed; omit nested objects from the CSV.
    flat_records = []
    for record in records:
        flat_records.append({
            key: value
            for key, value in record.items()
            if key not in {"trainer_timing"}
        })

    pd.DataFrame(flat_records).to_csv(
        output_dir / "seed_results.csv",
        index=False,
    )

    metric_summary = aggregate_results(records)

    pd.DataFrame(metric_summary).to_csv(
        output_dir / "metric_summary.csv",
        index=False,
    )

    successful = [
        row for row in records
        if row.get("status") == "SUCCESS"
    ]

    failed = [
        row for row in records
        if row.get("status") == "FAILED"
    ]

    report = {
        "experiment": "cross-seed comparison",
        "model": model_type,
        "config_path": str(config_path),
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed": completed,
        "requested_seeds": seeds,
        "total_seeds": len(seeds),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "selection_metric": "validation NDCG@10",
        "test_set_evaluated": False,
        "config_changed": "seed only",
        "metric_summary": metric_summary,
        "best_seed_by_validation_ndcg10": (
            max(
                (
                    row for row in successful
                    if finite_number(row.get("best_selection_score"))
                ),
                key=lambda row: row["best_selection_score"],
                default=None,
            )
        ),
        "runs": records,
    }

    write_json(output_dir / "cross_seed_report.json", report)


def run(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_config = load_config(config_path, dataset=args.dataset)
    dataset = base_config["dataset"]
    model_type = detect_model(base_config, args.model)

    output_dir = args.output_dir or search_dir(
        dataset, f"{model_type}_seed_search"
    )
    output_dir = (
        output_dir
        if output_dir.is_absolute()
        else PROJECT_ROOT / output_dir
    )
    output_dir = output_dir.resolve()
    scratch = scratch_dir(output_dir)

    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")

    report_path = output_dir / "cross_seed_report.json"

    if report_path.exists() and not args.allow_existing:
        raise SystemExit(
            f"Report already exists: {report_path}\n"
            "Choose a new output directory or pass --allow-existing."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    started_at = datetime.now().astimezone().isoformat()

    # Resume successful seeds only when the existing report matches.
    if report_path.exists() and not args.restart:
        try:
            with report_path.open("r", encoding="utf-8") as file:
                previous = json.load(file)

            compatible = (
                previous.get("model") == model_type
                and previous.get("config_path") == str(config_path)
                and previous.get("requested_seeds") == seeds
            )

            if not compatible:
                raise ValueError(
                    "Existing report uses a different model, config path, "
                    "or seed list. Use --restart or a new output directory."
                )

            records = [
                row for row in previous.get("runs", [])
                if (
                    isinstance(row, dict)
                    and row.get("status") == "SUCCESS"
                    and row.get("seed") in seeds
                )
            ]

            started_at = previous.get("started_at", started_at)
            LOGGER.info("Resuming: %d successful runs loaded.", len(records))

        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and "different model" in str(exc):
                raise
            LOGGER.warning("Could not resume report: %s", exc)
            records = []

    completed_seeds = {
        row["seed"] for row in records
    }

    LOGGER.info("Model: %s", model_type)
    LOGGER.info("Dataset: %s", dataset)
    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("Seeds: %s", seeds)

    save_reports(
        output_dir,
        records,
        model_type=model_type,
        config_path=config_path,
        seeds=seeds,
        started_at=started_at,
        completed=False,
    )

    for index, seed in enumerate(seeds, start=1):
        if not args.no_resume and seed in completed_seeds:
            LOGGER.info(
                "[%d/%d] Skipping seed %s; already successful.",
                index, len(seeds), seed,
            )
            continue

        LOGGER.info(
            "[%d/%d] Running %s with seed=%s",
            index, len(seeds), model_type, seed,
        )

        # Remove any earlier failed record for this seed before retrying.
        records = [
            row for row in records
            if row.get("seed") != seed
        ]

        try:
            record = train_one_seed(
                model_type, base_config, seed, scratch
            )

            if record.get("NDCG@10") is None:
                raise ValueError(
                    "No finite validation NDCG@10 was returned. "
                    "Check the trainer's metric keys and return structure."
                )

            records.append(record)

            LOGGER.info(
                "Seed %s complete | NDCG@10=%.5f | time=%.1f min",
                seed,
                record["NDCG@10"],
                record["elapsed_seconds"] / 60,
            )

        except Exception as exc:
            LOGGER.exception("Seed %s failed.", seed)

            records.append({
                "seed": seed,
                "status": "FAILED",
                "best_epoch": None,
                "completed_epochs": None,
                **{metric: None for metric in METRICS},
                "best_selection_score": None,
                "elapsed_seconds": None,
                "average_epoch_seconds": None,
                "trainer_timing": {},
                "error": f"{type(exc).__name__}: {exc}",
            })

        finally:
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            save_reports(
                output_dir,
                records,
                model_type=model_type,
                config_path=config_path,
                seeds=seeds,
                started_at=started_at,
                completed=False,
            )

    save_reports(
        output_dir,
        records,
        model_type=model_type,
        config_path=config_path,
        seeds=seeds,
        started_at=started_at,
        completed=True,
    )

    summary = aggregate_results(records)
    ndcg_summary = next(
        (row for row in summary if row["metric"] == "NDCG@10"),
        None,
    )

    successful_count = sum(
        row.get("status") == "SUCCESS" for row in records
    )

    LOGGER.info(
        "Finished: %d/%d successful runs.",
        successful_count, len(seeds),
    )

    if ndcg_summary:
        LOGGER.info(
            "Validation NDCG@10: mean=%.5f | median=%.5f | "
            "std=%s | min=%.5f | max=%.5f",
            ndcg_summary["mean"],
            ndcg_summary["median"],
            (
                f'{ndcg_summary["std"]:.5f}'
                if ndcg_summary["std"] is not None
                else "N/A"
            ),
            ndcg_summary["min"],
            ndcg_summary["max"],
        )

    LOGGER.info("Reports saved to %s", output_dir)

    return 0 if successful_count == len(seeds) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recommender performance across random seeds."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Existing JEPA or BERT4Rec YAML configuration.",
    )

    parser.add_argument(
        "--model",
        choices=["auto", "jepa", "bert4rec"],
        default="auto",
        help="Trainer selection. Auto-detected by default.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset (a file in configs/datasets/). Overrides the config's "
             "`dataset:` key.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregate cross-seed reports. "
             "Default: results/<dataset>/<model>_seed_search",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS}).",
    )

    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Permit using an existing report directory.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard previous progress and start a fresh comparison.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun seeds even if successful results already exist.",
    )

    return parser.parse_args()


def main() -> int:
    configure_logging()
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
