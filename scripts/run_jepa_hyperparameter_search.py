"""Resumable hyperparameter search for causal one-pass JEPA.

Place this file in the repository's scripts/ directory. It assumes the project
exposes ``train(config: dict)`` from ``src.training.train_jepa_causal`` and that
training reads ``training.window_stride`` and ``model.hidden_dim``. Verify these
field names against your actual config/trainer before running a real search.

Example:
    python -m scripts.run_jepa_hyperparameter_search \
        --config configs/jepa_tuned_causal.yaml

The search is a Cartesian grid. Keep the default grid small; uncomment or add
values in SEARCH_SPACE deliberately because the number of trials multiplies.
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

LOGGER = logging.getLogger("jepa_hyperparameter_search")

# ---------------------------------------------------------------------------
# SEARCH BOX: edit this section to choose what gets tuned.
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, list[Any]] = {
    # Optimizer / schedule
    # "lr_schedule": ["constant"],
    "lr_schedule": ["constant", "cosine"],
    "learning_rate": [5e-4],
    # "learning_rate": [5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
    "weight_decay": [0.01],
    # "weight_decay": [0.01, 0.05],
    "warmup_epochs": [0],
    # "warmup_epochs": [0, 5, 10],
    "ema_decay": [0.996],
    # "ema_decay": [0.90, 0.97, 0.990, 0.996, 0.999],
    "dropout": [0.2],
    # "dropout": [0.1, 0.2, 0.3, 0.4],

    # Window stride and latent dimension
    # "window_stride": [100],
    "window_stride": [25, 50, 100, 150],
    # "hidden_dim": [256],
    "hidden_dim": [128, 256, 384],

    # Architecture
    "num_layers": [4],
    # "num_layers": [2, 3, 4],
    "num_heads": [8],
    # "num_heads": [4, 8],
    "feed_forward_dim": [1024],
    # "feed_forward_dim": [512, 768, 1024, 1536],

    # Throughput and sequence length
    "batch_size": [512],
    # "batch_size": [32, 64, 128, 256, 512],
    "max_seq_len": [200],
    # "max_seq_len": [100, 150, 200],

    # Optional dimensions
    # "min_lr_ratio": [0.0, 0.01],
    # "selection_window": [1, 3],
}

# Parameter-to-config mappings. Confirm these against your YAML and trainer.
# Parameters with None are recorded in the trial metadata but not applied.
CONFIG_PATHS: dict[str, tuple[str, ...] | None] = {
    "lr_schedule": ("training", "lr_schedule"),
    "learning_rate": ("training", "learning_rate"),
    "weight_decay": ("training", "weight_decay"),
    "warmup_epochs": ("training", "warmup_epochs"),
    "ema_decay": ("training", "ema_decay"),
    "dropout": ("model", "dropout"),
    "window_stride": ("training", "window_stride"),
    "hidden_dim": ("model", "hidden_dim"),
    "num_layers": ("model", "num_layers"),
    "num_heads": ("model", "num_heads"),
    "ff_dim": ("model", "ff_dim"),
    "batch_size": ("training", "batch_size"),
    "max_seq_len": ("model", "max_seq_len"),
}

PARAM_KEYS = tuple(SEARCH_SPACE.keys())


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], force=True)


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
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, allow_nan=False)
        f.write("\n")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    for section in ("model", "training", "output"):
        config.setdefault(section, {})
    return config


def apply_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = config
    for key in path[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def trial_name(params: dict[str, Any], seed: int) -> str:
    def label(v: Any) -> str:
        return str(v).replace(".", "p").replace("-", "m")
    compact = "_".join(f"{k}-{label(v)}" for k, v in params.items())
    return f"jepa_{compact}_seed-{seed}"


def make_trial_config(base: dict[str, Any], params: dict[str, Any], seed: int,
                      name: str, trial_dir: Path, save_checkpoints: bool) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["seed"] = seed
    for key, value in params.items():
        path = CONFIG_PATHS.get(key)
        if path is not None:
            apply_nested(config, path, value)
    training = config["training"]
    training["save_checkpoint"] = bool(save_checkpoints)
    output = config["output"]
    output["experiment_name"] = name
    output["results_dir"] = str(trial_dir)
    if not save_checkpoints:
        output["checkpoint_dir"] = None
        output["best_checkpoint_name"] = None
    # A cosine schedule is not meaningfully tested if early stopping prevents
    # the schedule from running; allow it to reach its configured final epoch.
    if params.get("lr_schedule") == "cosine":
        epochs = training.get("num_epochs", training.get("epochs"))
        if epochs is not None:
            training["patience"] = max(int(training.get("patience", 0)), int(epochs))
    return config


def metric_summary(summary: Any) -> tuple[dict[str, Any], float | None]:
    if not isinstance(summary, dict):
        raise TypeError(f"train(config) must return dict, got {type(summary).__name__}")
    metrics = summary.get("best_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    score = summary.get("best_selection_score", summary.get("best_score"))
    if score is None:
        score = metrics.get("ndcg@10", metrics.get("ndcg10"))
    return metrics, float(score) if finite_number(score) else None


def aggregate(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for t in trials:
        if t.get("status") == "SUCCESS" and finite_number(t.get("best_selection_score")):
            groups.setdefault(tuple(t[k] for k in PARAM_KEYS), []).append(t)
    rows = []
    for key, members in groups.items():
        scores = [float(m["best_selection_score"]) for m in members]
        rows.append({**dict(zip(PARAM_KEYS, key)), "n_seeds": len(scores),
                     "seeds": [m["seed"] for m in members],
                     "mean_score": statistics.fmean(scores),
                     "std_score": statistics.stdev(scores) if len(scores) > 1 else None,
                     "min_score": min(scores), "max_score": max(scores)})
    return sorted(rows, key=lambda x: x["mean_score"], reverse=True)


def save_report(out_dir: Path, trials: list[dict[str, Any]], space: dict[str, list[Any]],
                seeds: list[int], started: str, completed: bool, planned: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for t in trials:
        row = {k: v for k, v in t.items() if k != "best_metrics"}
        row.update({f"val_{k}": v for k, v in (t.get("best_metrics") or {}).items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / "search_summary.csv", index=False)
    ranked = aggregate(trials)
    pd.DataFrame([{k: v for k, v in r.items() if k != "seeds"} for r in ranked]).to_csv(
        out_dir / "config_summary.csv", index=False)
    report = {"experiment": "causal one-pass JEPA hyperparameter search",
              "started_at": started, "updated_at": datetime.now().astimezone().isoformat(),
              "completed": completed, "total_trials_planned": planned,
              "trials_recorded": len(trials),
              "successful_trials": sum(t.get("status") == "SUCCESS" for t in trials),
              "failed_trials": sum(t.get("status") == "FAILED" for t in trials),
              "selection_metric": "best_selection_score (expected validation NDCG@10)",
              "search_space": space, "seeds": seeds,
              "best_configuration_by_mean": ranked[0] if ranked else None,
              "configurations_ranked": ranked, "trials": trials}
    write_json(out_dir / "grid_search_report.json", report)


def release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    base = load_config(config_path)
    if not base.get("model", {}).get("causal", False):
        raise SystemExit("Base config must set model.causal: true for train_jepa_causal.")

    space = copy.deepcopy(SEARCH_SPACE)
    # CLI overrides let you expand selected dimensions without editing the file.
    overrides = {
        "learning_rate": args.lr,
        "ema_decay": args.ema,
        "dropout": args.dropout,
        "window_stride": args.stride,
        "hidden_dim": args.hidden_dim,
    }
    for key, values in overrides.items():
        if values is not None:
            space[key] = values
    empty = [k for k, vals in space.items() if not vals]
    if empty:
        raise ValueError(f"Search-space dimensions cannot be empty: {empty}")
    # Validate obvious attention constraint before launching expensive runs.
    if "hidden_dim" in space and "num_heads" in space:
        for hidden, heads in itertools.product(space["hidden_dim"], space["num_heads"]):
            if int(hidden) % int(heads) != 0:
                raise ValueError(f"hidden_dim={hidden} must be divisible by num_heads={heads}")

    out_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    seeds = list(dict.fromkeys(args.seeds))
    params_list = [dict(zip(space.keys(), vals)) for vals in itertools.product(*space.values())]
    plan = [(p, s) for p in params_list for s in seeds]
    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Output: %s", out_dir)
    LOGGER.info("Configurations=%d | seeds=%s | total trials=%d", len(params_list), seeds, len(plan))
    if args.dry_run:
        for i, (params, seed) in enumerate(plan, 1):
            print(f"{i:>4}/{len(plan)} {trial_name(params, seed)}")
        print(f"Dry run only: {len(plan)} trials planned; no training started.")
        return 0

    report_path = out_dir / "grid_search_report.json"
    if report_path.exists() and not args.allow_existing:
        raise SystemExit(f"Output report already exists: {report_path}. Choose --output-dir or pass --allow-existing.")
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.training.train_jepa_causal import train

    started = datetime.now().astimezone().isoformat()
    trials: list[dict[str, Any]] = []
    save_report(out_dir, trials, space, seeds, started, False, len(plan))
    for i, (params, seed) in enumerate(plan, 1):
        name = trial_name(params, seed)
        trial_dir = out_dir / name
        summary_path = trial_dir / "run_summary.json"
        record: dict[str, Any] = {"trial_number": i, "experiment_name": name,
            "trial_results_dir": str(trial_dir), **params, "seed": seed,
            "status": "RUNNING", "resumed": False, "best_selection_score": None,
            "best_metrics": {}, "error": None}
        trials.append(record)
        try:
            if not args.no_resume and summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                metrics, score = metric_summary(summary)
                if score is not None:
                    record.update(status="SUCCESS", resumed=True,
                                  best_metrics=json_safe(metrics), best_selection_score=score)
                    LOGGER.info("[%d/%d] Reused completed trial %s", i, len(plan), name)
                    save_report(out_dir, trials, space, seeds, started, False, len(plan))
                    continue
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_config = make_trial_config(base, params, seed, name, trial_dir, args.save_checkpoints)
            with (trial_dir / "config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(trial_config, f, sort_keys=False)
            LOGGER.info("[%d/%d] Training %s", i, len(plan), name)
            summary = train(trial_config)
            metrics, score = metric_summary(summary)
            if score is None:
                raise ValueError("No finite best_selection_score found in trainer summary")
            record.update(status="SUCCESS", best_metrics=json_safe(metrics),
                          best_selection_score=score,
                          best_epoch=summary.get("best_epoch"),
                          completed_epochs=summary.get("completed_epochs"),
                          timing=json_safe(summary.get("timing", {})))
        except Exception as exc:  # one failed run must not kill the full search
            LOGGER.exception("Trial failed: %s", name)
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        finally:
            release_gpu_memory()
            save_report(out_dir, trials, space, seeds, started, False, len(plan))
    save_report(out_dir, trials, space, seeds, started, True, len(plan))
    ranked = aggregate(trials)
    LOGGER.info("Search complete: %d successful, %d failed",
        sum(t.get("status") == "SUCCESS" for t in trials),
        sum(t.get("status") == "FAILED" for t in trials))
    if ranked:
        LOGGER.info("Best mean validation score: %.5f | config=%s", ranked[0]["mean_score"],
                    {k: ranked[0][k] for k in PARAM_KEYS})
    LOGGER.info("Reports: %s", out_dir)
    return 0 if ranked else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="hyperparameter search for causal one-pass JEPA")
    p.add_argument("--config", type=Path, default=Path("configs/jepa_tuned.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("results/jepa_hyperparameter_search"))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--lr", nargs="+", type=float, default=None)
    p.add_argument("--ema", nargs="+", type=float, default=None)
    p.add_argument("--dropout", nargs="+", type=float, default=None)
    p.add_argument("--stride", nargs="+", type=int, default=None)
    p.add_argument("--hidden-dim", nargs="+", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-checkpoints", action="store_true", help="Save checkpoints for every trial (uses more storage).")
    p.add_argument("--allow-existing", action="store_true", help="Allow writing into an existing output directory/report.")
    return p.parse_args()


def main() -> int:
    configure_logging()
    return run(parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
