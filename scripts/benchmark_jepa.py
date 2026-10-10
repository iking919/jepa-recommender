"""
Short, isolated speed benchmark for JEPA training options.

Each variant runs in its own subprocess (clean CUDA/compile state) for a few
epochs. Outputs go to ``checkpoints/benchmarks/<name>`` and
``results/benchmarks/<name>``, so real experiment artifacts are never touched.

Usage (from the repository root):
    python -m scripts.benchmark_jepa --config configs/jepa_tuned.yaml
    python -m scripts.benchmark_jepa --config configs/jepa_tuned.yaml --epochs 3 \
        --variants ref bf16
    python -m scripts.benchmark_jepa --config configs/jepa_tuned.yaml --dry-run

Epoch 1 includes one-time warmup (CUDA init, and torch.compile if enabled), so
compare the "steady" column, which averages epochs 2..N.
"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path

import yaml

# Each variant changes exactly one thing relative to ``ref`` (GPU loader, no
# trimming). Length bucketing only helps through trimming, so that variant
# turns both on.
VARIANTS: dict[str, dict] = {
    "ref": {},
    "trim": {"trim_padding": True},
    "bf16": {"amp_dtype": "bfloat16"},
    "compile": {"compile": True},
    "bf16_compile": {"amp_dtype": "bfloat16", "compile": True},
    "trim_bucket20": {"trim_padding": True, "length_bucketing": 20},
}

DEFAULTS = {
    "amp_dtype": "float16",
    "compile": False,
    "trim_padding": False,
    "length_bucketing": 0,
}


def build_config(base: dict, name: str, overrides: dict, epochs: int) -> dict:
    config = copy.deepcopy(base)
    training = config["training"]

    training.update(DEFAULTS)
    training.update(overrides)
    training["num_epochs"] = epochs
    training["patience"] = epochs + 1  # never early-stop a benchmark

    experiment = f"bench_{name}"
    config["output"] = {
        "experiment_name": experiment,
        "checkpoint_dir": f"checkpoints/benchmarks/{name}",
        "results_dir": f"results/benchmarks/{name}",
        "best_checkpoint_name": f"{experiment}.pt",
    }
    return config


def read_history(results_dir: Path) -> list[dict]:
    path = results_dir / "training_history.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JEPA training options.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print them without training.")
    args = parser.parse_args()

    if args.epochs < 2:
        parser.error("--epochs must be >= 2 so steady-state time can be measured.")

    with open(args.config, encoding="utf-8") as file:
        base = yaml.safe_load(file)

    rows = []
    for name in args.variants:
        config = build_config(base, name, VARIANTS[name], args.epochs)
        results_dir = Path(config["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        config_path = results_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(f"\n=== {name}: {VARIANTS[name]} ===", flush=True)
        if args.dry_run:
            print(yaml.safe_dump(config["training"], sort_keys=False))
            continue

        proc = subprocess.run(
            [sys.executable, "-m", "src.training.train_jepa", "--config", str(config_path)]
        )
        history = read_history(results_dir)

        if proc.returncode != 0 or len(history) < args.epochs:
            print(f"!! {name} FAILED or incomplete (exit code {proc.returncode})", flush=True)
            rows.append({"name": name, "failed": True})
            continue

        times = [float(r["train_time_seconds"]) for r in history]
        steady = sum(times[1:]) / len(times[1:])
        last = history[-1]
        rows.append(
            {
                "name": name,
                "failed": False,
                "epoch1": times[0],
                "steady": steady,
                "loss": float(last["loss"]),
                "ndcg10": float(last["NDCG@10"]),
                "mem": float(last["peak_gpu_memory_gb"]),
            }
        )

    if args.dry_run:
        return

    ok = {r["name"]: r for r in rows if not r["failed"]}
    base_steady = ok["ref"]["steady"] if "ref" in ok else None

    header = f"{'variant':<15}{'epoch1 (s)':>11}{'steady (s)':>12}{'vs ref':>9}{'loss':>9}{'NDCG@10':>10}{'peak GB':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        if r["failed"]:
            print(f"{r['name']:<15}{'FAILED':>11}")
            continue
        speedup = f"{base_steady / r['steady']:.2f}x" if base_steady else "n/a"
        print(
            f"{r['name']:<15}{r['epoch1']:>11.1f}{r['steady']:>12.1f}{speedup:>9}"
            f"{r['loss']:>9.4f}{r['ndcg10']:>10.4f}{r['mem']:>9.2f}"
        )
    print(
        f"\n(Your original tuned run averaged "
        f"~103s/epoch on the old DataLoader; compare only if the GPU and load are the same.)"
    )


if __name__ == "__main__":
    main()