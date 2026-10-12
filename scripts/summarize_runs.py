"""
Summarise several runs (for example one config over several seeds) with
mean and standard deviation, so differences can be judged against noise.

Reads results/<run>/run_summary.json (validation) and, if present,
results/<run>/test_results.json (written by run_jepa_evaluation).

Usage:
    python -m scripts.summarize_runs --runs "results/causal_const_s*" "results/causal_cos_s*"
Each pattern is one group; each group gets its own mean +/- std line.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

COLUMNS = (
    ("val NDCG@10", "val_ndcg10"),
    ("val HR@10", "val_hr10"),
    ("test NDCG@10", "test_ndcg10"),
    ("test HR@10", "test_hr10"),
    ("best epoch", "best_epoch"),
    ("minutes", "minutes"),
)


def load_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    row = {
        "val_ndcg10": summary["best_metrics"]["NDCG@10"],
        "val_hr10": summary["best_metrics"]["HR@10"],
        "best_epoch": summary["best_epoch"],
        "minutes": summary["timing"]["total_training_time_minutes"],
    }

    test_path = run_dir / "test_results.json"
    if test_path.exists():
        metrics = json.loads(test_path.read_text(encoding="utf-8"))["metrics"]
        row["test_ndcg10"] = metrics["NDCG@10"]
        row["test_hr10"] = metrics["HR@10"]

    return row


def fmt(values: list[float], key: str) -> str:
    if not values:
        return "n/a"
    if key in ("best_epoch", "minutes"):
        return f"{statistics.mean(values):.1f}"
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f}+/-{statistics.stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean/std over groups of runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Glob pattern(s), one per group.")
    args = parser.parse_args()

    print(f"{'group':<32}{'n':>3}" + "".join(f"{label:>22}" for label, _ in COLUMNS))

    for pattern in args.runs:
        run_dirs = sorted(Path(p) for p in glob.glob(pattern) if Path(p, "run_summary.json").exists())
        if not run_dirs:
            print(f"{pattern:<32}  0  (no completed runs match)")
            continue

        rows = [load_run(d) for d in run_dirs]
        cells = [fmt([r[key] for r in rows if key in r], key) for _, key in COLUMNS]
        print(f"{pattern:<32}{len(rows):>3}" + "".join(f"{c:>22}" for c in cells))

    print("\nstd is the sample standard deviation across runs; with n<=3 treat it as rough.")


if __name__ == "__main__":
    main()
