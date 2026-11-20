"""Summarize groups of runs with mean and standard deviation.

Useful for one config over several seeds, so differences can be judged
against run-to-run noise.

Reads, for each run folder:
    results/<dataset>/<run>/run_summary.json   validation metrics
    results/<dataset>/<run>/test_results.json  test metrics, if present
                                               (written by the evaluation
                                               scripts)

Usage (from the repository root):
    python -m scripts.summarize_runs --runs \
        "results/ml-1m/causal_const_s*" "results/ml-1m/causal_cos_s*"

Each pattern is one group and gets its own mean +/- std line.
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
    """Read one run's validation (and test, if present) metrics."""

    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    row = {
        "val_ndcg10": summary["best_metrics"]["NDCG@10"],
        "val_hr10": summary["best_metrics"]["HR@10"],
        "best_epoch": summary["best_epoch"],
        "minutes": summary["timing"]["total_training_time_minutes"],
    }

    test_path = run_dir / "test_results.json"
    if test_path.exists():
        test_results = json.loads(test_path.read_text(encoding="utf-8"))
        metrics = test_results["metrics"]
        row["test_ndcg10"] = metrics["NDCG@10"]
        row["test_hr10"] = metrics["HR@10"]

    return row


def format_cell(values: list[float], key: str) -> str:
    """Format a column as ``mean`` or ``mean+/-std``."""

    if not values:
        return "n/a"

    mean = statistics.mean(values)

    if key in ("best_epoch", "minutes"):
        return f"{mean:.1f}"

    if len(values) == 1:
        return f"{mean:.4f}"

    return f"{mean:.4f}+/-{statistics.stdev(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean/std over groups of runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Glob pattern(s), one per group.",
    )
    args = parser.parse_args()

    header = f"{'group':<32}{'n':>3}"
    header += "".join(f"{label:>22}" for label, _ in COLUMNS)
    print(header)

    for pattern in args.runs:
        run_dirs = sorted(
            Path(path)
            for path in glob.glob(pattern)
            if Path(path, "run_summary.json").exists()
        )

        if not run_dirs:
            print(f"{pattern:<32}  0  (no completed runs match)")
            continue

        rows = [load_run(run_dir) for run_dir in run_dirs]
        cells = [
            format_cell([row[key] for row in rows if key in row], key)
            for _, key in COLUMNS
        ]
        print(
            f"{pattern:<32}{len(rows):>3}"
            + "".join(f"{cell:>22}" for cell in cells)
        )

    print(
        "\nstd is the sample standard deviation across runs; "
        "with n<=3 treat it as rough."
    )


if __name__ == "__main__":
    main()
