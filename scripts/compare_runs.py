"""
Compare training runs by time-to-quality rather than per-epoch time.

Different training schemes have very different epoch costs, so epoch counts
and seconds-per-epoch are not comparable. This reports, for each run, the
best validation NDCG@10 and how long it took to first reach a given fraction
of the reference run's best.

Usage:
    python -m scripts.compare_runs \
        --reference results/jepa_tuned \
        --candidates results/jepa_causal results/jepa_tuned_fast
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

FRACTIONS = (0.90, 0.95, 0.99, 1.00)


def load_history(results_dir: str) -> list[dict]:
    path = Path(results_dir) / "training_history.csv"
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def time_to_reach(history: list[dict], target: float) -> float | None:
    """Cumulative epoch time (train + validation) when NDCG@10 first >= target."""

    elapsed = 0.0
    for row in history:
        elapsed += row["epoch_time_seconds"]
        if row["NDCG@10"] >= target:
            return elapsed
    return None


def describe(name: str, history: list[dict], reference_best: float) -> None:
    best_row = max(history, key=lambda row: row["NDCG@10"])
    total = sum(row["epoch_time_seconds"] for row in history)
    median_epoch = sorted(row["epoch_time_seconds"] for row in history)[len(history) // 2]

    print(f"\n{name}")
    print(f"  epochs run          : {len(history)}")
    print(f"  best val NDCG@10    : {best_row['NDCG@10']:.4f} (epoch {int(best_row['epoch'])})")
    print(f"  best val HR@10      : {best_row['HR@10']:.4f}")
    print(f"  median epoch time   : {median_epoch:.1f}s")
    print(f"  total time          : {total / 60:.1f} min")

    for fraction in FRACTIONS:
        target = fraction * reference_best
        reached = time_to_reach(history, target)
        label = f"{int(fraction * 100)}% of reference best ({target:.4f})"
        print(
            f"  time to {label:<40}: "
            + (f"{reached / 60:.1f} min" if reached is not None else "not reached")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs by time-to-quality.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()

    reference = load_history(args.reference)
    reference_best = max(row["NDCG@10"] for row in reference)

    print(f"Reference: {args.reference} (best val NDCG@10 {reference_best:.4f})")
    print("Validation noise: ~0.003 NDCG@10 for 6,040 users; treat gaps below that as ties.")
    print("Wall-clock times are inflated if another job shared the GPU.")

    describe(f"{args.reference} (reference)", reference, reference_best)
    for candidate in args.candidates:
        describe(candidate, load_history(candidate), reference_best)


if __name__ == "__main__":
    main()
