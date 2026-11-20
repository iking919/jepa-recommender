"""Download and extract the MovieLens-1M dataset.

Usage (from the repository root):
    python -m scripts.download_ml1m
    python -m scripts.download_ml1m --raw-dir data/raw
    python -m scripts.download_ml1m --force

Output:
    data/raw/ml-1m/{ratings,movies,users}.dat
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import zipfile
from pathlib import Path

from src.data.download import download_file, get_ssl_context

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

EXPECTED_FILES = (
    "ratings.dat",
    "movies.dat",
    "users.dat",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract MovieLens-1M."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where ml-1m will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )

    return parser.parse_args()


def dataset_is_present(dataset_dir: Path) -> bool:
    return all((dataset_dir / filename).is_file() for filename in EXPECTED_FILES)


def extract_dataset(zip_path: Path, raw_dir: Path) -> Path:
    """Extract the archive and check that the expected files exist."""

    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)

    dataset_dir = raw_dir / "ml-1m"

    if not dataset_is_present(dataset_dir):
        raise RuntimeError(
            "Extraction completed, but the expected MovieLens-1M "
            "files were not found."
        )

    return dataset_dir


def main() -> int:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    dataset_dir = raw_dir / "ml-1m"
    zip_path = raw_dir / "ml-1m.zip"

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MovieLens-1M Downloader")
    print("=" * 70)

    if dataset_is_present(dataset_dir) and not args.force:
        print("\nMovieLens-1M is already present:")
        print(f"  {dataset_dir}")
        return 0

    if args.force and dataset_dir.exists():
        print("\nRemoving existing dataset...")
        shutil.rmtree(dataset_dir)

    context = get_ssl_context()

    try:
        download_file(DATASET_URL, zip_path, context)
        dataset_dir = extract_dataset(zip_path, raw_dir)

    except urllib.error.URLError as exc:
        print("\nERROR: Could not download MovieLens-1M.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print("\nTry installing/updating certifi:", file=sys.stderr)
        print("  pip install --upgrade certifi", file=sys.stderr)
        return 1

    except zipfile.BadZipFile:
        print(
            "\nERROR: The downloaded file is not a valid ZIP archive.",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    finally:
        if zip_path.exists():
            print(f"Removing temporary archive: {zip_path}")
            zip_path.unlink()

    print("\n" + "=" * 70)
    print("MovieLens-1M download complete.")
    print("=" * 70)

    print("\nDataset:")
    print(f"  {dataset_dir}")

    print("\nFiles:")
    for filename in EXPECTED_FILES:
        size_mb = (dataset_dir / filename).stat().st_size / 1024**2
        print(f"  ✓ {filename:<12} {size_mb:8.2f} MB")

    print("\nNext step:")
    print("  python -m scripts.preprocess_ml1m")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
