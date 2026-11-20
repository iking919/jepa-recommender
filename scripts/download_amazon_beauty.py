"""Download and extract the Stanford SNAP Amazon Beauty 5-core dataset.

Source:
    https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/
    reviews_Beauty_5.json.gz

Usage (from the repository root):
    python -m scripts.download_amazon_beauty
    python -m scripts.download_amazon_beauty --raw-dir data/raw/amazon_beauty
    python -m scripts.download_amazon_beauty --force

Output:
    data/raw/amazon_beauty/reviews_Beauty_5.json
    (one JSON review record per line)
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.error
import zlib
from pathlib import Path

from src.data.download import download_file, get_ssl_context

DATASET_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/"
    "categoryFiles/reviews_Beauty_5.json.gz"
)

ARCHIVE_FILENAME = "reviews_Beauty_5.json.gz"
DATASET_FILENAME = "reviews_Beauty_5.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and extract the Stanford SNAP Amazon "
            "Beauty 5-core dataset."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/amazon_beauty"),
        help="Directory where the Beauty 5-core dataset will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and replace the dataset if it already exists.",
    )

    return parser.parse_args()


def dataset_is_present(dataset_path: Path) -> bool:
    return dataset_path.is_file() and dataset_path.stat().st_size > 0


def extract_dataset(archive_path: Path, raw_dir: Path) -> Path:
    """Extract the gzip archive into a JSON file.

    Extraction goes through a temporary file, so an incomplete extraction
    cannot replace an existing valid dataset.
    """

    dataset_path = raw_dir / DATASET_FILENAME
    temporary_path = raw_dir / f"{DATASET_FILENAME}.tmp"

    print(f"Extracting {archive_path}...")

    try:
        with gzip.open(archive_path, "rb") as compressed:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(compressed, output, length=1024 * 1024)

        if not dataset_is_present(temporary_path):
            raise RuntimeError(
                "Extraction completed, but the expected "
                "Beauty 5-core JSON file is empty."
            )

        temporary_path.replace(dataset_path)

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return dataset_path


def main() -> int:
    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    dataset_path = raw_dir / DATASET_FILENAME
    archive_path = raw_dir / ARCHIVE_FILENAME

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stanford SNAP - Amazon Beauty 5-core Downloader")
    print("=" * 70)

    if dataset_is_present(dataset_path) and not args.force:
        print("\nBeauty 5-core dataset is already present:")
        print(f"  {dataset_path}")
        print("\nUse --force to download and replace it.")
        return 0

    context = get_ssl_context()

    try:
        download_file(DATASET_URL, archive_path, context, require_nonempty=True)
        dataset_path = extract_dataset(archive_path, raw_dir)

    except urllib.error.URLError as exc:
        print(
            "\nERROR: Could not download the Stanford SNAP "
            "Amazon Beauty dataset.",
            file=sys.stderr,
        )
        print(f"Reason: {exc}", file=sys.stderr)
        print("\nTry installing/updating certifi:", file=sys.stderr)
        print("  pip install --upgrade certifi", file=sys.stderr)
        return 1

    except (gzip.BadGzipFile, EOFError, zlib.error):
        print(
            "\nERROR: The downloaded file is not a valid "
            "or complete GZIP archive.",
            file=sys.stderr,
        )
        return 1

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    finally:
        if archive_path.exists():
            print(f"\nRemoving temporary archive: {archive_path}")
            archive_path.unlink()

    print("\n" + "=" * 70)
    print("Stanford SNAP - Amazon Beauty 5-core download complete.")
    print("=" * 70)

    print("\nDataset:")
    print(f"  {dataset_path}")

    print("\nFiles:")
    size_mb = dataset_path.stat().st_size / 1024**2
    print(f"  {DATASET_FILENAME:<28} {size_mb:8.2f} MB")

    print("\nNext step:")
    print("  python -m scripts.preprocess_amazon_beauty")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
