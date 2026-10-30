"""
Download and extract the Amazon Reviews 2023 All Beauty dataset.

Usage:
    python scripts/download_amazon_beauty.py

Optional:
    python scripts/download_amazon_beauty.py --raw-dir data/raw/amazon_beauty
    python scripts/download_amazon_beauty.py --force
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/"
    "data/amazon_2023/raw/review_categories/"
    "All_Beauty.jsonl.gz"
)

ARCHIVE_FILENAME = "All_Beauty.jsonl.gz"
DATASET_FILENAME = "All_Beauty.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract Amazon Reviews 2023 All Beauty."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/amazon_beauty"),
        help="Directory where the All Beauty dataset will be stored.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the dataset already exists.",
    )

    return parser.parse_args()


def get_ssl_context() -> ssl.SSLContext:
    """
    Create an SSL context using certifi when available.

    This avoids disabling certificate verification.
    """

    try:
        import certifi

        return ssl.create_default_context(
            cafile=certifi.where()
        )

    except ImportError:
        print(
            "WARNING: certifi is not installed; using the system "
            "certificate store."
        )
        return ssl.create_default_context()


def dataset_is_present(dataset_path: Path) -> bool:
    return (
        dataset_path.is_file()
        and dataset_path.stat().st_size > 0
    )


def download_file(
    url: str,
    destination: Path,
    context: ssl.SSLContext,
) -> None:

    print(f"Downloading:\n  {url}")
    print(f"Destination:\n  {destination}\n")

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "jepa-recommender/1.0"
            },
        )

        with urllib.request.urlopen(
            request,
            context=context,
        ) as response:

            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total else None

            downloaded = 0
            chunk_size = 1024 * 1024

            with destination.open("wb") as output:

                while True:
                    chunk = response.read(chunk_size)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes:
                        percent = (
                            downloaded / total_bytes * 100
                        )

                        print(
                            f"\rProgress: {percent:6.2f}% "
                            f"({downloaded / 1024**2:.1f} MB / "
                            f"{total_bytes / 1024**2:.1f} MB)",
                            end="",
                            flush=True,
                        )

                    else:
                        print(
                            f"\rDownloaded: "
                            f"{downloaded / 1024**2:.1f} MB",
                            end="",
                            flush=True,
                        )

        print("\nDownload complete.")

    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def extract_dataset(
    archive_path: Path,
    raw_dir: Path,
) -> Path:

    dataset_path = raw_dir / DATASET_FILENAME

    print(f"Extracting {archive_path}...")

    with gzip.open(archive_path, "rb") as compressed:
        with dataset_path.open("wb") as output:
            shutil.copyfileobj(
                compressed,
                output,
                length=1024 * 1024,
            )

    if not dataset_is_present(dataset_path):
        raise RuntimeError(
            "Extraction completed, but the expected "
            "All Beauty JSONL file was not found or is empty."
        )

    return dataset_path


def main() -> int:

    args = parse_args()

    raw_dir = args.raw_dir.resolve()
    dataset_path = raw_dir / DATASET_FILENAME
    archive_path = raw_dir / ARCHIVE_FILENAME

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Amazon Reviews 2023 - All Beauty Downloader")
    print("=" * 70)

    if dataset_is_present(dataset_path) and not args.force:

        print("\nAll Beauty dataset is already present:")
        print(f"  {dataset_path}")

        return 0

    if args.force and dataset_path.exists():

        print("\nRemoving existing dataset...")
        dataset_path.unlink()

    context = get_ssl_context()

    try:

        download_file(
            DATASET_URL,
            archive_path,
            context,
        )

        dataset_path = extract_dataset(
            archive_path,
            raw_dir,
        )

    except urllib.error.URLError as exc:

        print(
            "\nERROR: Could not download Amazon Reviews 2023.",
            file=sys.stderr,
        )

        print(
            f"Reason: {exc}",
            file=sys.stderr,
        )

        print(
            "\nTry installing/updating certifi:",
            file=sys.stderr,
        )

        print(
            "  pip install --upgrade certifi",
            file=sys.stderr,
        )

        return 1

    except (gzip.BadGzipFile, EOFError):

        print(
            "\nERROR: The downloaded file is not a valid "
            "or complete GZIP archive.",
            file=sys.stderr,
        )

        return 1

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        return 1

    finally:

        if archive_path.exists():
            print(
                f"Removing temporary archive: {archive_path}"
            )
            archive_path.unlink()

    print("\n" + "=" * 70)
    print("Amazon Reviews 2023 - All Beauty download complete.")
    print("=" * 70)

    print("\nDataset:")
    print(f"  {dataset_path}")

    print("\nFiles:")

    size_mb = dataset_path.stat().st_size / 1024**2

    print(
        f"  ✓ {DATASET_FILENAME:<24} "
        f"{size_mb:8.2f} MB"
    )

    print("\nNext step:")
    print("  python scripts/preprocess_amazon_beauty.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())