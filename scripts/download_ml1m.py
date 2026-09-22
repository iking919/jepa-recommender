#!/usr/bin/env python3
"""
Download and extract the MovieLens-1M dataset.

Usage:
    python scripts/download_ml1m.py

Optional:
    python scripts/download_ml1m.py --raw-dir data/raw
    python scripts/download_ml1m.py --force
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


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


def dataset_is_present(dataset_dir: Path) -> bool:
    return all(
        (dataset_dir / filename).is_file()
        for filename in EXPECTED_FILES
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
    zip_path: Path,
    raw_dir: Path,
) -> Path:

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

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        download_file(
            DATASET_URL,
            zip_path,
            context,
        )

        dataset_dir = extract_dataset(
            zip_path,
            raw_dir,
        )

    except urllib.error.URLError as exc:

        print(
            "\nERROR: Could not download MovieLens-1M.",
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

    except zipfile.BadZipFile:

        print(
            "\nERROR: The downloaded file is not a valid ZIP archive.",
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

        if zip_path.exists():
            print(
                f"Removing temporary archive: {zip_path}"
            )
            zip_path.unlink()

    print("\n" + "=" * 70)
    print("MovieLens-1M download complete.")
    print("=" * 70)

    print("\nDataset:")
    print(f"  {dataset_dir}")

    print("\nFiles:")

    for filename in EXPECTED_FILES:

        file_path = dataset_dir / filename
        size_mb = file_path.stat().st_size / 1024**2

        print(
            f"  ✓ {filename:<12} "
            f"{size_mb:8.2f} MB"
        )

    print("\nNext step:")
    print("  python scripts/preprocess_ml1m.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())