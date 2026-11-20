"""HTTP download helpers shared by the dataset download scripts."""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

USER_AGENT = "jepa-recommender/1.0"
CHUNK_SIZE = 1024 * 1024


def get_ssl_context() -> ssl.SSLContext:
    """Create an SSL context, using certifi's CA bundle when available.

    Certificate verification is never disabled.
    """

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())

    except ImportError:
        print(
            "WARNING: certifi is not installed; using the system "
            "certificate store."
        )
        return ssl.create_default_context()


def download_file(
    url: str,
    destination: Path,
    context: ssl.SSLContext,
    *,
    require_nonempty: bool = False,
) -> None:
    """Download ``url`` to ``destination`` with a progress display.

    A partial download is deleted if anything goes wrong.

    Args:
        url: Address to download.
        destination: File to write.
        context: SSL context used for the request.
        require_nonempty: Raise if the downloaded file has zero bytes.
    """

    print(f"Downloading:\n  {url}")
    print(f"Destination:\n  {destination}\n")

    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        with urllib.request.urlopen(request, context=context) as response:
            content_length = response.headers.get("Content-Length")
            total_bytes = int(content_length) if content_length else None
            downloaded = 0

            with destination.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)

                    if not chunk:
                        break

                    output.write(chunk)
                    downloaded += len(chunk)
                    _print_progress(downloaded, total_bytes)

        print("\nDownload complete.")

        if require_nonempty and destination.stat().st_size == 0:
            raise RuntimeError("Downloaded archive is empty.")

    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def _print_progress(downloaded: int, total_bytes: int | None) -> None:
    downloaded_mb = downloaded / 1024**2

    if total_bytes:
        percent = downloaded / total_bytes * 100
        print(
            f"\rProgress: {percent:6.2f}% "
            f"({downloaded_mb:.1f} MB / {total_bytes / 1024**2:.1f} MB)",
            end="",
            flush=True,
        )
    else:
        print(f"\rDownloaded: {downloaded_mb:.1f} MB", end="", flush=True)
