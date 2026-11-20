"""JSON helpers shared by the search and cross-seed reporting scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def finite_number(value: Any) -> bool:
    """Return True if ``value`` can be read as a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def json_safe(value: Any) -> Any:
    """Convert common Python/NumPy values to strictly valid JSON values.

    Non-finite floats become ``None`` and unknown objects become strings.
    """

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

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
    """Write ``data`` as JSON atomically.

    The file is written to a temporary sibling first so an interrupted run
    cannot leave a half-written report behind.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, allow_nan=False)
        file.write("\n")

    temporary_path.replace(path)
