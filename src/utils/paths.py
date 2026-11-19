"""
Single source of truth for where every artifact lives.

Layout (``<dataset>`` is e.g. ``ml-1m`` or ``amazon_beauty``; it matches the
folder names under ``data/raw``, ``data/interim`` and ``data/processed``)::

    configs/datasets/<dataset>.yaml                   dataset facts
    results/<dataset>/<experiment>/                   history, summary, config, test metrics
    results/<dataset>/<search_name>/                  grid-search / seed-search reports
    checkpoints/<dataset>/<experiment>/<experiment>_best.pt

No other module should build these paths by hand; call the helpers below.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"

RESULTS_ROOT = Path("results")
CHECKPOINTS_ROOT = Path("checkpoints")

# Subfolder inside a search directory where throwaway trial artifacts go, so
# searches can never overwrite a real experiment's results or checkpoints.
SCRATCH_NAME = "_scratch"

DERIVED_OUTPUT_KEYS = ("checkpoint_dir", "results_dir", "best_checkpoint_name")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_name(value: str, what: str = "name") -> str:
    """Reject empty names and anything that could escape its folder."""

    if not isinstance(value, str) or not _SAFE_NAME.match(value):
        raise ValueError(
            f"Invalid {what} {value!r}: use letters, digits, '.', '_' or '-' "
            "only (no slashes or spaces)."
        )
    return value


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

def available_datasets() -> list[str]:
    """Dataset names that have a ``configs/datasets/<name>.yaml`` file."""

    if not DATASET_CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in DATASET_CONFIG_DIR.glob("*.yaml"))


def dataset_config_path(dataset: str) -> Path:
    """Path of a dataset's config file; raises a helpful error if missing."""

    validate_name(dataset, "dataset")
    path = DATASET_CONFIG_DIR / f"{dataset}.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"Unknown dataset {dataset!r}: {path} does not exist. "
            f"Available datasets: {available_datasets() or 'none'}. "
            "To add one, create that file (copy an existing one)."
        )
    return path


# ---------------------------------------------------------------------
# Artifact locations
# ---------------------------------------------------------------------

def results_dir(dataset: str, experiment: str) -> Path:
    return RESULTS_ROOT / validate_name(dataset, "dataset") / validate_name(
        experiment, "experiment name"
    )


def checkpoint_dir(dataset: str, experiment: str) -> Path:
    return CHECKPOINTS_ROOT / validate_name(dataset, "dataset") / validate_name(
        experiment, "experiment name"
    )


def best_checkpoint_name(experiment: str) -> str:
    return f"{validate_name(experiment, 'experiment name')}_best.pt"


def best_checkpoint_path(dataset: str, experiment: str) -> Path:
    return checkpoint_dir(dataset, experiment) / best_checkpoint_name(experiment)


def search_dir(dataset: str, search_name: str) -> Path:
    """Folder for aggregate reports, e.g. ``results/ml-1m/jepa_grid_search``."""

    return RESULTS_ROOT / validate_name(dataset, "dataset") / validate_name(
        search_name, "search name"
    )


def scratch_dir(parent: str | Path) -> Path:
    return Path(parent) / SCRATCH_NAME


# ---------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------

def resolve_output(config: dict, *, force: bool = False) -> dict:
    """
    Fill ``config["output"]`` from ``config["dataset"]`` and the experiment name.

    Only missing keys are filled (so explicit values, such as the scratch
    redirects used by searches, survive) unless ``force`` is true.
    Modifies and returns ``config``.
    """

    dataset = config.get("dataset")
    if not dataset:
        raise ValueError(
            "Config has no 'dataset'. Set `dataset: <name>` in the YAML or "
            "pass --dataset."
        )

    output = config.setdefault("output", {})
    experiment = output.get("experiment_name")
    if not experiment:
        raise ValueError("Config has no output.experiment_name.")

    derived = {
        "checkpoint_dir": str(checkpoint_dir(dataset, experiment).as_posix()),
        "results_dir": str(results_dir(dataset, experiment).as_posix()),
        "best_checkpoint_name": best_checkpoint_name(experiment),
    }

    for key, value in derived.items():
        if force or key not in output:
            output[key] = value

    return config
