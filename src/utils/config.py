"""
Configuration loading.

A run's config is built in layers, later layers winning:

    1. the experiment YAML           (configs/<dataset>/<experiment>.yaml)
    2. the dataset YAML              (configs/datasets/<dataset>.yaml)
    3. command-line ``--set`` values
    4. ``--name`` (renames the experiment)

Then ``{dataset}`` / ``{experiment}`` placeholders are expanded and the
``output`` section is derived from the dataset and experiment name, giving
``results/<dataset>/<experiment>`` and ``checkpoints/<dataset>/<experiment>``.

The dataset is, in order of precedence: ``--dataset`` on the command line, a
``dataset:`` key in the YAML, or the name of the folder the YAML sits in
(``configs/ml-1m/x.yaml`` -> ``ml-1m``). ``--dataset`` lets one experiment YAML
run on any dataset; keep a separate YAML under ``configs/<dataset>/`` when a
dataset needs its own hyperparameters.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import yaml

from src.utils.overrides import apply_name, apply_overrides
from src.utils.paths import (
    available_datasets,
    dataset_config_path,
    resolve_output,
    validate_name,
)

logger = logging.getLogger(__name__)

# Keys in a dataset file that describe it rather than configure a run.
_DATASET_META_KEYS = ("name", "description")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")

    return data


def infer_dataset_from_path(config_path: Path) -> str | None:
    """Dataset named by the config's parent folder, if it is a known dataset."""

    folder = config_path.resolve().parent.name
    return folder if folder in available_datasets() else None


def deep_merge(base: dict, override: dict, *, _path: str = "") -> dict:
    """Return ``base`` updated with ``override`` (recursively for mappings).

    Logs a warning when ``override`` replaces a different existing value.
    """

    result = copy.deepcopy(base)

    for key, value in override.items():
        here = f"{_path}{key}"
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value, _path=f"{here}.")
        else:
            if key in result and result[key] != value:
                logger.warning(
                    "%s = %r in the experiment config is overridden by the "
                    "dataset config (%r).", here, result[key], value,
                )
            result[key] = copy.deepcopy(value)

    return result


def _expand_tokens(node, tokens: dict[str, str]):
    """Replace ``{dataset}`` / ``{experiment}`` in every string value."""

    if isinstance(node, dict):
        return {k: _expand_tokens(v, tokens) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tokens(v, tokens) for v in node]
    if isinstance(node, str):
        for token, value in tokens.items():
            node = node.replace("{" + token + "}", value)
    return node


def _check_belongs_to_dataset(config: dict, sections: tuple[str, ...]) -> None:
    """Refuse paths that point at a different dataset's folder.

    This is what stops a run configured for ``amazon_beauty`` from silently
    reading ``data/processed/ml-1m`` (or writing into its results).
    """

    dataset = config["dataset"]

    for section in sections:
        for key, value in (config.get(section) or {}).items():
            if not isinstance(value, str) or "/" not in value.replace("\\", "/"):
                continue
            parts = Path(value.replace("\\", "/")).parts
            if dataset not in parts:
                raise ValueError(
                    f"{section}.{key} = {value!r} does not contain the "
                    f"dataset folder {dataset!r}. Use the {{dataset}} "
                    f"placeholder, e.g. 'data/processed/{{dataset}}/...'."
                )


def load_config(
    config_path: str | Path,
    *,
    dataset: str | None = None,
    overrides: list[str] | None = None,
    name: str | None = None,
) -> dict:
    """Build the fully resolved config for one run (see module docstring)."""

    config_path = Path(config_path)
    raw = _read_yaml(config_path)

    from_folder = infer_dataset_from_path(config_path)
    declared = raw.get("dataset")
    if declared and from_folder and declared != from_folder and not dataset:
        raise ValueError(
            f"{config_path} sits in configs/{from_folder}/ but declares "
            f"dataset: {declared!r}. Remove the `dataset:` key (the folder "
            "decides) or move the file."
        )

    dataset = dataset or declared or from_folder
    if not dataset:
        raise ValueError(
            f"No dataset chosen for {config_path}. Put the YAML in "
            f"configs/<dataset>/, add `dataset: <name>` to it, or pass "
            f"--dataset (available: {available_datasets()})."
        )
    validate_name(dataset, "dataset")

    # Layer 2: dataset facts (vocab size, sequence length, ...).
    dataset_cfg = _read_yaml(dataset_config_path(dataset))
    dataset_name = dataset_cfg.get("name", dataset)
    if dataset_name != dataset:
        raise ValueError(
            f"configs/datasets/{dataset}.yaml declares name={dataset_name!r}; "
            "the file name and `name` must match."
        )
    for key in _DATASET_META_KEYS:
        dataset_cfg.pop(key, None)

    raw.setdefault("output", {})
    raw["output"].setdefault("experiment_name", config_path.stem)

    config = deep_merge(raw, dataset_cfg)
    config["dataset"] = dataset

    # Layer 3 and 4: command line.
    if overrides:
        config = apply_overrides(config, overrides)
        if config.get("dataset") != dataset:
            raise ValueError(
                "Do not change the dataset with --set; use --dataset so the "
                "dataset config is applied."
            )

    if name:
        config = apply_name(config, name)

    config = _expand_tokens(
        config,
        {"dataset": dataset, "experiment": config["output"]["experiment_name"]},
    )

    # Explicit paths written in the YAML must belong to this dataset.
    _check_belongs_to_dataset(config, ("data",))
    _check_belongs_to_dataset(config, ("output",))

    return resolve_output(config)


# ---------------------------------------------------------------------
# Shared command-line interface
# ---------------------------------------------------------------------

def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard --config/--dataset/--set/--name options."""

    parser.add_argument(
        "--config", required=True,
        help="Path to the YAML experiment configuration.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset to run on (a file in configs/datasets/). Overrides "
             "the `dataset:` key in the config.",
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.learning_rate=0.001. "
             "Repeatable. Use +KEY=VALUE to add a new key.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Experiment name; outputs go to results/<dataset>/<name> and "
             "checkpoints/<dataset>/<name>.",
    )


def load_config_from_args(args: argparse.Namespace) -> dict:
    """Build the config from arguments added by ``add_config_args``."""

    return load_config(
        args.config,
        dataset=args.dataset,
        overrides=args.overrides,
        name=args.name,
    )


def save_resolved_config(config: dict, results_dir: str | Path) -> Path:
    """Write the fully resolved config next to a run's results."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "config.yaml"

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return path
