"""
Command-line overrides for YAML experiment configs.

    --set training.learning_rate=0.001 --set seed=7
    --name my_run
    --dataset amazon_beauty

Values are parsed as YAML, so ``0.001`` is a float and ``true`` a bool.
Unknown keys are rejected (a typo would otherwise be silently ignored);
prefix a key with ``+`` to add a new one on purpose.
"""

from __future__ import annotations

import copy

import yaml

from src.utils.paths import DERIVED_OUTPUT_KEYS, resolve_output, validate_name


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Return a copy of ``config`` with ``KEY=VALUE`` overrides applied."""

    result = copy.deepcopy(config)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like KEY=VALUE, got {item!r}.")

        key, raw_value = item.split("=", 1)
        allow_new = key.startswith("+")
        key = key.lstrip("+")
        parts = key.split(".")

        node = result
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise ValueError(f"Unknown config section {part!r} in {key!r}.")
            node = node[part]

        if parts[-1] not in node and not allow_new:
            raise ValueError(
                f"Unknown config key {key!r}. Use '+{key}=...' to add a new key."
            )

        node[parts[-1]] = yaml.safe_load(raw_value)

    return result


def apply_name(config: dict, name: str) -> dict:
    """
    Rename the experiment; every output path is re-derived from it.

    Paths follow ``results/<dataset>/<name>`` and
    ``checkpoints/<dataset>/<name>`` (see ``src.utils.paths``).
    """

    result = copy.deepcopy(config)
    output = result.setdefault("output", {})

    output["experiment_name"] = validate_name(name, "experiment name")

    for key in DERIVED_OUTPUT_KEYS:
        output.pop(key, None)

    # Re-derive now if the dataset is already known; otherwise load_config
    # will do it once the dataset has been resolved.
    if result.get("dataset"):
        resolve_output(result)

    return result
