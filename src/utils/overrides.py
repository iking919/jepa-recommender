"""
Command-line overrides for YAML experiment configs.

    --set training.learning_rate=0.001 --set seed=7
    --name my_run

Values are parsed as YAML, so ``0.001`` is a float and ``true`` a bool.
Unknown keys are rejected (a typo would otherwise be silently ignored);
prefix a key with ``+`` to add a new one on purpose.
"""

from __future__ import annotations

import copy

import yaml


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
    """Point every output path at a fresh experiment called ``name``."""

    result = copy.deepcopy(config)

    result["output"].update(
        {
            "experiment_name": name,
            "checkpoint_dir": f"checkpoints/{name}",
            "results_dir": f"results/{name}",
            "best_checkpoint_name": f"{name}_best.pt",
        }
    )

    return result
