"""YAML config loading with ``key.sub=value`` command-line overrides."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply overrides such as ``ppo.learning_rate=1e-3``."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
