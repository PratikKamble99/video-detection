"""Config loader. Thin wrapper over YAML with attribute access.

Every model named in the config sits behind an interface, so swapping one is a
config change, not a rewrite (ARCHITECTURE.md §14). Keep this dumb: parse YAML,
expose it, done. Validation lives with the components that consume each section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config(dict):
    """dict with dotted attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(val) if isinstance(val, dict) else val


def load(path: str | Path | None = None) -> Config:
    """Load a config file (defaults to ``configs/default.yaml``)."""
    p = Path(path) if path else _DEFAULT
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f) or {})
