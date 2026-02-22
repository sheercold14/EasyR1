from __future__ import annotations

import importlib
from typing import Any


def deep_get(obj: Any, path: str, default: Any = None) -> Any:
    """Get nested value by dotted path, supports list indexes like a.b.0.c."""
    if not path:
        return obj
    cur = obj
    for key in path.split("."):
        if isinstance(cur, dict):
            if key not in cur:
                return default
            cur = cur[key]
            continue
        if isinstance(cur, list):
            if not key.isdigit():
                return default
            idx = int(key)
            if idx < 0 or idx >= len(cur):
                return default
            cur = cur[idx]
            continue
        return default
    return cur


def load_symbol(spec: str) -> Any:
    """Load symbol from 'module.sub:attr'."""
    if ":" not in spec:
        raise ValueError(f"Invalid spec '{spec}', expected module:attr")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise AttributeError(f"Module '{module_name}' has no attr '{attr}'")
    return getattr(module, attr)
