"""Local dataset utilities for gsplat examples.

This package shadows the external `datasets` (HuggingFace) module so that
imports like `from datasets.colmap import Dataset` resolve to the in-repo
implementations.
"""

from importlib import import_module as _import_module
from pathlib import Path as _Path
import sys as _sys

# Ensure the package directory itself is on sys.path (needed when this module
# is imported while another `datasets` pip package is already in sys.modules).
_PKG_DIR = _Path(__file__).resolve().parent
_PARENT = _PKG_DIR.parent
if str(_PKG_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PKG_DIR))
if str(_PARENT) not in _sys.path:
    _sys.path.insert(0, str(_PARENT))

# If another `datasets` implementation was imported earlier (e.g., HuggingFace),
# remove it so that our modules load correctly.
if "datasets" in _sys.modules and _sys.modules["datasets"] is not _sys.modules[__name__]:
    _sys.modules.pop("datasets", None)


def __getattr__(name: str):
    """Support attribute-style access to child modules (lazy import)."""
    try:
        module = _import_module(f"datasets.{name}")
    except ModuleNotFoundError as exc:
        raise AttributeError(name) from exc
    return module


__all__ = []
