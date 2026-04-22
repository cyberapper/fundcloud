"""Bundled example datasets — loader shell.

Sample data files live under ``_data/`` and are loaded with
:func:`load_example_panel`. For curated datasets pulled from live
providers, see :mod:`fundcloud.data` (``YF``, ``FMP``, ``AV``, ``Binance``).
"""

from __future__ import annotations

import importlib.resources
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from importlib.abc import Traversable

__all__ = ["available_datasets", "load_example_panel"]

# Store the Traversable reference — safe to keep as a module-level variable
# because Traversable does not open a file handle (unlike as_file()).
_DATA_REF: Traversable = importlib.resources.files("fundcloud.datasets") / "_data"


def available_datasets() -> list[str]:
    """Names of datasets (file stems) bundled with the package."""
    try:
        entries = list(_DATA_REF.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    return sorted(e.name[: -len(".parquet")] for e in entries if e.name.endswith(".parquet"))


def load_example_panel(name: str = "toy_equities_5y") -> pd.DataFrame:
    """Load a bundled example panel keyed by stem.

    Raises
    ------
    FileNotFoundError
        If no such dataset ships with this version.
    """
    ref = _DATA_REF / f"{name}.parquet"
    try:
        with importlib.resources.as_file(ref) as path:
            return pd.read_parquet(path)
    except (FileNotFoundError, TypeError):
        available = available_datasets()
        msg = f"dataset {name!r} not found. Available: {available}"
        raise FileNotFoundError(msg) from None
