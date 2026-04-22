"""Bundled example datasets — loader shell.

Sample data files live under ``_data/`` and are loaded with
:func:`load_example_panel`. For curated datasets pulled from live
providers, see :mod:`fundcloud.data` (``YF``, ``FMP``, ``AV``, ``Binance``).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pandas as pd

__all__ = ["DATASET_DIR", "available_datasets", "load_example_panel"]


def _dataset_dir() -> Path:
    with resources.as_file(resources.files("fundcloud.datasets") / "_data") as p:
        return p


DATASET_DIR: Path = _dataset_dir()


def available_datasets() -> list[str]:
    """Names of datasets (file stems) bundled with the package."""
    if not DATASET_DIR.exists():
        return []
    return sorted(p.stem for p in DATASET_DIR.glob("*.parquet"))


def load_example_panel(name: str = "toy_equities_5y") -> pd.DataFrame:
    """Load a bundled example panel keyed by stem.

    Raises
    ------
    FileNotFoundError
        If no such dataset ships with this version.
    """
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        available = available_datasets()
        msg = f"dataset {name!r} not found. Available: {available}"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)
