"""Register the ``.fc`` pandas accessor on import.

Importing ``fundcloud`` triggers import of this module as a side effect; the
accessor registrations happen in :mod:`fundcloud.accessors.series` and
:mod:`fundcloud.accessors.dataframe` via decorators.
"""

from __future__ import annotations

from fundcloud.accessors.dataframe import DataFrameAccessor
from fundcloud.accessors.series import SeriesAccessor

__all__ = ["DataFrameAccessor", "SeriesAccessor"]
