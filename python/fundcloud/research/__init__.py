"""Research utilities for the event-study engine.

Step 1 ships the data layer only: leak-free extraction of clean daily OHLCV from the
ClickHouse ``md_ohlcv_data`` feed (:func:`load_bars`, :func:`load_universe`) and a
quality gate + cleaner (:func:`check_quality`, :func:`clean_panel`) to run before any
research touches the data.
"""

from __future__ import annotations

from fundcloud.research.loader import load_bars, load_universe
from fundcloud.research.quality import QualityReport, check_quality, clean_panel

__all__ = [
    "QualityReport",
    "check_quality",
    "clean_panel",
    "load_bars",
    "load_universe",
]
