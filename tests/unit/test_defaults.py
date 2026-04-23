"""Tests for the network-backend default-window helper."""

from __future__ import annotations

import pandas as pd
from fundcloud.data._defaults import default_start_one_year_back


def test_explicit_start_passes_through() -> None:
    out = default_start_one_year_back("2024-01-01", "2024-12-31")
    assert out == "2024-01-01"


def test_no_start_no_end_defaults_to_today_minus_1y() -> None:
    out = default_start_one_year_back(None, None)
    assert isinstance(out, pd.Timestamp)
    today = pd.Timestamp.now().normalize()
    expected = today - pd.DateOffset(years=1)
    assert abs((out - expected).days) <= 1  # tolerate clock-edge runs


def test_no_start_with_end_uses_end_minus_1y() -> None:
    out = default_start_one_year_back(None, "2025-06-15")
    assert out == pd.Timestamp("2024-06-15")


def test_no_start_with_timestamp_end() -> None:
    end = pd.Timestamp("2025-06-15")
    out = default_start_one_year_back(None, end)
    assert out == pd.Timestamp("2024-06-15")
