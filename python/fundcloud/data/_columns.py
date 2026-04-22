"""Canonical OHLCV column naming used across every backend.

Every network backend (``YF``, ``FMP``, ``AV``, ``Binance``) emits
columns in lowercase snake_case via :func:`normalize_field`, with the
canonical OHLCV fields ordered consistently per
:func:`canonicalize_ohlcv_order`. This means downstream code can rely
on ``"open"``, ``"high"``, ``"low"``, ``"close"``, ``"volume"`` (and
``"adj_close"`` when the provider supplies it) regardless of which
backend produced the frame.
"""

from __future__ import annotations

import re

import pandas as pd

__all__ = [
    "OHLCV_COLUMNS",
    "canonicalize_ohlcv_order",
    "normalize_field",
    "normalize_ohlcv_columns",
]


OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
"""Canonical order of the standard OHLCV fields."""


# Two-pass split handles both lowercase→uppercase ("adjClose") and acronym
# boundaries ("HTTPRequest" → "HTTP_Request") without shredding all-caps
# words like "VWAP" / "CLOSE".
_ACRONYM_RE = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_REPEAT_RE = re.compile(r"_+")


def normalize_field(name: str) -> str:
    """Coerce a column field name to lowercase snake_case.

    Examples
    --------
    >>> normalize_field("Open")
    'open'
    >>> normalize_field("CLOSE")
    'close'
    >>> normalize_field("AdjClose")
    'adj_close'
    >>> normalize_field("Adj Close")
    'adj_close'
    >>> normalize_field("VWAP")
    'vwap'
    >>> normalize_field("HTTPRequest")
    'http_request'
    """
    s = str(name).strip()
    s = _ACRONYM_RE.sub(r"\1_\2", s)
    s = _CAMEL_RE.sub(r"\1_\2", s)
    s = s.lower().replace(" ", "_").replace("-", "_")
    s = _REPEAT_RE.sub("_", s).strip("_")
    return s


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case every column field name.

    Handles both flat columns and the canonical ``(field, symbol)``
    MultiIndex layout. Returns the same frame (column relabel is in
    place); pass a copy in if the caller cares about identity.
    """
    if len(df.columns) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([
            (normalize_field(field), sym) for field, sym in df.columns
        ])
    else:
        df.columns = [normalize_field(c) for c in df.columns]
    return df


def canonicalize_ohlcv_order(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns so OHLCV fields appear in canonical order.

    Non-OHLCV fields are kept after the canonical ones, in the order they
    already appear. Works for both flat and ``(field, symbol)`` MultiIndex
    layouts. Returns the reordered frame.
    """
    if len(df.columns) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        symbols: list[str] = list(dict.fromkeys(sym for _, sym in df.columns))
        present_fields = list(dict.fromkeys(field for field, _ in df.columns))
        ordered_fields = [f for f in OHLCV_COLUMNS if f in present_fields]
        ordered_fields += [f for f in present_fields if f not in OHLCV_COLUMNS]
        new_cols = [
            (field, sym)
            for field in ordered_fields
            for sym in symbols
            if (field, sym) in df.columns
        ]
        return df.reindex(columns=pd.MultiIndex.from_tuples(new_cols))
    present = list(df.columns)
    ordered = [c for c in OHLCV_COLUMNS if c in present]
    ordered += [c for c in present if c not in OHLCV_COLUMNS]
    return df.reindex(columns=ordered)
