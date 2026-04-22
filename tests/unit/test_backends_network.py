"""Tests for network-backed backends (mocked HTTP)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ----------------------------------------------------------------------- YF


def _fake_yf_download() -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5, freq="D").values)
    df = pd.DataFrame(
        {
            "Open": np.arange(5, dtype=float),
            "High": np.arange(5, dtype=float) + 1,
            "Low": np.arange(5, dtype=float) - 1,
            "Close": np.arange(5, dtype=float),
            "Volume": np.full(5, 1000, dtype=float),
        },
        index=idx,
    )
    return df


def test_yf_single_symbol_returns_multiindex() -> None:
    from fundcloud.data.yf import YF

    fake = _fake_yf_download()
    with patch("yfinance.download", return_value=fake):
        src = YF("AAPL")
        out = src.read(start="2024-01-01", end="2024-01-05")

    assert isinstance(out.columns, pd.MultiIndex)
    fields = [c[0] for c in out.columns]
    # Canonical OHLCV: lowercase + standard ordering.
    assert fields == ["open", "high", "low", "close", "volume"]
    assert out.columns.get_level_values(1).unique().tolist() == ["AAPL"]


def test_yf_multi_word_field_normalised_to_snake_case() -> None:
    """yfinance can return 'Adj Close' when auto_adjust=False — normalise it."""
    from fundcloud.data.yf import YF

    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=2))
    fake = pd.DataFrame(
        {"Open": [1.0, 2], "Adj Close": [1.4, 2.4], "Close": [1.5, 2.5]},
        index=idx,
    )
    with patch("yfinance.download", return_value=fake):
        out = YF("AAPL", adjust=False).read(start="2024-01-01")
    fields = [c[0] for c in out.columns]
    assert "adj_close" in fields
    # Canonical OHLCV fields come first; extras follow.
    assert fields[:2] == ["open", "close"]


def test_yf_is_read_only() -> None:
    from fundcloud.data import ReadOnlyError
    from fundcloud.data.yf import YF

    src = YF("AAPL")
    assert src.read_only is True
    with pytest.raises(ReadOnlyError):
        src.write("AAPL", pd.DataFrame())


# ----------------------------------------------------------------------- FMP


def _fake_fmp_payload() -> dict:
    return {
        "historical": [
            {
                "date": "2024-01-01",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 1000,
            },
            {
                "date": "2024-01-02",
                "open": 1.5,
                "high": 2.5,
                "low": 1.0,
                "close": 2.0,
                "volume": 2000,
            },
        ]
    }


def test_fmp_parses_historical_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fmp as fmp_mod

    def fake_get_json(self, url, params=None):
        return _fake_fmp_payload()

    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", fake_get_json)

    src = fmp_mod.FMP("AAPL", api_key="test")
    out = src.read(start="2024-01-01", end="2024-01-02")
    assert isinstance(out.columns, pd.MultiIndex)
    fields = [c[0] for c in out.columns]
    assert fields == ["open", "high", "low", "close", "volume"]
    assert ("close", "AAPL") in out.columns
    assert len(out) == 2


def test_fmp_normalises_camelcase_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """FMP can ship CamelCase keys like 'adjClose'; normalisation must catch them."""
    from fundcloud.data import fmp as fmp_mod

    payload = {
        "historical": [
            {"date": "2024-01-01", "Open": 1.0, "High": 2.0, "Low": 0.5,
             "Close": 1.5, "Volume": 1000, "adjClose": 1.4},
        ]
    }
    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", lambda *_a, **_k: payload)
    out = fmp_mod.FMP("AAPL", api_key="test").read(start="2024-01-01", end="2024-01-02")
    fields = {c[0] for c in out.columns}
    # Canonical OHLCV survives even when provider sent CamelCase.
    assert {"open", "high", "low", "close", "volume"}.issubset(fields)


def test_fmp_adjust_true_promotes_adj_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """With adjust=True (default), the adjClose value should land in `close`."""
    from fundcloud.data import fmp as fmp_mod

    payload = {
        "historical": [
            {"date": "2024-01-02", "open": 1.5, "high": 2.5, "low": 1.0,
             "close": 2.0, "adjClose": 1.95, "volume": 2000},
            {"date": "2024-01-01", "open": 1.0, "high": 2.0, "low": 0.5,
             "close": 1.5, "adjClose": 1.45, "volume": 1000},
        ]
    }
    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", lambda *_a, **_k: payload)
    out = fmp_mod.FMP("AAPL", api_key="test").read(start="2024-01-01", end="2024-01-02")
    # Adjusted value, not the raw 2.0 close.
    assert out[("close", "AAPL")].iloc[-1] == 1.95


def test_fmp_adjust_false_keeps_raw_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fmp as fmp_mod

    payload = {
        "historical": [
            {"date": "2024-01-02", "open": 1.5, "high": 2.5, "low": 1.0,
             "close": 2.0, "adjClose": 1.95, "volume": 2000},
        ]
    }
    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", lambda *_a, **_k: payload)
    out = fmp_mod.FMP("AAPL", adjust=False, api_key="test").read(
        start="2024-01-01", end="2024-01-02"
    )
    assert out[("close", "AAPL")].iloc[-1] == 2.0


def test_fmp_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.fmp import FMP

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        FMP("AAPL")


# ----------------------------------------------------------------------- AV


def _fake_av_payload() -> dict:
    """Adjusted-endpoint payload — both raw `4. close` and `5. adjusted close`."""
    return {
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "1.5",
                "2. high": "2.5",
                "3. low": "1.0",
                "4. close": "2.0",
                "5. adjusted close": "1.95",
                "6. volume": "2000",
            },
            "2024-01-01": {
                "1. open": "1.0",
                "2. high": "2.0",
                "3. low": "0.5",
                "4. close": "1.5",
                "5. adjusted close": "1.45",
                "6. volume": "1000",
            },
        }
    }


def _fake_av_raw_payload() -> dict:
    """Unadjusted-endpoint payload — only `4. close` and `5. volume`."""
    return {
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "1.5", "2. high": "2.5", "3. low": "1.0",
                "4. close": "2.0", "5. volume": "2000",
            },
            "2024-01-01": {
                "1. open": "1.0", "2. high": "2.0", "3. low": "0.5",
                "4. close": "1.5", "5. volume": "1000",
            },
        }
    }


def test_av_parses_daily_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import av as av_mod

    monkeypatch.setattr(
        av_mod.HttpClient, "get_json",
        lambda self, url, params=None: _fake_av_payload(),
    )
    src = av_mod.AV("AAPL", api_key="test")
    # Override 1y default so the 2024 fixture rows survive the slice.
    out = src.read(start="2024-01-01", end="2024-01-02")
    assert isinstance(out.columns, pd.MultiIndex)
    fields = [c[0] for c in out.columns]
    assert fields == ["open", "high", "low", "close", "volume"]
    assert ("close", "AAPL") in out.columns
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    # adjust=True (default): close is the *adjusted* close, not the raw one.
    assert out[("close", "AAPL")].iloc[-1] == 1.95


def test_av_adjust_false_uses_raw_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import av as av_mod

    captured: dict[str, str] = {}

    def fake_get_json(self, url, params=None):
        # Verify the unadjusted endpoint name is requested.
        captured["function"] = (params or {}).get("function", "")
        return _fake_av_raw_payload()

    monkeypatch.setattr(av_mod.HttpClient, "get_json", fake_get_json)
    src = av_mod.AV("AAPL", adjust=False, api_key="test")
    out = src.read(start="2024-01-01", end="2024-01-02")

    assert captured["function"] == "TIME_SERIES_DAILY"
    # Raw `4. close` is what landed in the canonical close column.
    assert out[("close", "AAPL")].iloc[-1] == 2.0


def test_av_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.av import AV

    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AV("AAPL")


# ----------------------------------------------------------------------- Binance


def test_binance_pages_through_ohlcv(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import binance as bn_mod

    pages = [
        [
            [
                int((pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).timestamp() * 1000),
                1.0,
                2.0,
                0.5,
                1.5,
                100.0,
            ]
            for i in range(10)
        ],
        [
            [
                int((pd.Timestamp("2024-01-11") + pd.Timedelta(days=i)).timestamp() * 1000),
                1.0,
                2.0,
                0.5,
                1.5,
                100.0,
            ]
            for i in range(3)
        ],
    ]
    fake_exchange = MagicMock()
    fake_exchange.fetch_ohlcv.side_effect = pages

    fake_ccxt = SimpleNamespace(binance=lambda *_a, **_kw: fake_exchange)
    monkeypatch.setattr(bn_mod, "_require_ccxt", lambda: fake_ccxt)

    src = bn_mod.Binance("BTC/USDT", limit=10)
    out = src.read(start="2024-01-01")
    assert len(out) == 13
    fields = [c[0] for c in out.columns]
    assert fields == ["open", "high", "low", "close", "volume"]
    assert ("close", "BTC/USDT") in out.columns


# ----------------------------------------------------------------------- lazy loading


def test_network_backend_lazy_resolution() -> None:
    from fundcloud import data as fc_data

    # Attribute access without importing the submodule should still work.
    assert fc_data.YF.__name__ == "YF"
    assert fc_data.FMP.__name__ == "FMP"
    assert fc_data.AV.__name__ == "AV"
    assert fc_data.Binance.__name__ == "Binance"
