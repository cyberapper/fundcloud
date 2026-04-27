"""Tests for the FundCloud market-data backend."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def _fake_ohlcv_payload(symbol: str = "AAPL") -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": "1D",
        "period": "2Y",
        "category": "stock-us",
        "candles": [
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "open": 186.3,
                "high": 187.8,
                "low": 185.8,
                "close": 185.6,
                "volume": 72_000_000,
            },
            {
                "timestamp": "2024-01-03T00:00:00Z",
                "open": 185.5,
                "high": 186.0,
                "low": 183.0,
                "close": 184.2,
                "volume": 68_000_000,
            },
            {
                "timestamp": "2024-01-04T00:00:00Z",
                "open": 184.0,
                "high": 184.5,
                "low": 181.5,
                "close": 182.0,
                "volume": 75_000_000,
            },
        ],
    }


def _fake_payload_for(
    symbol: str, candles_override: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload = _fake_ohlcv_payload(symbol)
    if candles_override is not None:
        payload["candles"] = candles_override
    return payload


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a dummy API key unless it explicitly deletes it."""
    monkeypatch.setenv("FUNDCLOUD_API_KEY", "fc_test_unit")


def test_single_symbol_returns_multiindex(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        captured["path"] = path
        captured["params"] = dict(params or {})
        return _fake_ohlcv_payload("AAPL")

    monkeypatch.setattr(fc_mod.FundCloudClient, "get", fake_get)

    src = fc_mod.FundCloud("AAPL")
    out = src.read()

    assert isinstance(out.columns, pd.MultiIndex)
    fields = [c[0] for c in out.columns]
    assert fields == ["open", "high", "low", "close", "volume"]
    assert out.columns.get_level_values(1).unique().tolist() == ["AAPL"]
    assert len(out) == 3
    # Index is tz-naive.
    assert out.index.tz is None
    # URL + params
    assert captured["path"] == "/market/AAPL/ohlcv"
    assert captured["params"]["timeframe"] == "1D"
    assert captured["params"]["period"] == "2Y"
    assert captured["params"]["adjusted"] == "true"


def test_multi_symbol_loops_and_concats(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    paths: list[str] = []

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        paths.append(path)
        sym = path.split("/")[2]
        return _fake_payload_for(sym)

    monkeypatch.setattr(fc_mod.FundCloudClient, "get", fake_get)

    out = fc_mod.FundCloud(["AAPL", "MSFT"]).read()

    assert paths == ["/market/AAPL/ohlcv", "/market/MSFT/ohlcv"]
    symbols = out.columns.get_level_values(1).unique().tolist()
    assert sorted(symbols) == ["AAPL", "MSFT"]


def test_weekly_interval_maps_to_1w(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        captured["params"] = dict(params or {})
        return _fake_ohlcv_payload("AAPL")

    monkeypatch.setattr(fc_mod.FundCloudClient, "get", fake_get)

    fc_mod.FundCloud("AAPL", interval="1wk").read()
    assert captured["params"]["timeframe"] == "1W"


def test_adjusted_false_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    captured: dict[str, Any] = {}

    def fake_get(self: Any, path: str, params: Any = None) -> Any:
        captured["params"] = dict(params or {})
        return _fake_ohlcv_payload("AAPL")

    monkeypatch.setattr(fc_mod.FundCloudClient, "get", fake_get)

    fc_mod.FundCloud("AAPL", adjust=False).read()
    assert captured["params"]["adjusted"] == "false"


def test_start_end_slice_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    monkeypatch.setattr(
        fc_mod.FundCloudClient,
        "get",
        lambda self, path, params=None: _fake_ohlcv_payload("AAPL"),
    )

    # Full range has 3 bars (2024-01-02, 03, 04); slice to just 01-03.
    out = fc_mod.FundCloud("AAPL").read(start="2024-01-03", end="2024-01-03")
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2024-01-03")


def test_invalid_interval_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    with pytest.raises(ValueError, match="interval '1m' not supported"):
        fc_mod.FundCloud("AAPL", interval="1m")


def test_invalid_period_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    with pytest.raises(ValueError, match="period '5Y' not supported"):
        fc_mod.FundCloud("AAPL", period="5Y")


def test_empty_symbols_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    with pytest.raises(ValueError, match="at least one symbol"):
        fc_mod.FundCloud([])


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod
    from fundcloud.errors import AuthError

    monkeypatch.delenv("FUNDCLOUD_API_KEY", raising=False)
    with pytest.raises(AuthError, match="FUNDCLOUD_API_KEY"):
        fc_mod.FundCloud("AAPL")


def test_empty_payload_returns_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fundcloud as fc_mod

    monkeypatch.setattr(
        fc_mod.FundCloudClient,
        "get",
        lambda self, path, params=None: {"symbol": "AAPL", "candles": []},
    )
    out = fc_mod.FundCloud("AAPL").read()
    assert out.empty


def test_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import ReadOnlyError
    from fundcloud.data import fundcloud as fc_mod

    src = fc_mod.FundCloud("AAPL")
    assert src.read_only is True
    with pytest.raises(ReadOnlyError):
        src.write("AAPL", pd.DataFrame())


def test_lazy_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """fc.data.FundCloud must be accessible via lazy __getattr__."""
    from fundcloud.data import FundCloud as LazyImported
    from fundcloud.data.fundcloud import FundCloud as DirectImported

    assert LazyImported is DirectImported
