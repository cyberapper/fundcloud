"""Smoke tests for network data backends — YF / FMP / AV / Binance.

All four wrap external services (yfinance, FMP REST, Alpha Vantage REST,
Binance via ccxt). The tests stay offline by monkey-patching the
relevant client method per backend, focusing on:

* construction validation (symbols, interval, missing API key)
* the ``read`` happy path with a stub response
* the empty-response branch
* the ``columns=`` and ``start/end`` slicers
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------- _defaults


def test_default_start_one_year_back_passes_through_when_set() -> None:
    from fundcloud.data._defaults import default_start_one_year_back

    given = pd.Timestamp("2024-01-15")
    assert default_start_one_year_back(given, None) == given


def test_default_start_one_year_back_computes_from_end() -> None:
    from fundcloud.data._defaults import default_start_one_year_back

    end = pd.Timestamp("2024-04-15")
    out = default_start_one_year_back(None, end)
    assert isinstance(out, pd.Timestamp)
    assert out == end - pd.DateOffset(years=1)


def test_default_start_one_year_back_uses_today_when_both_none() -> None:
    from fundcloud.data._defaults import default_start_one_year_back

    out = default_start_one_year_back(None, None)
    assert isinstance(out, pd.Timestamp)
    delta = pd.Timestamp.now().normalize() - pd.DateOffset(years=1) - out
    assert abs(delta.total_seconds()) < 86_400  # within one day


def test_interval_aware_default_start_minute_intervals() -> None:
    from fundcloud.data._defaults import interval_aware_default_start

    end = pd.Timestamp("2024-04-15")
    one_min = interval_aware_default_start("1m", end)
    five_min = interval_aware_default_start("5m", end)
    one_hour = interval_aware_default_start("1h", end)
    daily_default = interval_aware_default_start("1d", end)

    assert (end - one_min).days == 7
    assert (end - five_min).days == 60
    assert (end - one_hour).days == 730
    assert (end - daily_default).days == 365


def test_interval_aware_default_start_uses_now_when_end_none() -> None:
    from fundcloud.data._defaults import interval_aware_default_start

    out = interval_aware_default_start("1d", None)
    assert isinstance(out, pd.Timestamp)


# --------------------------------------------------------------------- _http


def test_httpclient_close_and_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data._http import HttpClient

    closed: list[bool] = []

    class _FakeHttpx:
        ConnectError = ConnectionError
        ReadTimeout = TimeoutError
        RemoteProtocolError = RuntimeError

        class Client:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def close(self) -> None:
                closed.append(True)

            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError

    monkeypatch.setattr("fundcloud.data._http.require_httpx", lambda: _FakeHttpx)
    with HttpClient(base_url="x") as c:
        assert isinstance(c, HttpClient)
    assert closed == [True]


def test_httpclient_get_json_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first 503 then a 200 → retry, then success."""
    from fundcloud.data._http import HttpClient

    pytest.importorskip("httpx")

    calls: list[int] = []

    class _Resp:
        def __init__(self, status: int, body: dict[str, Any]) -> None:
            self.status_code = status
            self._body = body

        def json(self) -> dict[str, Any]:
            return self._body

        def raise_for_status(self) -> None:
            return None

    responses = [_Resp(503, {}), _Resp(200, {"ok": True})]

    client = HttpClient(base_url="https://example.test")

    def fake_get(url: str, params: Any = None) -> _Resp:
        calls.append(len(calls))
        return responses[len(calls) - 1]

    monkeypatch.setattr(client._client, "get", fake_get)
    out = client.get_json("/path")
    assert out == {"ok": True}
    assert len(calls) == 2


def test_httpclient_get_json_raises_for_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 propagates `httpx.HTTPStatusError` after `raise_for_status`."""
    from fundcloud.data._http import HttpClient

    httpx = pytest.importorskip("httpx")

    class _Resp:
        def __init__(self, status: int) -> None:
            self.status_code = status

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("GET", "https://example.test/path"),
                response=httpx.Response(self.status_code),
            )

        def json(self) -> dict[str, Any]:
            return {}

    client = HttpClient(base_url="https://example.test")
    monkeypatch.setattr(client._client, "get", lambda url, params=None: _Resp(404))
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("/missing")


# --------------------------------------------------------------------- YF


def _yf_panel(symbols: list[str], n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    cols: dict[tuple[str, str], np.ndarray] = {}
    for s in symbols:
        base = 100 + np.cumsum(rng.normal(0, 0.5, n))
        cols[("Open", s)] = base
        cols[("High", s)] = base + 1
        cols[("Low", s)] = base - 1
        cols[("Close", s)] = base
        cols[("Volume", s)] = np.full(n, 1_000_000.0)
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_yf_construction_rejects_empty_symbols() -> None:
    from fundcloud.data.yf import YF

    with pytest.raises(ValueError, match="at least one symbol"):
        YF([])


def test_yf_construction_rejects_unknown_interval() -> None:
    from fundcloud.data.yf import YF

    with pytest.raises(ValueError, match="not supported"):
        YF("AAPL", interval="bogus")


def test_yf_keys_lists_symbols() -> None:
    from fundcloud.data.yf import YF

    src = YF(["AAPL", "MSFT"])
    assert src.keys() == ["AAPL", "MSFT"]


def test_yf_read_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import yf as yf_mod

    class _StubYF:
        @staticmethod
        def download(**kwargs: Any) -> pd.DataFrame:
            return _yf_panel(kwargs["tickers"])

    monkeypatch.setattr(yf_mod, "_require_yfinance", lambda: _StubYF)
    out = yf_mod.YF(["AAPL", "MSFT"]).read()
    assert not out.empty
    assert isinstance(out.columns, pd.MultiIndex)


def test_yf_read_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import yf as yf_mod

    class _StubYF:
        @staticmethod
        def download(**kwargs: Any) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(yf_mod, "_require_yfinance", lambda: _StubYF)
    out = yf_mod.YF("AAPL").read()
    assert out.empty


def test_yf_columns_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import yf as yf_mod

    class _StubYF:
        @staticmethod
        def download(**kwargs: Any) -> pd.DataFrame:
            return _yf_panel(["AAPL"])

    monkeypatch.setattr(yf_mod, "_require_yfinance", lambda: _StubYF)
    out = yf_mod.YF("AAPL").read(columns=["close"])
    fields = [c[0] for c in out.columns]
    assert fields == ["close"]


def test_yf_normalises_flat_single_symbol_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """yfinance returns a flat-column frame for a single symbol; the
    normaliser builds the MultiIndex."""
    from fundcloud.data import yf as yf_mod

    class _StubYF:
        @staticmethod
        def download(**kwargs: Any) -> pd.DataFrame:
            idx = pd.date_range("2024-01-02", periods=5, freq="B")
            return pd.DataFrame(
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000.0},
                index=idx,
            )

    monkeypatch.setattr(yf_mod, "_require_yfinance", lambda: _StubYF)
    out = yf_mod.YF("AAPL").read()
    assert isinstance(out.columns, pd.MultiIndex)


def test_yf_require_yfinance_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runtime guard raises a friendly ImportError when yfinance is absent."""
    import builtins

    from fundcloud.data.yf import _require_yfinance

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "yfinance":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="yfinance"):
        _require_yfinance()


# --------------------------------------------------------------------- FMP


def _fmp_daily_payload(symbol: str, n: int = 3) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "historical": [
            {
                "date": f"2024-01-0{i + 2}",
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "adjClose": 100.5 + i,
                "volume": 1_000_000,
            }
            for i in range(n)
        ],
    }


def test_fmp_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.fmp import FMP

    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        FMP("AAPL")


def test_fmp_rejects_empty_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.fmp import FMP

    monkeypatch.setenv("FMP_API_KEY", "x")
    with pytest.raises(ValueError, match="at least one symbol"):
        FMP([])


def test_fmp_rejects_unknown_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.fmp import FMP

    monkeypatch.setenv("FMP_API_KEY", "x")
    with pytest.raises(ValueError, match="not supported"):
        FMP("AAPL", interval="bogus")


def test_fmp_read_daily_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fmp as fmp_mod

    monkeypatch.setenv("FMP_API_KEY", "x")
    captured: dict[str, Any] = {}

    def fake_get_json(self: Any, url: str, *, params: Any = None) -> Any:
        captured["url"] = url
        captured["params"] = dict(params or {})
        sym = url.rsplit("/", 1)[-1]
        return _fmp_daily_payload(sym)

    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", fake_get_json)
    out = fmp_mod.FMP("AAPL").read()
    assert "/historical-price-full/" in captured["url"]
    assert not out.empty
    fields = [c[0] for c in out.columns]
    assert "close" in fields


def test_fmp_read_intraday_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-daily endpoint path returns a list of bars (not a wrapped dict)."""
    from fundcloud.data import fmp as fmp_mod

    monkeypatch.setenv("FMP_API_KEY", "x")

    def fake_get_json(self: Any, url: str, *, params: Any = None) -> Any:
        return [
            {
                "date": "2024-01-02 09:30:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100,
            }
        ]

    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", fake_get_json)
    out = fmp_mod.FMP("AAPL", interval="5m").read()
    assert not out.empty


def test_fmp_read_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fmp as fmp_mod

    monkeypatch.setenv("FMP_API_KEY", "x")
    monkeypatch.setattr(fmp_mod.HttpClient, "get_json", lambda self, url, params=None: {})
    out = fmp_mod.FMP("AAPL").read()
    assert out.empty


def test_fmp_keys() -> None:
    from fundcloud.data.fmp import FMP

    src = FMP(["AAPL", "MSFT"], api_key="x")
    assert src.keys() == ["AAPL", "MSFT"]


def test_fmp_no_adjust_keeps_raw_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import fmp as fmp_mod

    monkeypatch.setenv("FMP_API_KEY", "x")
    monkeypatch.setattr(
        fmp_mod.HttpClient, "get_json", lambda self, url, params=None: _fmp_daily_payload("AAPL")
    )
    out = fmp_mod.FMP("AAPL", adjust=False).read()
    # Without adjust, raw close (100.5) is published.
    assert out.iloc[0][("close", "AAPL")] == 100.5


# --------------------------------------------------------------------- AV


def _av_daily_payload(n: int = 3) -> dict[str, Any]:
    """Recent-dated AV payload — the AV backend slices by `start = end - 1y`
    by default, so synthetic dates need to be within the last year."""
    today = pd.Timestamp.utcnow().normalize()
    return {
        "Time Series (Daily)": {
            (today - pd.Timedelta(days=i)).strftime("%Y-%m-%d"): {
                "1. open": str(100.0 + i),
                "2. high": str(101.0 + i),
                "3. low": str(99.0 + i),
                "4. close": str(100.5 + i),
                "5. adjusted close": str(100.5 + i),
                "6. volume": str(1_000_000),
                "5. volume": str(1_000_000),
            }
            for i in range(n)
        }
    }


def test_av_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.av import AV

    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        AV("AAPL")


def test_av_rejects_empty_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.av import AV

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    with pytest.raises(ValueError, match="at least one symbol"):
        AV([])


def test_av_rejects_unknown_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data.av import AV

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    with pytest.raises(ValueError, match="not supported"):
        AV("AAPL", interval="bogus")


def test_av_read_daily_adjusted(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import av as av_mod

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    monkeypatch.setattr(
        av_mod.HttpClient, "get_json", lambda self, url, params=None: _av_daily_payload()
    )
    out = av_mod.AV("AAPL").read()
    assert not out.empty
    fields = [c[0] for c in out.columns]
    assert "close" in fields


def test_av_read_unadjusted(monkeypatch: pytest.MonkeyPatch) -> None:
    """`adjust=False` hits the unadjusted endpoint family with `4. close`."""
    from fundcloud.data import av as av_mod

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    monkeypatch.setattr(
        av_mod.HttpClient, "get_json", lambda self, url, params=None: _av_daily_payload()
    )
    out = av_mod.AV("AAPL", adjust=False).read()
    assert not out.empty


def test_av_read_empty_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import av as av_mod

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    monkeypatch.setattr(av_mod.HttpClient, "get_json", lambda self, url, params=None: {})
    out = av_mod.AV("AAPL").read()
    assert out.empty


def test_av_columns_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import av as av_mod

    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "x")
    monkeypatch.setattr(
        av_mod.HttpClient, "get_json", lambda self, url, params=None: _av_daily_payload()
    )
    out = av_mod.AV("AAPL").read(columns=["open"])
    fields = [c[0] for c in out.columns]
    assert fields == ["open"]


def test_av_keys() -> None:
    from fundcloud.data.av import AV

    src = AV("AAPL", api_key="x")
    assert src.keys() == ["AAPL"]


# --------------------------------------------------------------------- Binance


def _binance_batch(n: int = 3, start_ts_ms: int = 1_700_000_000_000) -> list[list[float]]:
    bars = []
    for i in range(n):
        ts = start_ts_ms + i * 86_400_000  # +1 day each
        bars.append([float(ts), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1_000.0])
    return bars


class _StubExchange:
    def __init__(self, batches: list[list[list[float]]]) -> None:
        self._batches = batches
        self.calls: list[dict[str, Any]] = []

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, *, since: int | None = None, limit: int = 1000
    ) -> list[list[float]]:
        self.calls.append({
            "symbol": symbol,
            "timeframe": timeframe,
            "since": since,
            "limit": limit,
        })
        return self._batches.pop(0) if self._batches else []


def test_binance_rejects_empty_symbols() -> None:
    from fundcloud.data.binance import Binance

    with pytest.raises(ValueError, match="at least one symbol"):
        Binance([])


def test_binance_rejects_unknown_interval() -> None:
    from fundcloud.data.binance import Binance

    with pytest.raises(ValueError, match="not supported"):
        Binance("BTC/USDT", interval="bogus")


def test_binance_read_with_stub_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import binance as bn_mod

    stub = _StubExchange([_binance_batch(3), []])
    src = bn_mod.Binance("BTC/USDT")
    monkeypatch.setattr(src, "_exchange", lambda: stub)
    out = src.read()
    assert not out.empty
    fields = [c[0] for c in out.columns]
    assert "close" in fields


def test_binance_read_paginates_until_short_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first full-limit batch then a partial batch terminates pagination."""
    from fundcloud.data import binance as bn_mod

    full = _binance_batch(2)
    partial = _binance_batch(1, start_ts_ms=1_700_172_800_000)
    stub = _StubExchange([full, partial])
    src = bn_mod.Binance("BTC/USDT", limit=2)
    monkeypatch.setattr(src, "_exchange", lambda: stub)
    out = src.read()
    assert len(out) == 3


def test_binance_read_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import binance as bn_mod

    stub = _StubExchange([[]])
    src = bn_mod.Binance("BTC/USDT")
    monkeypatch.setattr(src, "_exchange", lambda: stub)
    out = src.read()
    assert out.empty


def test_binance_columns_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from fundcloud.data import binance as bn_mod

    stub = _StubExchange([_binance_batch(3), []])
    src = bn_mod.Binance("BTC/USDT")
    monkeypatch.setattr(src, "_exchange", lambda: stub)
    out = src.read(columns=["open"])
    fields = [c[0] for c in out.columns]
    assert fields == ["open"]


def test_binance_read_with_explicit_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """`start=` and `end=` pass through to the exchange via since/until."""
    from fundcloud.data import binance as bn_mod

    stub = _StubExchange([_binance_batch(3), []])
    src = bn_mod.Binance("BTC/USDT")
    monkeypatch.setattr(src, "_exchange", lambda: stub)
    out = src.read(start="2023-11-15", end="2023-11-25")
    assert isinstance(out, pd.DataFrame)


def test_binance_keys() -> None:
    from fundcloud.data.binance import Binance

    src = Binance(["BTC/USDT", "ETH/USDT"])
    assert src.keys() == ["BTC/USDT", "ETH/USDT"]


def test_binance_exchange_lazy_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_exchange()` lazily constructs and caches the ccxt client."""
    from fundcloud.data import binance as bn_mod

    constructed: list[bool] = []

    class _FakeBinanceExchange:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructed.append(True)

        def set_sandbox_mode(self, on: bool) -> None:
            pass

    class _FakeCcxt:
        binance = _FakeBinanceExchange

    monkeypatch.setattr(bn_mod, "_require_ccxt", lambda: _FakeCcxt)
    src = bn_mod.Binance("BTC/USDT", sandbox=True)
    ex = src._exchange()
    assert ex is not None
    # Second call returns the cached instance.
    assert src._exchange() is ex
    assert len(constructed) == 1


def test_binance_require_ccxt_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from fundcloud.data.binance import _require_ccxt

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ccxt":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ccxt"):
        _require_ccxt()


# --------------------------------------------------------------------- bars edges


def test_bars_to_prices_keyerror_on_missing_field() -> None:
    from fundcloud.data.bars import to_prices

    idx = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame(
        {("close", "A"): [1.0, 2.0, 3.0]},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    with pytest.raises(KeyError, match="not found"):
        to_prices(df, field="open")  # type: ignore[arg-type]


def test_bars_to_returns_log_method() -> None:
    from fundcloud.data.bars import to_returns

    idx = pd.bdate_range("2024-01-02", periods=4)
    s = pd.Series([100.0, 101.0, 99.0, 102.0], index=idx)
    out = to_returns(s, method="log")
    assert isinstance(out, pd.Series)
    # log return between 100 and 101.
    assert np.isclose(out.iloc[0], np.log(101 / 100))


def test_bars_to_returns_dataframe_input() -> None:
    """`to_returns` on a MultiIndex Bars frame extracts the close field via
    `to_prices` first, so the output has flat columns (one per asset)."""
    from fundcloud.data.bars import to_returns

    idx = pd.bdate_range("2024-01-02", periods=4)
    df = pd.DataFrame(
        {("close", "A"): [100, 101, 99, 102], ("close", "B"): [50, 50.5, 49, 51]},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    out = to_returns(df)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"A", "B"}


def test_bars_align_inner_intersection() -> None:
    from fundcloud.data.bars import align

    idx_a = pd.bdate_range("2024-01-02", periods=5)
    idx_b = pd.bdate_range("2024-01-04", periods=5)
    a = pd.DataFrame({"x": range(5)}, index=idx_a)
    b = pd.DataFrame({"x": range(5)}, index=idx_b)
    aligned = align(a, b)
    # Intersection drops the non-overlapping start days.
    assert len(aligned[0]) < 5


def test_bars_align_empty_input() -> None:
    from fundcloud.data.bars import align

    assert align() == []


def test_bars_resample_multiindex() -> None:
    from fundcloud.data.bars import resample

    idx = pd.bdate_range("2024-01-02", periods=10)
    df = pd.DataFrame(
        {
            ("open", "A"): np.arange(10, dtype=float),
            ("high", "A"): np.arange(10, 20, dtype=float),
            ("low", "A"): np.arange(-10, 0, dtype=float),
            ("close", "A"): np.arange(100, 110, dtype=float),
            ("volume", "A"): np.full(10, 1000.0),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    weekly = resample(df, "W")
    assert len(weekly) <= 3  # 10 business days span ~2 weeks


def test_bars_resample_flat_columns() -> None:
    from fundcloud.data.bars import resample

    idx = pd.bdate_range("2024-01-02", periods=10)
    df = pd.DataFrame({"close": np.arange(10, dtype=float)}, index=idx)
    out = resample(df, "W")
    assert len(out) <= 3


def test_bars_long_wide_roundtrip() -> None:
    from fundcloud.data.bars import as_long, as_wide

    idx = pd.bdate_range("2024-01-02", periods=3)
    wide = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    long = as_long(wide, value_name="px")
    assert set(long.columns) == {"ts", "asset", "px"}

    rewide = as_wide(long, value="px")
    assert rewide.shape == wide.shape


def test_to_log_returns_alias() -> None:
    from fundcloud.data.bars import to_log_returns, to_returns

    idx = pd.bdate_range("2024-01-02", periods=4)
    s = pd.Series([100.0, 101.0, 99.0, 102.0], index=idx)
    out_alias = to_log_returns(s)
    out_direct = to_returns(s, method="log")
    assert np.allclose(out_alias.to_numpy(), out_direct.to_numpy())
