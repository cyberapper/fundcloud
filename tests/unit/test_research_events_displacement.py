"""Synthetic-fixture tests for ``fundcloud.research.events._displacement``.

Builds tiny single-asset OHLCV frames where the displacement condition is
hand-controllable, then asserts:

* a strong up bar fires exactly one ``ev_disp_up`` row with the right
  ``confirmed_ts`` and NaN zone/stop,
* a strong down bar fires exactly one ``ev_disp_dn`` row,
* a quiet, noiseless frame fires nothing,
* the volume gate suppresses an otherwise-qualifying bar,
* :func:`assert_prefix_invariant` passes — the proof there is no future-bar leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.research.events._causality import assert_prefix_invariant
from fundcloud.research.events._displacement import detect_displacement
from fundcloud.research.events.schema import OBSERVATION_COLUMNS

ATR_N = 5


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")


def _flat_bars(n: int, price: float = 100.0) -> dict[str, np.ndarray]:
    """Tiny-range flat bars: a 0.2-wide doji every day (establishes a small ATR)."""
    open_ = np.full(n, price)
    close = np.full(n, price)
    high = np.full(n, price + 0.1)
    low = np.full(n, price - 0.1)
    volume = np.full(n, 1_000.0)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _frame(cols: dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(cols["open"])
    return pd.DataFrame(cols, index=_index(n))


def test_bullish_displacement_fires() -> None:
    cols = _flat_bars(10)
    t = 7  # past the ATR warmup (t >= ATR_N)
    # Strong up bar closing at its high: body ~5 >> z_body * atr[t-1] (~0.2).
    cols["open"][t] = 100.0
    cols["low"][t] = 99.9
    cols["high"][t] = 105.0
    cols["close"][t] = 105.0  # clv == 1.0

    bars = _frame(cols)
    out = detect_displacement(bars, asset="AAA", atr_n=ATR_N)

    assert list(out.columns) == list(OBSERVATION_COLUMNS)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_disp_up"
    assert row["asset"] == "AAA"
    assert row["confirmed_ts"] == bars.index[t]
    assert row["formation_end_ts"] == bars.index[t]
    assert row["execution_ts"] == bars.index[t + 1]
    assert row["entry_ref_price"] == bars["open"].iloc[t + 1]
    assert np.isnan(row["zone_lo"]) and np.isnan(row["zone_hi"])
    assert np.isnan(row["stop_ref_price"]) and np.isnan(row["quality"])
    assert np.isfinite(row["atr_at_confirm"])


def test_bearish_displacement_fires() -> None:
    cols = _flat_bars(10)
    t = 7
    # Strong down bar closing at its low: clv == 0.0.
    cols["open"][t] = 100.0
    cols["high"][t] = 100.1
    cols["low"][t] = 95.0
    cols["close"][t] = 95.0

    bars = _frame(cols)
    out = detect_displacement(bars, asset="BBB", atr_n=ATR_N)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["event_id"] == "ev_disp_dn"
    assert row["confirmed_ts"] == bars.index[t]


def test_both_branches_share_one_params_hash() -> None:
    # A strong up bar at t=6 and a strong down bar at t=8 -> one of each branch,
    # both from the same detector call, so they share one params_hash.
    cols = _flat_bars(12)
    cols["open"][6] = 100.0
    cols["low"][6] = 99.9
    cols["high"][6] = 105.0
    cols["close"][6] = 105.0
    cols["open"][8] = 100.0
    cols["high"][8] = 100.1
    cols["low"][8] = 95.0
    cols["close"][8] = 95.0

    out = detect_displacement(_frame(cols), asset="AAA", atr_n=ATR_N)

    assert set(out["event_id"]) == {"ev_disp_up", "ev_disp_dn"}
    assert out["params_hash"].nunique() == 1


def test_last_bar_has_no_execution() -> None:
    cols = _flat_bars(9)
    t = 8  # final bar
    cols["open"][t] = 100.0
    cols["low"][t] = 99.9
    cols["high"][t] = 105.0
    cols["close"][t] = 105.0

    bars = _frame(cols)
    out = detect_displacement(bars, atr_n=ATR_N)

    assert len(out) == 1
    row = out.iloc[0]
    assert pd.isna(row["execution_ts"])
    assert np.isnan(row["entry_ref_price"])


def test_quiet_frame_yields_nothing() -> None:
    bars = _frame(_flat_bars(15))
    out = detect_displacement(bars, atr_n=ATR_N)
    assert out.empty
    assert list(out.columns) == list(OBSERVATION_COLUMNS)


def test_empty_frame_yields_empty_with_columns() -> None:
    empty = pd.DataFrame(
        {c: np.array([], dtype=float) for c in ("open", "high", "low", "close", "volume")},
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    out = detect_displacement(empty, atr_n=ATR_N)
    assert out.empty
    assert list(out.columns) == list(OBSERVATION_COLUMNS)


def test_volume_gate_suppresses() -> None:
    cols = _flat_bars(10)
    t = 7
    cols["open"][t] = 100.0
    cols["low"][t] = 99.9
    cols["high"][t] = 105.0
    cols["close"][t] = 105.0
    # Bar volume equals the trailing mean (1.0x) -> a z_vol=2.0 gate must reject.
    out = detect_displacement(_frame(cols), atr_n=ATR_N, z_vol=2.0)
    assert out.empty

    # Same bar with a volume spike clears the gate.
    cols["volume"][t] = 10_000.0
    out2 = detect_displacement(_frame(cols), atr_n=ATR_N, z_vol=2.0)
    assert len(out2) == 1
    assert out2.iloc[0]["event_id"] == "ev_disp_up"


def test_prefix_invariant_no_future_leak() -> None:
    rng = np.random.default_rng(7)
    n = 60
    base = 100.0 + np.cumsum(rng.normal(0.0, 0.3, size=n))
    open_ = base.copy()
    close = base + rng.normal(0.0, 0.2, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.2, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.2, size=n))
    volume = rng.uniform(800.0, 1_200.0, size=n)

    # Inject a few decisive displacement bars so the detector actually emits.
    for t, sign in ((20, 1), (33, -1), (47, 1)):
        if sign == 1:
            open_[t] = base[t]
            low[t] = base[t] - 0.05
            high[t] = base[t] + 4.0
            close[t] = high[t]
        else:
            open_[t] = base[t]
            high[t] = base[t] + 0.05
            low[t] = base[t] - 4.0
            close[t] = low[t]
        volume[t] = 5_000.0

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=_index(n),
    )

    assert_prefix_invariant(detect_displacement, bars, asset="ZZZ", atr_n=ATR_N, z_vol=1.5)
