"""Tests for cost + slippage + execution models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import (
    FixedBps,
    HalfSpread,
    NextBarClose,
    NextBarOpen,
    NoCost,
    NoSlippage,
    PerShare,
)

# -------------------------------------------------------------------- costs


def test_no_cost_is_zero() -> None:
    assert NoCost().fee(price=100, qty=10) == 0.0


def test_fixed_bps_scales_with_notional() -> None:
    model = FixedBps(bps=10.0)
    assert model.fee(price=100, qty=10) == pytest.approx(1.0)  # 1000 notional * 10 bps
    assert model.fee(price=100, qty=-20) == pytest.approx(2.0)  # signed qty OK


@pytest.mark.parametrize("bad", [-1.0, -0.0001])
def test_fixed_bps_rejects_negative_bps(bad: float) -> None:
    with pytest.raises(ValueError, match="bps"):
        FixedBps(bps=bad)


def test_fixed_bps_rejects_negative_minimum() -> None:
    with pytest.raises(ValueError, match="minimum"):
        FixedBps(bps=5.0, minimum=-1.0)


@pytest.mark.parametrize("bad", [-0.005, -1.0])
def test_per_share_rejects_negative_rate(bad: float) -> None:
    with pytest.raises(ValueError, match="rate"):
        PerShare(rate=bad)


def test_per_share_rejects_negative_minimum() -> None:
    with pytest.raises(ValueError, match="minimum"):
        PerShare(rate=0.005, minimum=-0.5)


def test_fixed_bps_minimum() -> None:
    model = FixedBps(bps=1.0, minimum=5.0)
    assert model.fee(price=1, qty=1) == 5.0


def test_per_share_flat_rate() -> None:
    model = PerShare(rate=0.01, minimum=0.5)
    assert model.fee(price=100, qty=100) == 1.0
    assert model.fee(price=100, qty=10) == 0.5


# -------------------------------------------------------------------- slippage


def test_no_slippage() -> None:
    px, bps = NoSlippage().apply(price=100.0, side="buy")
    assert px == 100.0
    assert bps == 0.0


def test_half_spread_adjusts_buy_up_sell_down() -> None:
    m = HalfSpread(spread_bps=20)
    buy_px, buy_bps = m.apply(price=100.0, side="buy")
    sell_px, sell_bps = m.apply(price=100.0, side="sell")
    assert buy_px > 100.0
    assert sell_px < 100.0
    assert buy_bps == sell_bps == pytest.approx(10.0)


# -------------------------------------------------------------------- execution


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    cols = pd.MultiIndex.from_tuples([
        ("open", "A"),
        ("high", "A"),
        ("low", "A"),
        ("close", "A"),
        ("volume", "A"),
    ])
    data = np.array(
        [
            [100, 101, 99, 100.5, 1],
            [100.5, 102, 100, 101.5, 1],
            [101.5, 103, 101, 102.5, 1],
            [102.5, 103, 101, 102, 1],
            [102, 104, 101, 103, 1],
        ],
        dtype=float,
    )
    return pd.DataFrame(data, index=idx, columns=cols)


def test_next_bar_open_references_next_bar_open(ohlcv: pd.DataFrame) -> None:
    ex = NextBarOpen()
    assert ex.fill_at(signal_index=0, bars_index_size=5) == 1
    assert ex.fill_at(signal_index=4, bars_index_size=5) is None
    price = ex.reference_price(bars=ohlcv, fill_index=1, asset="A")
    assert price == pytest.approx(100.5)


def test_next_bar_close_references_next_bar_close(ohlcv: pd.DataFrame) -> None:
    """``NextBarClose`` fills at the close of bar ``signal_index + 1`` — no look-ahead."""
    ex = NextBarClose()
    assert ex.fill_at(signal_index=2, bars_index_size=5) == 3
    assert ex.fill_at(signal_index=4, bars_index_size=5) is None
    # bar 1's close is 101.5 in the fixture
    price = ex.reference_price(bars=ohlcv, fill_index=1, asset="A")
    assert price == pytest.approx(101.5)
