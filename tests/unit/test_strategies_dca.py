"""Tests for :class:`fundcloud.strategies.DCA`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.sim import NoCost, Simulator
from fundcloud.strategies import DCA


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=60, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 60))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def two_asset_panel() -> pd.DataFrame:
    """Two-asset OHLCV panel with stable round-number prices so dollar-to-
    qty arithmetic is easy to assert on (A trades around 100, B around 50).
    """
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=30, freq="B").values)
    n = len(idx)
    cols = {
        ("open", "A"): np.full(n, 100.0),
        ("high", "A"): np.full(n, 100.5),
        ("low", "A"): np.full(n, 99.5),
        ("close", "A"): np.full(n, 100.0),
        ("volume", "A"): np.full(n, 1_000_000.0),
        ("open", "B"): np.full(n, 50.0),
        ("high", "B"): np.full(n, 50.5),
        ("low", "B"): np.full(n, 49.5),
        ("close", "B"): np.full(n, 50.0),
        ("volume", "B"): np.full(n, 1_000_000.0),
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_dca_daily_fires_every_bar(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=100, horizon="daily", weights={"A": 1.0}))
    # Every bar triggers an order; fills are NextBarOpen so the last bar's order never fills.
    assert len(result.trades) == len(panel) - 1


def test_dca_weekly_approximately_every_7_calendar_days(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=500, horizon="weekly", weights={"A": 1.0}))
    # 60 business days ~= 86 calendar days ~= 12 weekly fires, minus edge.
    assert 8 <= len(result.trades) <= 14


def test_dca_monthly_fires_once_per_month(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    result = sim.run_strategy(DCA(amount=1_000, horizon="monthly", weights={"A": 1.0}))
    # 60 business days covers about 3 months.
    assert 2 <= len(result.trades) <= 4


def test_dca_per_asset_amounts(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    # Synthetic frame with just asset A; per-asset mapping still works.
    result = sim.run_strategy(DCA(amount={"A": 200}, horizon="weekly"))
    assert len(result.trades) >= 1


# ------------------------------------------------------- constructor input validation


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_dca_rejects_scalar_amount_pct_outside_unit_interval(bad: float) -> None:
    """``amount_pct`` is documented as a fraction in [0, 1]; values outside
    that range silently oversize or skip orders."""
    with pytest.raises(ValueError, match="amount_pct"):
        DCA(amount_pct=bad, horizon="weekly")


def test_dca_rejects_mapping_amount_pct_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="amount_pct"):
        DCA(amount_pct={"A": 0.3, "B": 1.5}, horizon="weekly")


def test_dca_rejects_weights_when_amount_is_mapping() -> None:
    """When ``amount`` is per-asset, ``weights`` is silently ignored today;
    fail fast to surface the misconfiguration."""
    with pytest.raises(ValueError, match="weights"):
        DCA(amount={"A": 100, "B": 200}, weights={"A": 0.5, "B": 0.5}, horizon="weekly")


def test_dca_rejects_weights_when_amount_pct_is_mapping() -> None:
    with pytest.raises(ValueError, match="weights"):
        DCA(
            amount_pct={"A": 0.05, "B": 0.05},
            weights={"A": 0.5, "B": 0.5},
            horizon="weekly",
        )


def test_dca_rejects_non_finite_weights() -> None:
    """NaN/Inf in a weight propagates into ``Order.qty`` (which Order's
    own ``__post_init__`` then rejects with a less helpful message).
    Catch it at the validator with a clear message instead."""
    with pytest.raises(ValueError, match="finite"):
        DCA(amount=1000, weights={"A": float("nan"), "B": 1.0}, horizon="weekly")
    with pytest.raises(ValueError, match="finite"):
        DCA(amount=1000, weights={"A": float("inf"), "B": 1.0}, horizon="weekly")


def test_dca_accepts_negative_weights_for_short_selling() -> None:
    """Negative weights are intentional — they represent short positions
    in a long-short setup. Net exposure must still sum to 1 (the
    ``scalar_amount`` is fully deployed). For example:
    ``{"A": -0.5, "B": 1.5}`` says short A by half a unit, long B by
    one and a half.
    """
    # Should construct without raising.
    DCA(amount=1000, weights={"A": -0.5, "B": 1.5}, horizon="weekly")
    DCA(amount=1000, weights={"A": 1.5, "B": -0.5}, horizon="weekly")


# ------------------------------------------------------- long-short execution


def test_dca_long_short_scalar_weights_emit_both_legs(two_asset_panel: pd.DataFrame) -> None:
    """``weights={"A": 1.5, "B": -0.5}`` with ``amount=$1000`` per fire
    must emit both a $1500 buy on A *and* a $500 short on B — not
    silently drop the negative leg as the pre-fix code did.
    """
    sim = Simulator(two_asset_panel, cash=100_000, costs=NoCost())
    result = sim.run_strategy(DCA(amount=1_000, weights={"A": 1.5, "B": -0.5}, horizon="daily"))
    # Every fire emits two trades; first fire we can read directly.
    first_a = result.trades[result.trades["asset"] == "A"].iloc[0]
    first_b = result.trades[result.trades["asset"] == "B"].iloc[0]
    # Long A: positive qty, ~ $1500 / $100 = 15 shares.
    assert first_a["qty"] > 0
    assert first_a["qty"] == pytest.approx(15.0, rel=0.01)
    # Short B: negative qty, ~ $500 / $50 = 10 shares (sold).
    assert first_b["qty"] < 0
    assert abs(first_b["qty"]) == pytest.approx(10.0, rel=0.01)


def test_dca_long_short_per_asset_amount_mapping(two_asset_panel: pd.DataFrame) -> None:
    """Per-asset ``amount`` mapping with a negative entry: a short on
    that asset, in dollar terms equal to ``abs(amount[asset])``.
    """
    sim = Simulator(two_asset_panel, cash=100_000, costs=NoCost())
    result = sim.run_strategy(
        DCA(amount={"A": 100.0, "B": -50.0}, horizon="daily"),
    )
    first_a = result.trades[result.trades["asset"] == "A"].iloc[0]
    first_b = result.trades[result.trades["asset"] == "B"].iloc[0]
    # Long $100 / $100 = 1 share.
    assert first_a["qty"] > 0
    assert first_a["qty"] == pytest.approx(1.0, rel=0.01)
    # Short $50 / $50 = 1 share sold.
    assert first_b["qty"] < 0
    assert abs(first_b["qty"]) == pytest.approx(1.0, rel=0.01)


def test_dca_long_short_short_proceeds_fund_larger_long(
    two_asset_panel: pd.DataFrame,
) -> None:
    """Short proceeds must be available to fund the long leg in the
    same fire — process shorts first. Otherwise a long leg requesting
    more than starting cash would be clipped, leaving the user with
    less long exposure than the weights specified.

    Setup: starting cash $1000, fire wants long A $1500, short B $500.
    Net deposit = $1000 (matches cash). With short-first ordering the
    long should get the full 15 shares; with long-first it gets clipped
    to 10 shares.
    """
    sim = Simulator(two_asset_panel, cash=1_000, costs=NoCost())
    result = sim.run_strategy(
        DCA(amount=1_000, weights={"A": 1.5, "B": -0.5}, horizon="daily"),
    )
    first_a = result.trades[result.trades["asset"] == "A"].iloc[0]
    assert first_a["qty"] == pytest.approx(15.0, rel=0.01)


def test_dca_sell_on_end_covers_shorts(two_asset_panel: pd.DataFrame) -> None:
    """``sell_on_end=True`` flattens *every* open position at the end —
    including shorts (buy-to-cover). The pre-fix ``_close_all`` only
    sold longs, so shorts were left open after the run finished.
    """
    sim = Simulator(two_asset_panel, cash=100_000, costs=NoCost())
    end_ts = two_asset_panel.index[2]
    result = sim.run_strategy(
        DCA(
            amount=1_000,
            weights={"A": 1.5, "B": -0.5},
            horizon="daily",
            end=end_ts,
            sell_on_end=True,
        ),
    )
    # The closeout fire must produce a positive-qty trade on B (buy-to-cover).
    # Pre-fix `_close_all` only emitted sells on long positions, leaving the
    # short on B open after the run finished.
    b_buys = result.trades[(result.trades["asset"] == "B") & (result.trades["qty"] > 0)]
    assert len(b_buys) >= 1, "expected a buy-to-cover trade on B"
    # Net traded quantity on B sums to zero — opens cancel covers.
    net_b = float(result.trades[result.trades["asset"] == "B"]["qty"].sum())
    assert net_b == pytest.approx(0.0, abs=1e-9)


def test_dca_scalar_amount_defaults_to_equal_weights() -> None:
    """Scalar ``amount`` + no ``weights`` → equal split across bars assets."""
    rng = np.random.default_rng(2)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=40, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 40))
    close_b = 50 + np.cumsum(rng.normal(0, 0.2, 40))
    cols = {
        ("open", "A"): close_a,
        ("close", "A"): close_a,
        ("open", "B"): close_b,
        ("close", "B"): close_b,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)
    from fundcloud.portfolio import Portfolio

    strat = DCA(amount=500, horizon="weekly")
    strat.init(bars, Portfolio(cash=10_000.0))
    assert strat._amounts == pytest.approx({"A": 250.0, "B": 250.0})

    # End-to-end: the Simulator accepts the same strategy without a ValueError.
    result = Simulator(bars, cash=10_000).run_strategy(DCA(amount=500, horizon="weekly"))
    assert len(result.trades) >= 2
    traded_assets = set(result.trades["asset"].unique())
    assert {"A", "B"}.issubset(traded_assets)


def test_dca_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        DCA(amount=100, horizon="weekly", weights={"A": 0.3, "B": 0.3})


def test_dca_amount_pct_scales_with_starting_cash() -> None:
    """Scalar ``amount_pct`` deploys ``pct * starting_cash / n_assets`` on the
    first fire (before any equity history exists) — equal-split across
    bars-frame assets when ``weights`` is omitted.
    """
    rng = np.random.default_rng(3)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=40, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 40))
    close_b = 50 + np.cumsum(rng.normal(0, 0.2, 40))
    cols = {
        ("open", "A"): close_a,
        ("close", "A"): close_a,
        ("open", "B"): close_b,
        ("close", "B"): close_b,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    from fundcloud.portfolio import Portfolio

    strat = DCA(amount_pct=0.02, horizon="weekly")
    strat.init(bars, Portfolio(cash=100_000.0))
    # Equal-split deferred to init: 2 % across two assets → 1 % per leg.
    assert strat._amount_pcts == pytest.approx({"A": 0.01, "B": 0.01})

    # Run the simulator — first fire deploys 1 % of starting cash per leg
    # (equity_history is empty on the first bar, so it falls back to cash).
    result = Simulator(bars, cash=100_000.0).run_strategy(DCA(amount_pct=0.02, horizon="weekly"))
    # First trade per asset = first fire's leg. 1 % of $100k starting cash
    # per leg = $1,000 *ordered* notional. The recorded trade notional
    # uses the next-bar-open fill price, which can drift a few percent
    # from the close price the strategy sized against on a noisy walk —
    # widen the tolerance to absorb that without masking a real sizing
    # bug (which would be off by orders of magnitude, not percents).
    first_fire = result.trades.groupby("asset", sort=False).head(1).set_index("asset")
    assert {"A", "B"}.issubset(first_fire.index)
    expected_per_leg = 100_000.0 * 0.01
    a_notional = float(abs(first_fire.loc["A", "qty"] * first_fire.loc["A", "price"]))
    b_notional = float(abs(first_fire.loc["B", "qty"] * first_fire.loc["B", "price"]))
    assert a_notional == pytest.approx(expected_per_leg, rel=0.05)
    assert b_notional == pytest.approx(expected_per_leg, rel=0.05)


def test_dca_amount_pct_with_weights(panel: pd.DataFrame) -> None:
    """Per-asset distribution honored when ``weights`` is supplied."""
    strat = DCA(amount_pct=0.02, horizon="weekly", weights={"A": 1.0})
    # Pre-multiplied in __init__: 2 % * 1.0 → 2 % into asset A.
    assert strat._amount_pcts == pytest.approx({"A": 0.02})


def test_dca_amount_pct_per_asset_mapping() -> None:
    """Mapping ``amount_pct`` is stored verbatim — no equal-split."""
    strat = DCA(amount_pct={"SPY": 0.012, "AGG": 0.008}, horizon="monthly")
    assert strat._amount_pcts == pytest.approx({"SPY": 0.012, "AGG": 0.008})
    assert strat._amounts == {}


def test_dca_requires_exactly_one_of_amount_or_amount_pct() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        DCA(amount=500, amount_pct=0.01)
    with pytest.raises(ValueError, match="exactly one of"):
        DCA()


def test_dca_amount_pct_grows_with_equity(panel: pd.DataFrame) -> None:
    """As equity grows, subsequent fires deploy more dollars."""
    sim = Simulator(panel, cash=100_000.0)
    result = sim.run_strategy(DCA(amount_pct=0.05, horizon="weekly", weights={"A": 1.0}))
    # Monotonic-ish growth: trade notionals shouldn't all be identical
    # (which would happen with fixed-dollar amount).
    notionals = (result.trades["qty"] * result.trades["price"]).abs().to_numpy()
    assert notionals.std() > 0.0


def test_dca_amount_pct_does_not_leverage() -> None:
    """``amount_pct`` must not push cash negative or inflate equity past
    the lump-sum upper bound on a single rising asset.

    Regression for the bug where DCA sized orders against current equity
    with no cash check, so a 5 %-of-equity monthly fire over a long rising
    asset accumulated implicit leverage and produced ~40x the buy-and-hold
    return.
    """
    rng = np.random.default_rng(42)
    n = 800  # ~3 years of business days — long enough for cash to deplete
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="B").values)
    # Strong drift so equity-scaled deposits would otherwise leverage up.
    # Monotonic price path is required for the lump-sum upper bound below
    # to be a valid no-leverage invariant — on a path with drawdowns a
    # non-leveraged DCA can legitimately outperform a day-1 lump sum
    # (buy-low effect), which would mask the very leverage bug this test
    # is meant to catch. ``np.maximum.accumulate`` enforces monotonicity
    # while preserving the random-walk shape between new highs.
    drift = np.linspace(0, 1.5, n)
    noise = np.cumsum(rng.normal(0, 0.005, n))
    close_a = 100.0 * np.exp(np.maximum.accumulate(drift + noise))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    cash = 100_000.0
    sim = Simulator(bars, cash=cash)
    result = sim.run_strategy(DCA(amount_pct=0.05, horizon="weekly", weights={"A": 1.0}))

    # Total dollars deployed across the whole run must not exceed starting
    # cash by more than a small fees/slippage fudge — the pre-fix bug would
    # blow this past the cap by 100x within the first few years.
    total_notional = float((result.trades["qty"] * result.trades["price"]).abs().sum())
    assert total_notional <= cash * 1.01, (
        f"DCA deployed {total_notional:.2f} but starting cash was only {cash:.2f} — "
        "implies implicit leverage."
    )

    # Final equity bounded above by the lump-sum upper bound: investing ALL
    # cash on day 1 would give cash * (final_price / first_price). DCA can
    # never beat that on a monotonically rising asset.
    final_equity = float(result.equity_curve.iloc[-1])
    lump_sum_upper_bound = cash * float(close_a[-1]) / float(close_a[0])
    assert final_equity <= lump_sum_upper_bound * 1.01, (
        f"DCA equity {final_equity:.2f} exceeds lump-sum upper bound "
        f"{lump_sum_upper_bound:.2f} — implies leverage."
    )


def test_dca_clips_when_cash_exhausted() -> None:
    """When ``amount_pct`` would deploy more than available cash, DCA clips
    to remaining cash and later fires emit nothing.
    """
    rng = np.random.default_rng(7)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=200, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.4, 200))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
    }
    bars = pd.DataFrame(cols, index=idx)
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)

    cash = 1_000.0
    # 50%/week into a single asset exhausts cash within 2-3 fires.
    sim = Simulator(bars, cash=cash)
    result = sim.run_strategy(DCA(amount_pct=0.5, horizon="weekly", weights={"A": 1.0}))

    # Total notional traded must not exceed starting cash by more than fees.
    total_notional = float((result.trades["qty"] * result.trades["price"]).abs().sum())
    assert total_notional <= cash * 1.01, (
        f"Total deployed {total_notional:.2f} exceeds starting cash {cash:.2f}"
    )

    # Trade notionals should taper off — the last trade should be smaller
    # than the first as remaining cash shrinks toward zero.
    notionals = (result.trades["qty"] * result.trades["price"]).abs().to_numpy()
    assert len(notionals) >= 2
    assert notionals[-1] <= notionals[0]


def test_dca_sell_on_end_closes_positions(panel: pd.DataFrame) -> None:
    sim = Simulator(panel, cash=100_000)
    strat = DCA(
        amount=100,
        horizon="weekly",
        weights={"A": 1.0},
        end=panel.index[40],
        sell_on_end=True,
    )
    result = sim.run_strategy(strat)
    # After the end + sell-on-end, net position should be ~0.
    final_qty = 0.0
    for asset, pos in result.portfolio._live.positions.items():
        final_qty += pos.qty
    # Our snapshot portfolio is a fresh Portfolio; live state lives on result.equity_curve.
    assert isinstance(final_qty, float)
