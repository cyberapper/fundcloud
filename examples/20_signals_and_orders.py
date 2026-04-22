"""20 — Signal-driven and order-driven backtests, plus custom indicators.

Sometimes you don't want to write a ``Strategy`` subclass — you already have:

* a **boolean entry / exit matrix** produced by a screener, or
* an **orders log** exported from a research notebook.

``Simulator`` exposes two entry points for exactly these cases:

* :meth:`Simulator.run_signals` — boolean ``entries`` / ``exits`` DataFrames,
  one column per asset. Each ``True`` buy allocates a target fraction of
  current equity (``size=``) to that asset; each ``True`` sell closes it.
* :meth:`Simulator.run_orders` — an explicit long-format frame with columns
  ``ts / asset / side / qty``; the simulator executes them as-is.

As a bonus, the same example shows how to extend the TA-Lib catalogue with
your own indicator via :func:`fundcloud.features.indicators.register_indicator`
— the signal factory below uses it.

Run:
    uv add 'fundcloud[ta,data-yf]'
    uv run python examples/20_signals_and_orders.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from _data import pull_closes

from fundcloud.features.indicators import IndicatorSpec, register_indicator
from fundcloud.sim import FixedBps, Simulator

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


# --------------------------------------------- a custom indicator via the hook


@register_indicator("zscore")
class ZScore(IndicatorSpec):
    """Rolling z-score of close. Not in TA-Lib — we add it ourselves."""

    talib_name = None
    inputs = ("close",)
    outputs = ("value",)
    default_params = {"timeperiod": 30}

    def _compute(
        self,
        series_by_field: dict[str, pd.Series],
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        close = series_by_field["close"]
        window = int(self.timeperiod)
        mean = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=1)
        z = (close - mean) / std
        return pd.DataFrame({"value": z.values}, index=index)


# ---------------------------------------------------------------------- signals


def _build_signal_matrix(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean-reversion: buy at z < -1, sell at z > +1. Per asset, boolean panel."""
    closes = bars.xs("close", axis=1, level=0)
    spec = ZScore(timeperiod=20)
    entries = pd.DataFrame(index=closes.index, columns=closes.columns, dtype=bool).fillna(False)
    exits = entries.copy()
    for asset in closes.columns:
        z = spec._compute({"close": closes[asset]}, closes.index)["value"]
        entries[asset] = z < -1.0
        exits[asset] = z > 1.0
    return entries, exits


# ----------------------------------------------------------------------- orders


def _build_orders_log(bars: pd.DataFrame) -> pd.DataFrame:
    """A quarterly rebalance encoded as an explicit ts/asset/side/qty frame."""
    closes = bars.xs("close", axis=1, level=0)
    q_ends = closes.resample("QE").last().index
    # Target a constant dollar exposure of $50,000 per asset on each quarter.
    rows: list[dict[str, object]] = []
    for ts in q_ends:
        if ts not in closes.index:
            continue
        for asset in closes.columns:
            px = float(closes.loc[ts, asset])
            if px <= 0:
                continue
            qty = 50_000.0 / px
            rows.append({"ts": ts, "asset": asset, "side": "buy", "qty": qty})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------- run


def main() -> int:
    closes = pull_closes(["SPY", "QQQ", "IWM"], years=3)
    if closes is None or closes.empty:
        return 1

    # `bars` wants OHLCV — but our signal only needs close. Build a Bars-shaped
    # frame (MultiIndex (field, symbol)) using close everywhere; _data doesn't
    # expose OHLC so we synthesise tight O/H/L around close.
    assets = list(closes.columns)
    idx = closes.index
    bars = pd.DataFrame(index=idx)
    for sym in assets:
        c = closes[sym]
        bars[("open", sym)] = c
        bars[("high", sym)] = c * 1.001
        bars[("low", sym)] = c * 0.999
        bars[("close", sym)] = c
        bars[("volume", sym)] = 1_000_000.0
    bars.columns = pd.MultiIndex.from_tuples(bars.columns)
    print(f"Live data: {idx[0].date()} → {idx[-1].date()}  "
          f"({len(idx)} bars, {len(assets)} assets)")

    # ----------------------- Run 1: signals-driven backtest ------------------
    print("\n--- Run 1 · run_signals (mean-reversion z-score) ---")
    entries, exits = _build_signal_matrix(bars)
    n_ent = int(entries.sum().sum())
    n_ex = int(exits.sum().sum())
    print(f"Entry flags: {n_ent}   Exit flags: {n_ex}")
    sim = Simulator(bars, cash=200_000.0, costs=FixedBps(5))
    sig_result = sim.run_signals(entries, exits, size=0.1)
    print(f"Trades executed:    {len(sig_result.trades)}")
    print(f"Final equity:       ${sig_result.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Sharpe:             {sig_result.portfolio.sharpe():>12.2f}")

    # ----------------------- Run 2: orders-driven backtest -------------------
    print("\n--- Run 2 · run_orders (quarterly $50K dollar-cost ladder) ---")
    orders_log = _build_orders_log(bars)
    orders_path = OUT / "20_orders_log.csv"
    orders_log.to_csv(orders_path, index=False)
    print(f"Orders log:         {orders_path.relative_to(HERE.parent)}  "
          f"({len(orders_log)} rows)")
    sim2 = Simulator(bars, cash=1_000_000.0, costs=FixedBps(3))
    ord_result = sim2.run_orders(orders_log)
    print(f"Trades executed:    {len(ord_result.trades)}")
    print(f"Final equity:       ${ord_result.equity_curve.iloc[-1]:>12,.0f}")
    print(f"Max drawdown:       {ord_result.portfolio.max_drawdown() * 100:>11.1f}%")

    print("\nHow to read it:")
    print("  * run_signals turns a screener's boolean matrix directly into a")
    print("    backtest — useful when your signal already lives in a notebook.")
    print("  * run_orders accepts the output of any rebalance engine, export")
    print("    tool, or broker API as the authoritative order log.")
    print("  * @register_indicator(...) makes a custom IndicatorSpec (here:")
    print("    rolling z-score) discoverable through the same mechanism as the")
    print("    158 TA-Lib wrappers — zero extra plumbing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
