"""21 — Cost, slippage, and execution-model lab.

Friction matters. Fundcloud's ``Simulator`` accepts three orthogonal knobs:

* ``costs`` — one of :class:`NoCost`, :class:`FixedBps` (default 5 bps),
  :class:`PerShare` (broker-style per-share commission).
* ``slippage`` — :class:`NoSlippage` or :class:`HalfSpread` (assumes you pay
  half the bid/ask gap on each fill).
* ``execution`` — :class:`NextBarOpen` (default, realistic — fills hit the
  next bar's open price) or :class:`SameBarClose` (optimistic — fills at
  the signal bar's close).

This example runs the **same weekly-DCA strategy** under six configurations
so you can see how each knob eats into realised return.

Run:
    uv run python examples/21_cost_model_lab.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
from _synth import AssetProfile, generate_ohlcv
from fundcloud.sim import (
    FixedBps,
    HalfSpread,
    NoCost,
    NoSlippage,
    PerShare,
    SameBarClose,
    Simulator,
)
from fundcloud.strategies import DCA

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _simulate(bars: pd.DataFrame, make_sim: Callable[[pd.DataFrame], Simulator]) -> dict:
    sim = make_sim(bars)
    result = sim.run_strategy(DCA(amount=500.0, horizon="weekly", weights={"SPY": 1.0}))
    fees = float(result.trades["fee"].sum()) if "fee" in result.trades.columns else 0.0
    invested = float((result.trades["qty"] * result.trades["price"]).sum())
    equity = float(result.equity_curve.iloc[-1])
    return {
        "end_equity": equity,
        "invested": invested,
        "fees": fees,
        "trades": len(result.trades),
        "sharpe": float(result.portfolio.sharpe()),
    }


def main() -> int:
    bars = generate_ohlcv(
        {"SPY": AssetProfile(mu=0.08, sigma=0.16, price0=450.0)},
        start="2022-01-03",
        periods=756,  # three years
        seed=9,
    )
    configs: list[tuple[str, Callable[[pd.DataFrame], Simulator]]] = [
        (
            "NoCost · NoSlippage · NextBarOpen",
            lambda b: Simulator(b, cash=100_000.0, costs=NoCost(), slippage=NoSlippage()),
        ),
        (
            "FixedBps(5) · NoSlippage · NextBarOpen",
            lambda b: Simulator(b, cash=100_000.0, costs=FixedBps(5), slippage=NoSlippage()),
        ),
        (
            "FixedBps(25) · NoSlippage · NextBarOpen",
            lambda b: Simulator(b, cash=100_000.0, costs=FixedBps(25), slippage=NoSlippage()),
        ),
        (
            "FixedBps(5) · HalfSpread(10bps) · NextBarOpen",
            lambda b: Simulator(
                b,
                cash=100_000.0,
                costs=FixedBps(5),
                slippage=HalfSpread(spread_bps=10.0),
            ),
        ),
        (
            "PerShare($0.005, $1 min) · NoSlippage · NextBarOpen",
            lambda b: Simulator(
                b,
                cash=100_000.0,
                costs=PerShare(rate=0.005, minimum=1.0),
                slippage=NoSlippage(),
            ),
        ),
        (
            "FixedBps(5) · NoSlippage · SameBarClose",
            lambda b: Simulator(
                b,
                cash=100_000.0,
                costs=FixedBps(5),
                slippage=NoSlippage(),
                execution=SameBarClose(),
            ),
        ),
    ]

    rows = []
    for label, make in configs:
        stats = _simulate(bars, make)
        rows.append({"config": label, **stats})

    df = pd.DataFrame(rows).set_index("config")
    df["drag_vs_free"] = (
        df.loc["NoCost · NoSlippage · NextBarOpen", "end_equity"] - df["end_equity"]
    )
    df["drag_bps"] = df["drag_vs_free"] / df["invested"] * 10_000
    print("\nCost-lab results (same DCA, different friction):\n")
    print(
        df[["end_equity", "fees", "trades", "drag_vs_free", "drag_bps"]].to_string(
            float_format=lambda v: f"{v:>11,.2f}",
        )
    )

    path = OUT / "21_cost_lab.csv"
    df.to_csv(path)
    print(f"\nSaved:  {path.relative_to(HERE.parent)}")

    print("\nHow to read it:")
    print("  * The 'NoCost · NoSlippage · NextBarOpen' row is the frictionless")
    print("    benchmark. Every other row's drag is the price of realism.")
    print("  * FixedBps is linear in notional — the 25bps row bleeds ~5x the")
    print("    5bps row, all else equal.")
    print("  * HalfSpread(10bps) is similar-scale friction but attacks fill-price")
    print("    rather than commission; under a realistic bid/ask it adds on top.")
    print("  * PerShare matters most at small dollar sizes (the $1 minimum bites")
    print("    when you're only buying a couple of shares per weekly buy).")
    print("  * SameBarClose is optimistic — same signal bar, same bar's close —")
    print("    and typically prints a slightly rosier equity curve than the")
    print("    realistic NextBarOpen default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
