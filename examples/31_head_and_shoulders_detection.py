"""31 — Head & Shoulders chart-pattern detection.

End-to-end walkthrough of the new chart-pattern feature surface.

What this script demonstrates:

1. Building a multi-asset OHLCV ``Bars`` frame (SPY = textbook H&S, AGG = noise).
2. Running ``HeadAndShoulders().fit_transform(bars)`` to get a per-bar
   signal panel (``one column per asset``).
3. Pulling the rich event log via ``indicator.events(bars)`` — the
   canonical 14-column schema every detector emits.
4. Switching the projection mode (BREAKOUT vs FORMATION vs DECAY) without
   re-running the scan.
5. Overriding the ``PatternCondition`` (intuitive presets, user-tweakable).
6. Tightening ``min_quality`` to filter borderline detections.
7. Composing the pattern indicator inside a ``FeaturePipeline``.

Run:
    uv run python examples/31_head_and_shoulders_detection.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.features import FeaturePipeline
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    HeadAndShoulders,
    Pattern,
    PatternCondition,
    SignalMode,
)


# --------------------------------------------------------------------------- data


def _interp(anchors: list[tuple[int, float]], n: int) -> np.ndarray:
    """Piecewise-linear close path passing through ``(bar, price)`` anchors.

    Anchors must be in ascending bar order. The first anchor's bar must be
    0 and the last must be ``n - 1``.
    """
    out = np.empty(n, dtype=np.float64)
    for (a_bar, a_p), (b_bar, b_p) in zip(anchors, anchors[1:], strict=False):
        for k in range(a_bar, b_bar + 1):
            t = (k - a_bar) / (b_bar - a_bar)
            out[k] = a_p + t * (b_p - a_p)
    return out


def _build_panel() -> pd.DataFrame:
    """Two-asset ``Bars`` frame.

    SPY: 60 bars — 30-bar uptrend (90→100), then a 21-bar Head & Shoulders
    formation (anchors at bars 30, 35, 40, 45, 50 with prices 100, 92,
    110, 92, 100), then 10 bars of post-breakout decline.

    AGG: 60 bars of low-volatility random walk — should produce no
    detection at default thresholds.
    """
    n = 60
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=42)

    spy_anchors = [
        (0, 90.0),
        (30, 100.0),
        (35, 92.0),
        (40, 110.0),
        (45, 92.0),
        (50, 100.0),
        (59, 80.0),  # post-breakout fall
    ]
    spy_close = _interp(spy_anchors, n) + rng.normal(0, 0.15, n)
    spy_high = spy_close + rng.uniform(0.3, 0.6, n)
    spy_low = spy_close - rng.uniform(0.3, 0.6, n)
    spy_open = spy_close + rng.normal(0, 0.05, n)
    spy_volume = rng.uniform(800, 1200, n)

    agg_close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    agg_high = agg_close + rng.uniform(0.05, 0.15, n)
    agg_low = agg_close - rng.uniform(0.05, 0.15, n)
    agg_open = agg_close + rng.normal(0, 0.02, n)
    agg_volume = rng.uniform(800, 1200, n)

    columns = pd.MultiIndex.from_product(
        [["open", "high", "low", "close", "volume"], ["SPY", "AGG"]],
        names=["field", "asset"],
    )
    data = np.column_stack([
        spy_open, agg_open,
        spy_high, agg_high,
        spy_low, agg_low,
        spy_close, agg_close,
        spy_volume, agg_volume,
    ])
    return pd.DataFrame(data, index=index, columns=columns)


# ------------------------------------------------------------------ formatting


def _print_section(title: str) -> None:
    print(f"\n{'─' * 8} {title} {'─' * 8}")


# --------------------------------------------------------------------------- main


def main() -> None:
    bars = _build_panel()

    _print_section("1. Bars frame")
    print(f"shape           : {bars.shape}  (T={bars.shape[0]}, fields×assets={bars.shape[1]})")
    print(f"index           : {bars.index[0].date()} → {bars.index[-1].date()}  (tz={bars.index.tz})")
    print(f"assets          : {sorted(set(bars.columns.get_level_values('asset')))}")
    print(f"fields per asset: {sorted(set(bars.columns.get_level_values('field')))}")

    _print_section("2. fit_transform → per-bar signal panel")
    indicator = HeadAndShoulders()
    print(f"indicator       : {type(indicator).__name__}")
    print(f"pattern enum    : {Pattern.HEAD_AND_SHOULDERS}")
    print(f"min_quality     : {indicator.min_quality}")
    print(f"pivot_orders    : {indicator.pivot_orders}")
    print(f"signal_mode     : {indicator.signal_mode}")

    signals = indicator.fit_transform(bars)
    print(f"\nsignals.shape   : {signals.shape}  → one column per asset")
    print(f"signals.dtypes  : {dict(signals.dtypes)}")
    nonzero = signals[(signals != 0).any(axis=1)]
    if not nonzero.empty:
        print("\nNon-zero signal rows:")
        print(nonzero)
    else:
        print("\n(no signal fired — try raising min_quality or check input)")

    _print_section("3. events(bars) → canonical events table")
    events = indicator.events(bars)
    assert tuple(events.columns) == EVENTS_COLUMNS, "events schema drifted"
    print(f"events shape    : {events.shape}")
    print(f"columns         : {list(events.columns)}")
    if not events.empty:
        ev = events.iloc[0]
        print(f"\nFirst event ({ev['asset']}):")
        print(f"  pattern        : {ev['pattern']}")
        print(f"  direction      : {ev['direction']}")
        print(f"  formation      : {ev['formation_start'].date()} → {ev['formation_end'].date()}")
        print(f"  breakout_ts    : {ev['breakout_ts'].date()}")
        print(f"  entry_price    : {ev['entry_price']:.2f}")
        print(f"  breakout_price : {ev['breakout_price']:.2f}")
        print(f"  quality        : {ev['quality']:.1f} / 100")
        print(f"  variant        : {ev['variant']}")
        print(f"  pivots         : {len(ev['pivots'])}")
        for piv in ev["pivots"]:
            print(f"    {piv['ts'].date()} {piv['kind']:<5} @ {piv['price']:.2f}")
        print(f"  meta keys      : {sorted(ev['meta'].keys())}")
        print(f"  features       : {ev['meta']['features']}")

    _print_section("4. SignalMode swap (FORMATION marks the whole window)")
    formation_signals = HeadAndShoulders(signal_mode=SignalMode.FORMATION).fit_transform(bars)
    formation_nonzero = (formation_signals != 0).sum().to_dict()
    print(f"non-zero rows per asset: {formation_nonzero}")

    _print_section("5. PatternCondition override")
    custom = HeadAndShoulders(
        condition=PatternCondition().override(
            entry_rule="on_pullback",  # accepts str OR EntryRule.ON_PULLBACK
            time_stop_bars=15,
        )
    )
    print(f"effective condition: {custom.effective_condition}")

    _print_section("6. min_quality filter")
    strict = HeadAndShoulders(min_quality=95.0).fit_transform(bars)
    print(f"min_quality=95 → SPY non-zero rows: {int((strict['SPY'] != 0).sum())}")
    permissive = HeadAndShoulders(min_quality=0.0).fit_transform(bars)
    print(f"min_quality=0  → SPY non-zero rows: {int((permissive['SPY'] != 0).sum())}")

    _print_section("7. FeaturePipeline composition")
    pipe = FeaturePipeline([("hns", HeadAndShoulders())])
    panel = pipe.fit_transform(bars)
    print(f"pipeline output shape: {panel.shape}")
    print(f"pipeline_hash         : {pipe.pipeline_hash}")

    _print_section("Summary")
    expected_breakout_idx = 50  # right shoulder anchor
    spy_signal = signals["SPY"]
    if spy_signal.iloc[expected_breakout_idx] > 0:
        print(f"✓ Breakout signal fired at bar {expected_breakout_idx} "
              f"({bars.index[expected_breakout_idx].date()}) — as expected.")
    else:
        print(f"✗ Expected SPY breakout at bar {expected_breakout_idx} but got "
              f"{spy_signal.iloc[expected_breakout_idx]:.2f}")
    if (signals["AGG"] == 0).all():
        print("✓ AGG noise produced 0 detections — false-positive guard working.")
    else:
        print(f"✗ AGG had {int((signals['AGG'] != 0).sum())} unexpected hits")


if __name__ == "__main__":
    main()
