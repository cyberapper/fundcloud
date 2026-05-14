"""Synthetic-fixture tests for ``fundcloud.metrics.feature_quality``.

Builds a tiny 2-asset bars frame where we can hand-calculate the metric
values, then asserts that ``evaluate`` produces them. Catches regressions
in:

* directional sign handling (bullish vs bearish events),
* the MFE / MAE non-negativity invariants,
* the asset-weighted baseline,
* the per-horizon ``n_events`` truncation rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fundcloud.features.patterns import EVENTS_COLUMNS, Direction, Pattern
from fundcloud.metrics import feature_quality as fq


def _build_bars(asset: str, n: int, *, drift_per_bar: float, seed: int) -> pd.DataFrame:
    """Synthetic OHLCV with a known per-bar drift and small noise."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift_per_bar, 0.005, size=n)))
    high = close * (1.0 + rng.uniform(0.001, 0.005, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.005, size=n))
    open_ = close.copy()
    volume = np.full(n, 1_000_000.0)
    index = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            ("open", asset): open_,
            ("high", asset): high,
            ("low", asset): low,
            ("close", asset): close,
            ("volume", asset): volume,
        },
        index=index,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _empty_event(asset: str, ts: pd.Timestamp, direction: Direction, entry: float) -> dict:
    """Construct a minimal event row matching ``EVENTS_COLUMNS``."""
    return {
        "pattern": Pattern.DOUBLE_TOP if direction is Direction.BEARISH else Pattern.DOUBLE_BOTTOM,
        "asset": asset,
        "direction": direction,
        "formation_start": ts - pd.Timedelta(days=10),
        "formation_end": ts,
        "breakout_ts": ts,
        "entry_price": entry,
        "breakout_price": entry,
        "target_price": float("nan"),
        "stop_price": float("nan"),
        "quality": 75.0,
        "variant": None,
        "pivots": [],
        "meta": {},
    }


def test_evaluate_bullish_event_in_uptrend_hits_mfe_positive() -> None:
    """A bullish event placed at a known bar in a strong uptrend should
    show hit_rate==1.0 at all horizons and a positive MFE in ATR units.
    """
    bars = _build_bars("AAA", n=300, drift_per_bar=0.005, seed=42)
    breakout_ts = bars.index[100]
    entry = float(bars[("close", "AAA")].iloc[100])
    events = pd.DataFrame(
        [_empty_event("AAA", breakout_ts, Direction.BULLISH, entry)],
        columns=EVENTS_COLUMNS,
    )

    panel = fq.evaluate(events, bars, horizons=(5, 10, 20))

    assert panel.loc[5, "n_events"] == 1
    # Strong uptrend, single event → every horizon "hits" the bullish move.
    assert panel.loc[5, "hit_rate"] == 1.0
    assert panel.loc[20, "hit_rate"] == 1.0
    # MFE strictly > 0 (favourable excursion exists) and is finite.
    assert panel.loc[20, "mfe_atr"] > 0
    assert np.isfinite(panel.loc[20, "mae_atr"])
    # mfe_atr non-negative invariant — guarded against the bug where a
    # bearish-style formula was applied to a bullish event.
    assert panel.loc[5, "mfe_atr"] >= 0
    # MAE p95 ≥ MAE mean — distribution sanity check.
    assert panel.loc[20, "mae_p95_atr"] >= panel.loc[20, "mae_atr"]


def test_evaluate_bearish_event_in_uptrend_hits_zero() -> None:
    """A bearish event in an uptrend should be wrong on every horizon —
    locks the directional-sign handling.
    """
    bars = _build_bars("BBB", n=300, drift_per_bar=0.005, seed=7)
    breakout_ts = bars.index[100]
    entry = float(bars[("close", "BBB")].iloc[100])
    events = pd.DataFrame(
        [_empty_event("BBB", breakout_ts, Direction.BEARISH, entry)],
        columns=EVENTS_COLUMNS,
    )

    # Grade these "bearish" events as shorts; the uptrend makes them losers.
    panel = fq.evaluate(events, bars, horizons=(5, 10, 20), trade_direction="short")

    assert panel.loc[5, "n_events"] == 1
    assert panel.loc[5, "hit_rate"] == 0.0
    assert panel.loc[20, "hit_rate"] == 0.0
    # Still required: MFE and MAE both ≥ 0 (max-and-min invariants).
    assert panel.loc[20, "mfe_atr"] >= 0
    assert panel.loc[20, "mae_atr"] >= 0


def test_evaluate_baseline_reflects_asset_drift() -> None:
    """Baseline for a bullish event in a heavy uptrend should be > 0.5;
    for a bearish event in the same series, < 0.5.
    """
    bars = _build_bars("CCC", n=400, drift_per_bar=0.008, seed=1)
    ts = bars.index[100]
    entry = float(bars[("close", "CCC")].iloc[100])
    bullish = pd.DataFrame(
        [_empty_event("CCC", ts, Direction.BULLISH, entry)], columns=EVENTS_COLUMNS
    )
    bearish = pd.DataFrame(
        [_empty_event("CCC", ts, Direction.BEARISH, entry)], columns=EVENTS_COLUMNS
    )

    # Pass trade_direction explicitly — long for bullish, short for bearish.
    # The baseline-mirror invariant still holds.
    bull_panel = fq.evaluate(bullish, bars, horizons=(20,), trade_direction="long")
    bear_panel = fq.evaluate(bearish, bars, horizons=(20,), trade_direction="short")

    # Heavy uptrend → most random 20-bar windows close higher.
    assert bull_panel.loc[20, "baseline_hit"] > 0.6
    # Mirror: bearish baseline should be roughly 1 - bullish baseline.
    assert bear_panel.loc[20, "baseline_hit"] < 0.4
    # Bullish + bearish baselines on the same series sum to ~1 (modulo
    # zero-return bars, which are vanishingly rare with continuous noise).
    total = bull_panel.loc[20, "baseline_hit"] + bear_panel.loc[20, "baseline_hit"]
    assert abs(total - 1.0) < 0.05


def test_evaluate_drops_horizons_beyond_lookahead() -> None:
    """An event near the end of the bar series should be retained for
    short horizons but dropped for ones that exceed available lookahead.
    """
    bars = _build_bars("DDD", n=120, drift_per_bar=0.001, seed=11)
    # Place the event 15 bars from the end — h=10 fits, h=60 doesn't.
    ts = bars.index[-15]
    entry = float(bars[("close", "DDD")].iloc[-15])
    events = pd.DataFrame(
        [_empty_event("DDD", ts, Direction.BULLISH, entry)], columns=EVENTS_COLUMNS
    )

    panel = fq.evaluate(events, bars, horizons=(5, 10, 60))

    assert panel.loc[5, "n_events"] == 1
    assert panel.loc[10, "n_events"] == 1
    assert panel.loc[60, "n_events"] == 0
    # No-data horizon → all metrics NaN, n_events 0.
    assert np.isnan(panel.loc[60, "hit_rate"])
    assert np.isnan(panel.loc[60, "expectancy"])


def test_evaluate_empty_events_returns_zero_row_panel() -> None:
    """Empty events frame → zero-event rows with NaN metrics; doesn't crash."""
    bars = _build_bars("EEE", n=200, drift_per_bar=0.001, seed=99)
    empty = pd.DataFrame(columns=EVENTS_COLUMNS)
    panel = fq.evaluate(empty, bars, horizons=(5, 10, 20))
    assert (panel["n_events"] == 0).all()
    assert panel["hit_rate"].isna().all()
    assert panel["expectancy"].isna().all()


def test_evaluate_long_vs_short_are_signed_mirrors() -> None:
    """Grading the same event set as `long` vs `short` should produce
    hit-rates and expectancies that mirror each other — baseline included.
    """
    bars = _build_bars("FFF", n=300, drift_per_bar=0.005, seed=23)
    ts = bars.index[100]
    entry = float(bars[("close", "FFF")].iloc[100])
    events = pd.DataFrame(
        [_empty_event("FFF", ts, Direction.BEARISH, entry)], columns=EVENTS_COLUMNS
    )

    short = fq.evaluate(events, bars, horizons=(20,), trade_direction="short")
    long_ = fq.evaluate(events, bars, horizons=(20,), trade_direction="long")

    # Single event in an uptrend — graded short, hit_rate=0; graded long, hit_rate=1.
    assert short.loc[20, "hit_rate"] == 0.0
    assert long_.loc[20, "hit_rate"] == 1.0
    # Baselines flip in lockstep — sum ≈ 1.
    total = short.loc[20, "baseline_hit"] + long_.loc[20, "baseline_hit"]
    assert abs(total - 1.0) < 0.05
    # Expectancies are exact mirrors.
    assert abs(short.loc[20, "expectancy"] + long_.loc[20, "expectancy"]) < 1e-9


def test_evaluate_rejects_unknown_trade_direction() -> None:
    bars = _build_bars("GGG", n=100, drift_per_bar=0.001, seed=5)
    events = pd.DataFrame(
        [_empty_event("GGG", bars.index[50], Direction.BULLISH, 100.0)],
        columns=EVENTS_COLUMNS,
    )
    import pytest

    with pytest.raises(ValueError, match="unknown trade_direction"):
        fq.evaluate(events, bars, horizons=(20,), trade_direction="sideways")


def test_quality_buckets_monotonic_when_quality_drives_outcome() -> None:
    """Synthetic events whose forward returns scale with their quality —
    high quality → high return — should produce monotonically increasing
    metrics across buckets. Locks bucketing + per-bucket aggregation.
    """
    bars = _build_bars("HHH", n=600, drift_per_bar=0.0, seed=2)
    # Place 50 bullish events. Quality = signed 20-bar forward close
    # return so high quality → bigger upside realised. Expectancy is
    # close-based and signed, so it should sort monotonically across
    # buckets even though MFE has too much noise to test directly.
    rng = np.random.default_rng(0)
    ev_indices = rng.choice(np.arange(50, 500), size=50, replace=False)
    rows = []
    close = bars[("close", "HHH")]
    for idx in ev_indices:
        signed_fwd = float(close.iloc[idx + 20]) - float(close.iloc[idx])
        ev = _empty_event("HHH", bars.index[idx], Direction.BULLISH, float(close.iloc[idx]))
        ev["quality"] = float(signed_fwd * 100)
        rows.append(ev)
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)

    bk = fq.quality_buckets(events, bars, horizon=20, n_buckets=5)

    # 5 buckets, 10 events each, all populated.
    assert (bk["n_events"] == 10).all()
    # Expectancy is signed close-return / stop_distance — quality was
    # built from signed forward return, so Q5 (high quality) should have
    # strictly higher expectancy than Q1.
    exp_vals = bk["expectancy"].to_numpy()
    assert exp_vals[-1] > exp_vals[0], (
        f"Q5 expectancy ({exp_vals[-1]:.3f}) should exceed Q1 ({exp_vals[0]:.3f})"
    )


def test_quality_buckets_handles_empty_events() -> None:
    bars = _build_bars("III", n=200, drift_per_bar=0.001, seed=11)
    empty = pd.DataFrame(columns=EVENTS_COLUMNS)
    bk = fq.quality_buckets(empty, bars, horizon=20, n_buckets=5)
    assert (bk["n_events"] == 0).all()
    assert bk["hit_rate"].isna().all()


def test_per_asset_separates_by_asset() -> None:
    """Build two assets with very different drift; assert their per-asset
    rows differ in hit_rate accordingly.
    """
    df_a = _build_bars("AAA", n=300, drift_per_bar=0.008, seed=1)
    df_b = _build_bars("BBB", n=300, drift_per_bar=-0.008, seed=2)
    bars = pd.concat([df_a, df_b], axis=1)
    rows = [
        _empty_event(
            "AAA", df_a.index[100], Direction.BULLISH, float(df_a[("close", "AAA")].iloc[100])
        ),
        _empty_event(
            "BBB", df_b.index[100], Direction.BULLISH, float(df_b[("close", "BBB")].iloc[100])
        ),
    ]
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)

    pa = fq.per_asset(events, bars, horizon=20)

    assert sorted(pa.index.tolist()) == ["AAA", "BBB"]
    # AAA is in heavy uptrend → bullish hit, BBB downtrend → bullish miss.
    assert pa.loc["AAA", "hit_rate"] == 1.0
    assert pa.loc["BBB", "hit_rate"] == 0.0


def test_time_stability_assigns_events_to_chronological_folds() -> None:
    """Events spread across the bar series should land in different folds
    in chronological order.
    """
    bars = _build_bars("JJJ", n=500, drift_per_bar=0.001, seed=33)
    ev_bars = [50, 150, 250, 350, 450]
    rows = [
        _empty_event("JJJ", bars.index[b], Direction.BULLISH, float(bars[("close", "JJJ")].iloc[b]))
        for b in ev_bars
    ]
    events = pd.DataFrame(rows, columns=EVENTS_COLUMNS)

    ts = fq.time_stability(events, bars, horizon=10, n_folds=5)

    assert len(ts) == 5
    assert (ts["n_events"] == 1).all()
    # Folds should be chronologically ordered.
    starts = pd.to_datetime(ts["start"]).tolist()
    assert starts == sorted(starts)
