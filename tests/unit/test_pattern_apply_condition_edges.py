"""Edge-case coverage for ``apply_condition`` and its private helpers.

The happy-path geometry tests live in ``test_pattern_condition.py``; this
file fills the degenerate-input branches (NaN inputs, missing pivots,
unknown assets, ATR-failure fallbacks, etc.) under the post-FLG-1015
schema where detection is direction-agnostic and the events frame
carries unsigned ``breakout_level`` / ``formation_height`` instead of
the old signed ``entry_price`` / ``breakout_price`` / ``direction``
trio.

We also pin the ``PatternCondition.override`` validation paths because
they share the same enum/coercion surface and were under-covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.features.patterns import (
    EVENTS_COLUMNS,
    Direction,
    EntryRule,
    ExitRule,
    Pattern,
    PatternCondition,
    StopMethod,
    TargetMethod,
    apply_condition,
)
from fundcloud.features.patterns._apply_condition import (
    _direction_sign,
    _pivot_prices,
    _resolve_event_direction,
    _resolve_height,
    _resolve_stop,
    _resolve_target,
    _select_asset,
    _wilder_atr,
)


def _bars(n: int = 100, asset: str = "AAA") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.05, 0.5, size=n))
    df = pd.DataFrame(
        {
            ("open", asset): close,
            ("high", asset): close + 0.5,
            ("low", asset): close - 0.5,
            ("close", asset): close,
            ("volume", asset): np.full(n, 1.0e6),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "asset"])
    return df


def _event_row(
    *,
    asset: str,
    formation_start: pd.Timestamp,
    breakout_ts: pd.Timestamp,
    breakout_level: float | None,
    formation_height: float | None,
    pivots: list[dict],
    pattern: Pattern = Pattern.DOUBLE_BOTTOM,
) -> dict:
    return {
        "pattern": pattern,
        "asset": asset,
        "formation_start": formation_start,
        "formation_end": breakout_ts,
        "breakout_ts": breakout_ts,
        "breakout_level": breakout_level,
        "formation_height": formation_height,
        "target_price": float("nan"),
        "stop_price": float("nan"),
        "quality": 50.0,
        "variant": None,
        "pivots": pivots,
        "meta": {},
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestWilderATR:
    def test_empty_input_returns_empty(self) -> None:
        out = _wilder_atr(np.array([]), np.array([]), np.array([]), 14)
        assert out.shape == (0,)

    def test_invalid_window_returns_all_nan(self) -> None:
        out = _wilder_atr(np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.7, 1.5]), 0)
        assert np.isnan(out).all()

    def test_short_input_returns_all_nan(self) -> None:
        out = _wilder_atr(np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.7, 1.5]), 14)
        assert np.isnan(out).all()

    def test_single_bar_skips_recursion_branch(self) -> None:
        out = _wilder_atr(np.array([2.0]), np.array([1.0]), np.array([1.5]), 1)
        assert out[0] == pytest.approx(1.0)


class TestSelectAsset:
    def test_rejects_flat_columns(self) -> None:
        flat = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
        with pytest.raises(TypeError, match="MultiIndex"):
            _select_asset(flat, "AAA")


class TestDirectionSign:
    def test_bullish_is_long(self) -> None:
        assert _direction_sign(Direction.BULLISH) == 1

    def test_bearish_is_short(self) -> None:
        assert _direction_sign(Direction.BEARISH) == -1


class TestResolveEventDirection:
    def test_no_map_returns_default(self) -> None:
        out = _resolve_event_direction(
            Pattern.DOUBLE_TOP, default=Direction.BULLISH, direction_map=None
        )
        assert out is Direction.BULLISH

    def test_enum_key_lookup(self) -> None:
        dmap = {Pattern.DOUBLE_TOP: Direction.BEARISH}
        out = _resolve_event_direction(
            Pattern.DOUBLE_TOP, default=Direction.BULLISH, direction_map=dmap
        )
        assert out is Direction.BEARISH

    def test_string_key_lookup(self) -> None:
        dmap = {"double_top": Direction.BEARISH}
        out = _resolve_event_direction(
            Pattern.DOUBLE_TOP, default=Direction.BULLISH, direction_map=dmap
        )
        assert out is Direction.BEARISH

    def test_missing_pattern_falls_back_to_default(self) -> None:
        dmap = {Pattern.HEAD_AND_SHOULDERS: Direction.BEARISH}
        out = _resolve_event_direction(
            Pattern.DOUBLE_TOP, default=Direction.BULLISH, direction_map=dmap
        )
        assert out is Direction.BULLISH


class TestPivotPrices:
    def test_skips_none_price(self) -> None:
        pivots = [{"kind": "LOW", "price": None}, {"kind": "LOW", "price": 1.5}]
        assert _pivot_prices(pivots, "LOW") == [1.5]

    def test_skips_non_numeric_price(self) -> None:
        pivots = [{"kind": "LOW", "price": "abc"}, {"kind": "LOW", "price": 2.0}]
        assert _pivot_prices(pivots, "LOW") == [2.0]

    def test_skips_non_finite_price(self) -> None:
        pivots = [
            {"kind": "LOW", "price": float("inf")},
            {"kind": "LOW", "price": float("nan")},
            {"kind": "LOW", "price": 3.0},
        ]
        assert _pivot_prices(pivots, "LOW") == [3.0]


class TestResolveHeight:
    def test_finite_positive_passthrough(self) -> None:
        assert _resolve_height(5.0, fallback=2.0) == 5.0

    def test_none_falls_back(self) -> None:
        assert _resolve_height(None, fallback=2.0) == 2.0

    def test_nan_falls_back(self) -> None:
        assert _resolve_height(float("nan"), fallback=2.0) == 2.0

    def test_zero_falls_back(self) -> None:
        assert _resolve_height(0.0, fallback=2.0) == 2.0

    def test_negative_falls_back(self) -> None:
        # Detector contract says non-negative; defensive check anyway.
        assert _resolve_height(-1.5, fallback=2.0) == 2.0

    def test_non_numeric_falls_back(self) -> None:
        assert _resolve_height("abc", fallback=2.0) == 2.0


class TestResolveTargetAndStopErrors:
    def test_resolve_target_fixed_atr(self) -> None:
        out = _resolve_target(
            entry=100.0,
            sign=1,
            pattern_height=5.0,
            atr=2.0,
            method=TargetMethod.FIXED_ATR,
            atr_multiple=3.0,
            fib_multiple=1.618,
        )
        assert out == pytest.approx(106.0)

    def test_resolve_target_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported target method"):
            _resolve_target(
                entry=100.0,
                sign=1,
                pattern_height=5.0,
                atr=2.0,
                method="bogus",  # type: ignore[arg-type]
                atr_multiple=3.0,
                fib_multiple=1.618,
            )

    def test_resolve_stop_below_pivot_bearish_uses_high(self) -> None:
        out = _resolve_stop(
            entry=100.0,
            sign=-1,
            pivots=[{"kind": "HIGH", "price": 105.0}],
            atr=1.0,
            method=StopMethod.BELOW_PIVOT,
            atr_multiple=2.0,
            fixed_pct=0.05,
        )
        assert out == 105.0

    def test_resolve_stop_below_pivot_falls_back_when_no_pivots(self) -> None:
        out_bull = _resolve_stop(
            entry=100.0,
            sign=1,
            pivots=[],
            atr=1.0,
            method=StopMethod.BELOW_PIVOT,
            atr_multiple=2.0,
            fixed_pct=0.05,
        )
        assert out_bull == pytest.approx(98.0)
        out_bear = _resolve_stop(
            entry=100.0,
            sign=-1,
            pivots=[],
            atr=1.0,
            method=StopMethod.BELOW_PIVOT,
            atr_multiple=2.0,
            fixed_pct=0.05,
        )
        assert out_bear == pytest.approx(102.0)

    def test_resolve_stop_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported stop method"):
            _resolve_stop(
                entry=100.0,
                sign=1,
                pivots=[],
                atr=1.0,
                method="bogus",  # type: ignore[arg-type]
                atr_multiple=2.0,
                fixed_pct=0.05,
            )


# ---------------------------------------------------------------------------
# apply_condition() per-row degenerate paths
# ---------------------------------------------------------------------------


class TestApplyConditionDegenerateRows:
    def test_unknown_asset_yields_nan(self) -> None:
        bars = _bars()
        ts = bars.index[50]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="ZZZ",  # not present in bars
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                ),
                # second row hits the cached `None` branch.
                _event_row(
                    asset="ZZZ",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                ),
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert out["target_price"].isna().all()
        assert out["stop_price"].isna().all()

    def test_missing_breakout_ts_yields_nan(self) -> None:
        bars = _bars()
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=pd.NaT,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_breakout_ts_off_grid_yields_nan(self) -> None:
        bars = _bars()
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=pd.Timestamp("1999-01-01", tz="UTC"),
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert np.isnan(out.loc[0, "target_price"])

    def test_atr_invalid_with_atr_relative_method_yields_nan(self) -> None:
        bars = _bars(n=20)
        ts = bars.index[2]  # too early for window=14 ATR to be defined
        fs = bars.index[0]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        cond = PatternCondition().override(target_method=TargetMethod.FIXED_ATR)
        out = apply_condition(events, cond, bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_no_breakout_level_yields_nan(self) -> None:
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=None,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert np.isnan(out.loc[0, "target_price"])

    def test_zero_height_falls_back_to_atr(self) -> None:
        # formation_height = 0 → falls back to ATR. With a fully-warm ATR
        # window, target/stop come out finite.
        bars = _bars(n=60)
        ts = bars.index[40]
        fs = bars.index[35]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=0.0,
                    pivots=[],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert np.isfinite(out.loc[0, "target_price"])
        assert np.isfinite(out.loc[0, "stop_price"])

    def test_invalid_height_at_cold_atr_yields_nan(self) -> None:
        # formation_height = 0 with an ATR window not yet warm → nothing
        # to fall back to → NaN.
        bars = _bars(n=10)
        ts = bars.index[2]
        fs = bars.index[0]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=0.0,
                    pivots=[],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_direction_kwarg_flips_sign(self) -> None:
        """Same event computed long vs. short produces mirrored target/stop."""
        bars = _bars(n=80)
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        long_out = apply_condition(events, PatternCondition(), bars, direction=Direction.BULLISH)
        short_out = apply_condition(events, PatternCondition(), bars, direction=Direction.BEARISH)
        # MEASURED_MOVE: long target = entry + h, short target = entry - h.
        assert long_out.loc[0, "target_price"] == pytest.approx(105.0)
        assert short_out.loc[0, "target_price"] == pytest.approx(95.0)

    def test_direction_map_overrides_default(self) -> None:
        bars = _bars(n=80)
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    formation_start=fs,
                    breakout_ts=ts,
                    breakout_level=100.0,
                    formation_height=5.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                    pattern=Pattern.HEAD_AND_SHOULDERS,
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        # Default LONG, but the map flips H&S to SHORT.
        out = apply_condition(
            events,
            PatternCondition(),
            bars,
            direction=Direction.BULLISH,
            direction_map={Pattern.HEAD_AND_SHOULDERS: Direction.BEARISH},
        )
        assert out.loc[0, "target_price"] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# PatternCondition.override edge cases
# ---------------------------------------------------------------------------


class TestPatternConditionOverride:
    def test_unknown_field_raises_with_valid_list(self) -> None:
        cond = PatternCondition()
        with pytest.raises(TypeError, match="unknown PatternCondition fields"):
            cond.override(nonsense=1)

    def test_string_coercion_for_entry_and_exit_rules(self) -> None:
        cond = PatternCondition().override(
            entry_rule="on_pullback",
            exit_rule="time_stop",
        )
        assert cond.entry_rule is EntryRule.ON_PULLBACK
        assert cond.exit_rule is ExitRule.TIME_STOP
