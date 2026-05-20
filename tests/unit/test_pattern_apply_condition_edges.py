"""Edge-case coverage for ``apply_condition`` and its private helpers.

The happy-path geometry tests live in ``test_pattern_condition.py``; this
file fills the degenerate-input branches that codecov flagged as
uncovered on the FLG-1015 PR (NaN inputs, missing pivots, unknown
assets, ATR-failure fallbacks, NEUTRAL direction, etc.).

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
    _pattern_height,
    _pivot_prices,
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


# Maps each Direction to a pattern whose classical shape carries that
# direction — used to construct test events when a test needs a
# bullish / bearish / neutral example.
_PATTERN_FOR_DIRECTION = {
    Direction.BULLISH: Pattern.DOUBLE_BOTTOM,
    Direction.BEARISH: Pattern.DOUBLE_TOP,
    Direction.NEUTRAL: Pattern.SYMMETRICAL_TRIANGLE,
}


def _event_row(
    *,
    asset: str,
    direction: Direction,
    formation_start: pd.Timestamp,
    breakout_ts: pd.Timestamp,
    entry_price: float | None,
    breakout_price: float | None,
    pivots: list[dict],
) -> dict:
    return {
        "pattern": _PATTERN_FOR_DIRECTION[direction],
        "asset": asset,
        "formation_start": formation_start,
        "formation_end": breakout_ts,
        "breakout_ts": breakout_ts,
        "entry_price": entry_price,
        "breakout_price": breakout_price,
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
        # n < window → atr stays NaN.
        out = _wilder_atr(np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.7, 1.5]), 14)
        assert np.isnan(out).all()

    def test_single_bar_skips_recursion_branch(self) -> None:
        out = _wilder_atr(np.array([2.0]), np.array([1.0]), np.array([1.5]), 1)
        # Only TR[0] = high - low = 1.0 → ATR[0] = mean(TR[:1]) = 1.0.
        assert out[0] == pytest.approx(1.0)


class TestSelectAsset:
    def test_rejects_flat_columns(self) -> None:
        flat = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
        with pytest.raises(TypeError, match="MultiIndex"):
            _select_asset(flat, "AAA")


class TestDirectionSign:
    def test_neutral_returns_zero(self) -> None:
        assert _direction_sign(Direction.NEUTRAL) == 0

    def test_string_aliases(self) -> None:
        assert _direction_sign("BULLISH") == 1
        assert _direction_sign("Bearish") == -1

    def test_unknown_string_returns_zero(self) -> None:
        assert _direction_sign("sideways") == 0


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


class TestPatternHeight:
    def test_empty_pivots_returns_fallback(self) -> None:
        assert _pattern_height([], 100.0, 1, fallback=2.0) == 2.0

    def test_neutral_sign_returns_fallback(self) -> None:
        pivots = [{"kind": "LOW", "price": 90.0}]
        assert _pattern_height(pivots, 100.0, 0, fallback=2.0) == 2.0

    def test_bullish_without_low_pivots_returns_fallback(self) -> None:
        pivots = [{"kind": "HIGH", "price": 110.0}]
        assert _pattern_height(pivots, 100.0, 1, fallback=2.0) == 2.0

    def test_bearish_without_high_pivots_returns_fallback(self) -> None:
        pivots = [{"kind": "LOW", "price": 90.0}]
        assert _pattern_height(pivots, 100.0, -1, fallback=2.0) == 2.0

    def test_zero_height_falls_back(self) -> None:
        # Bullish, but the only LOW equals entry → height = 0.
        pivots = [{"kind": "LOW", "price": 100.0}]
        assert _pattern_height(pivots, 100.0, 1, fallback=2.0) == 2.0


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
    def test_neutral_direction_yields_nan(self) -> None:
        bars = _bars()
        ts = bars.index[50]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.NEUTRAL,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        # The "neutral yields NaN" contract is expressed by passing
        # NEUTRAL on the condition.
        out = apply_condition(events, PatternCondition(direction=Direction.NEUTRAL), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_unknown_asset_yields_nan(self) -> None:
        bars = _bars()
        ts = bars.index[50]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="ZZZ",  # not present in bars
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                ),
                # second row hits the cached `None` branch (asset already
                # seen and stored as None).
                _event_row(
                    asset="ZZZ",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                ),
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        assert out["target_price"].isna().all()
        assert out["stop_price"].isna().all()

    def test_missing_breakout_ts_yields_nan(self) -> None:
        bars = _bars()
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=pd.NaT,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_breakout_ts_off_grid_yields_nan(self) -> None:
        bars = _bars()
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=pd.Timestamp("1999-01-01", tz="UTC"),
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        assert np.isnan(out.loc[0, "target_price"])

    def test_atr_invalid_with_atr_relative_method_yields_nan(self) -> None:
        # Use a tiny bar frame so ATR at the early breakout position is NaN.
        bars = _bars(n=20)
        ts = bars.index[2]  # too early for window=14 ATR to be defined
        fs = bars.index[0]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        cond = PatternCondition(direction=Direction.BULLISH).override(
            target_method=TargetMethod.FIXED_ATR
        )
        out = apply_condition(events, cond, bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_breakout_price_falls_back_to_entry_price(self) -> None:
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=None,  # forces entry_price fallback
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        # MEASURED_MOVE: target = 100 + (100 - 95) = 105.
        assert out.loc[0, "target_price"] == pytest.approx(105.0)

    def test_no_entry_price_at_all_yields_nan(self) -> None:
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=None,
                    breakout_price=None,
                    pivots=[{"kind": "LOW", "price": 95.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        assert np.isnan(out.loc[0, "target_price"])

    def test_invalid_height_after_pivots_yields_nan(self) -> None:
        # All LOW pivots == entry → height collapses to 0; with a NaN
        # fallback (very short ATR window not yet warm in the bar context
        # we'll target via the time_stop_bars-irrelevant condition), both
        # values fall to NaN.
        # Easier: stub the fallback path by giving entry == pivot AND a
        # very early breakout where ATR is NaN.
        bars = _bars(n=10)
        ts = bars.index[2]
        fs = bars.index[0]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=100.0,
                    breakout_price=100.0,
                    pivots=[{"kind": "LOW", "price": 100.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        # MEASURED_MOVE doesn't need ATR, so the early-NaN-ATR path falls
        # through; height = 0 (entry == pivot) → fallback ATR is NaN → NaN.
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_bearish_subcent_target_nan_when_height_exceeds_entry(self) -> None:
        # Sub-cent OTC quote with a split-unadjusted HIGH pivot: measured-move
        # target = 0.01 - (30 - 0.01) ≈ -29.99 → Guard B must NaN both legs.
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BEARISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=0.01,
                    breakout_price=0.01,
                    pivots=[{"kind": "HIGH", "price": 30.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BEARISH), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_bearish_target_exactly_zero_is_nanned(self) -> None:
        # Boundary: entry=10, HIGH=20 → target = 10 - (20 - 10) = 0 exactly.
        # Guard B's `target <= 0` must include equality, not just strict.
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BEARISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=10.0,
                    breakout_price=10.0,
                    pivots=[{"kind": "HIGH", "price": 20.0}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BEARISH), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])

    def test_bullish_subcent_target_is_preserved(self) -> None:
        # Negative control: legitimate sub-cent bullish setup. Guard must
        # not over-fire — target = 0.01 + (0.01 - 0.001) = 0.019 is positive.
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BULLISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=0.01,
                    breakout_price=0.01,
                    pivots=[{"kind": "LOW", "price": 0.001}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BULLISH), bars)
        target = out.loc[0, "target_price"]
        stop = out.loc[0, "stop_price"]
        assert np.isfinite(target) and target > 0
        assert np.isfinite(stop) and stop > 0

    def test_negative_entry_price_triggers_precondition_nan(self) -> None:
        # Guard A: breakout_price=None forces the entry_raw fallback at
        # lines 285-288 onto entry_price=-0.0008. Without the precondition,
        # the negative entry would propagate into geometry — Guard A NaNs
        # the row before any of that runs.
        bars = _bars()
        ts = bars.index[60]
        fs = bars.index[40]
        events = pd.DataFrame(
            [
                _event_row(
                    asset="AAA",
                    direction=Direction.BEARISH,
                    formation_start=fs,
                    breakout_ts=ts,
                    entry_price=-0.0008,
                    breakout_price=None,
                    pivots=[{"kind": "HIGH", "price": 0.01}],
                )
            ],
            columns=EVENTS_COLUMNS,
        )
        out = apply_condition(events, PatternCondition(direction=Direction.BEARISH), bars)
        assert np.isnan(out.loc[0, "target_price"])
        assert np.isnan(out.loc[0, "stop_price"])


# ---------------------------------------------------------------------------
# PatternCondition.override edge cases
# ---------------------------------------------------------------------------


class TestPatternConditionOverride:
    def test_unknown_field_raises_with_valid_list(self) -> None:
        cond = PatternCondition()
        with pytest.raises(TypeError, match="unknown PatternCondition fields"):
            cond.override(nonsense=1)

    def test_string_coercion_for_entry_and_exit_rules(self) -> None:
        cond = PatternCondition(direction=Direction.BULLISH).override(
            entry_rule="on_pullback",
            exit_rule="time_stop",
        )
        assert cond.entry_rule is EntryRule.ON_PULLBACK
        assert cond.exit_rule is ExitRule.TIME_STOP
