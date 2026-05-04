"""Tests for the ``.fc`` pandas accessor.

Covers both the :class:`SeriesAccessor` and :class:`DataFrameAccessor`.
The accessors are thin delegates over :mod:`fundcloud.metrics` /
:mod:`fundcloud.plots` / :mod:`fundcloud.sim`, so most tests only assert
the dispatch shape (return type, presence of expected fields), not the
underlying numeric correctness — that lives in the per-module test files.
"""

from __future__ import annotations

from pathlib import Path

import fundcloud  # noqa: F401 — registers the accessor
import numpy as np
import pandas as pd
import pytest

# --------------------------------------------------------------------- fixtures


@pytest.fixture
def bars_panel() -> pd.DataFrame:
    """Tiny two-asset OHLCV bars frame, MultiIndex columns."""
    rng = np.random.default_rng(2)
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=40, freq="B").values)
    close_a = 100 + np.cumsum(rng.normal(0, 0.5, 40))
    close_b = 50 + np.cumsum(rng.normal(0, 0.3, 40))
    cols = {
        ("open", "A"): close_a,
        ("high", "A"): close_a + 1,
        ("low", "A"): close_a - 1,
        ("close", "A"): close_a,
        ("volume", "A"): 1_000_000.0,
        ("open", "B"): close_b,
        ("high", "B"): close_b + 1,
        ("low", "B"): close_b - 1,
        ("close", "B"): close_b,
        ("volume", "B"): 1_000_000.0,
    }
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def benchmark_series() -> pd.Series:
    rng = np.random.default_rng(99)
    idx = pd.bdate_range("2021-01-04", periods=252)
    return pd.Series(rng.normal(0.0003, 0.008, 252), index=idx, name="bench")


# --------------------------------------------------------------------- registry


def test_series_accessor_registered() -> None:
    s = pd.Series([0.01, -0.02, 0.015, 0.005])
    assert hasattr(s, "fc")


def test_dataframe_accessor_registered() -> None:
    df = pd.DataFrame({"A": [0.01, -0.02], "B": [0.005, 0.01]})
    assert hasattr(df, "fc")


# --------------------------------------------------------------------- Series core


def test_series_sharpe_matches_free_function(returns_series: pd.Series) -> None:
    from fundcloud.metrics import core as M

    assert np.isclose(returns_series.fc.sharpe(), M.sharpe(returns_series))


def test_series_all_metrics_callable(returns_series: pd.Series) -> None:
    s = returns_series
    assert np.isfinite(s.fc.sharpe())
    assert np.isfinite(s.fc.sortino())
    assert np.isfinite(s.fc.calmar())
    assert np.isfinite(s.fc.omega())
    assert s.fc.max_drawdown() <= 0
    assert len(s.fc.drawdown_series()) == len(s)
    assert s.fc.cvar() <= s.fc.value_at_risk()


def test_series_extended_risk_adjusted(returns_series: pd.Series) -> None:
    s = returns_series
    assert isinstance(s.fc.adjusted_sortino(), float)
    assert isinstance(s.fc.probabilistic_sharpe(), float)
    assert isinstance(s.fc.smart_sharpe(), float)
    assert isinstance(s.fc.smart_sortino(), float)


def test_series_return_and_risk_metrics(returns_series: pd.Series) -> None:
    s = returns_series
    assert isinstance(s.fc.total_return(), float)
    assert isinstance(s.fc.cagr(), float)
    assert s.fc.volatility() > 0
    assert s.fc.downside_volatility() >= 0
    assert isinstance(s.fc.avg_return(), float)
    assert s.fc.best() >= s.fc.worst()
    assert isinstance(s.fc.skew(), float)
    assert isinstance(s.fc.kurtosis(), float)
    assert isinstance(s.fc.tail_ratio(), float)
    assert isinstance(s.fc.common_sense_ratio(), float)
    assert isinstance(s.fc.gain_to_pain_ratio(), float)
    assert s.fc.pain_index() >= 0
    assert isinstance(s.fc.pain_ratio(), float)
    assert isinstance(s.fc.ulcer_performance_index(), float)


def test_series_trade_stats(returns_series: pd.Series) -> None:
    s = returns_series
    assert 0.0 <= s.fc.win_rate() <= 1.0
    assert s.fc.avg_win() >= 0
    assert s.fc.avg_loss() <= 0
    assert isinstance(s.fc.payoff_ratio(), float)
    assert isinstance(s.fc.profit_factor(), float)
    assert 0.0 <= s.fc.exposure() <= 1.0
    assert isinstance(s.fc.kelly_criterion(), float)
    assert 0.0 <= s.fc.risk_of_ruin() <= 1.0
    assert s.fc.consecutive_wins() >= 0
    assert s.fc.consecutive_losses() >= 0


def test_series_drawdown_and_tail(returns_series: pd.Series) -> None:
    s = returns_series
    assert s.fc.max_drawdown() <= 0
    assert isinstance(s.fc.drawdown_series(), pd.Series)
    dd = s.fc.drawdown_details()
    assert isinstance(dd, pd.DataFrame)
    assert s.fc.ulcer_index() >= 0


def test_series_benchmark_relative(returns_series: pd.Series, benchmark_series: pd.Series) -> None:
    s = returns_series
    b = benchmark_series
    assert isinstance(s.fc.alpha(b), float)
    assert isinstance(s.fc.beta(b), float)
    assert 0.0 <= s.fc.r_squared(b) <= 1.0
    assert isinstance(s.fc.information_ratio(b), float)
    assert s.fc.tracking_error(b) >= 0
    assert isinstance(s.fc.up_capture(b), float)
    assert isinstance(s.fc.down_capture(b), float)
    assert isinstance(s.fc.capture_ratio(b), float)
    assert isinstance(s.fc.treynor_ratio(b), float)


def test_series_calendar_periods(returns_series: pd.Series, benchmark_series: pd.Series) -> None:
    s = returns_series
    monthly = s.fc.monthly_returns()
    assert isinstance(monthly, pd.DataFrame)

    yearly = s.fc.yearly_returns()
    assert isinstance(yearly, pd.Series)

    yearly_bench = s.fc.yearly_returns(benchmark=benchmark_series)
    assert isinstance(yearly_bench, pd.DataFrame)
    assert yearly_bench.shape[1] == 2

    # Unnamed benchmark falls back to "benchmark" column name.
    unnamed = pd.Series(benchmark_series.to_numpy(), index=benchmark_series.index)
    yearly_unnamed = s.fc.yearly_returns(benchmark=unnamed)
    assert isinstance(yearly_unnamed, pd.DataFrame)
    assert "benchmark" in yearly_unnamed.columns

    assert isinstance(s.fc.best_month(), float)
    assert isinstance(s.fc.worst_month(), float)
    assert isinstance(s.fc.best_year(), float)
    assert isinstance(s.fc.worst_year(), float)
    assert s.fc.positive_months() >= 0
    assert s.fc.negative_months() >= 0


def test_series_rolling_metrics(returns_series: pd.Series, benchmark_series: pd.Series) -> None:
    s = returns_series
    assert isinstance(s.fc.rolling_sharpe(window=30), pd.Series)
    assert isinstance(s.fc.rolling_sortino(window=30), pd.Series)
    assert isinstance(s.fc.rolling_volatility(window=30), pd.Series)
    assert isinstance(s.fc.rolling_beta(benchmark_series, window=30), pd.Series)
    assert isinstance(s.fc.rolling_drawdown(), pd.Series)


def test_series_one_shot_bundles(returns_series: pd.Series, benchmark_series: pd.Series) -> None:
    s = returns_series
    bundle = s.fc.metrics(benchmark=benchmark_series)
    assert isinstance(bundle, pd.Series)
    assert {"sharpe", "sortino", "max_drawdown"}.issubset(set(map(str, bundle.index)))

    period = s.fc.period_returns()
    assert isinstance(period, (pd.Series, pd.DataFrame))


# --------------------------------------------------------------------- Series EDA


def test_series_describe_returns_dataframe(returns_series: pd.Series) -> None:
    out = returns_series.fc.describe()
    assert isinstance(out, pd.DataFrame)


def test_series_profile_returns_report(returns_series: pd.Series) -> None:
    report = returns_series.fc.profile()
    # Soft assertion — report object is opaque but should not raise.
    assert report is not None


# --------------------------------------------------------------------- Series renderers


def test_series_render_html_writes_file(returns_series: pd.Series, tmp_path: Path) -> None:
    out = tmp_path / "series.html"
    returns_series.fc.render_html(out, title="Series HTML")
    assert out.exists()


def test_series_render_pdf_writes_pdf(returns_series: pd.Series, tmp_path: Path) -> None:
    out = tmp_path / "series.pdf"
    returns_series.fc.render_pdf(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_series_render_excel_writes_xlsx(returns_series: pd.Series, tmp_path: Path) -> None:
    out = tmp_path / "series.xlsx"
    returns_series.fc.render_excel(out)
    assert out.exists()


# --------------------------------------------------------------------- Series plots


def test_series_plot_methods_return_objects(
    returns_series: pd.Series, benchmark_series: pd.Series
) -> None:
    s = returns_series
    assert s.fc.plot_cumulative() is not None
    assert s.fc.plot_drawdown() is not None
    assert s.fc.plot_rolling_sharpe(window=30) is not None
    assert s.fc.plot_return_distribution() is not None
    assert s.fc.plot_monthly_heatmap() is not None
    assert s.fc.plot_yearly_returns(benchmark=benchmark_series) is not None
    assert s.fc.plot_summary() is not None


def test_series_to_returns_smoke() -> None:
    """`to_returns` on a Series of prices returns a Series of returns."""
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.Series([100.0, 101.0, 99.0, 102.0], index=idx)
    out = prices.fc.to_returns()
    assert isinstance(out, pd.Series)
    assert len(out) == 3  # first NaN dropped


# --------------------------------------------------------------------- DataFrame core


def test_dataframe_summary(returns_panel: pd.DataFrame) -> None:
    stats = returns_panel.fc.summary()
    assert "sharpe" in stats.index
    assert list(stats.columns) == list(returns_panel.columns)


def test_dataframe_metrics_bundle(returns_panel: pd.DataFrame, benchmark_series: pd.Series) -> None:
    bundle = returns_panel.fc.metrics(benchmark=benchmark_series)
    assert isinstance(bundle, pd.DataFrame)
    assert "sharpe" in bundle.index


def test_dataframe_per_column_metrics(returns_panel: pd.DataFrame) -> None:
    df = returns_panel
    assert isinstance(df.fc.sharpe(), pd.Series)
    assert isinstance(df.fc.sortino(), pd.Series)
    assert isinstance(df.fc.calmar(), pd.Series)
    assert isinstance(df.fc.omega(), pd.Series)
    assert isinstance(df.fc.max_drawdown(), pd.Series)
    assert isinstance(df.fc.drawdown_series(), pd.DataFrame)
    assert isinstance(df.fc.ulcer_index(), pd.Series)
    assert isinstance(df.fc.cvar(), pd.Series)
    assert isinstance(df.fc.value_at_risk(), pd.Series)


def test_dataframe_period_and_yearly_returns(
    returns_panel: pd.DataFrame, benchmark_series: pd.Series
) -> None:
    period = returns_panel.fc.period_returns()
    assert isinstance(period, (pd.Series, pd.DataFrame))

    yearly = returns_panel.fc.yearly_returns()
    assert isinstance(yearly, pd.DataFrame)

    yearly_bench = returns_panel.fc.yearly_returns(benchmark=benchmark_series)
    assert isinstance(yearly_bench, pd.DataFrame)
    # Benchmark column is included alongside the per-strategy columns.
    assert "bench" in yearly_bench.columns


def test_dataframe_to_returns(ohlcv_panel: pd.DataFrame) -> None:
    ret = ohlcv_panel.fc.to_returns()
    assert isinstance(ret, pd.DataFrame)
    assert len(ret) == len(ohlcv_panel) - 1


def test_dataframe_to_prices(ohlcv_panel: pd.DataFrame) -> None:
    prices = ohlcv_panel.fc.to_prices()
    assert isinstance(prices, pd.DataFrame)


# --------------------------------------------------------------------- DataFrame EDA


def test_dataframe_describe(returns_panel: pd.DataFrame) -> None:
    out = returns_panel.fc.describe()
    assert isinstance(out, pd.DataFrame)


def test_dataframe_profile(returns_panel: pd.DataFrame) -> None:
    report = returns_panel.fc.profile()
    assert report is not None


def test_dataframe_compare(returns_panel: pd.DataFrame) -> None:
    other = returns_panel.iloc[: len(returns_panel) // 2].copy()
    out = returns_panel.fc.compare(other)
    assert isinstance(out, (str, Path))


# --------------------------------------------------------------------- DataFrame renderers


def test_dataframe_render_html_single_col_writes_file(
    returns_panel: pd.DataFrame, tmp_path: Path
) -> None:
    """Single-column frames go through the Tearsheet path."""
    out = tmp_path / "single.html"
    returns_panel[["AAA"]].fc.render_html(out)
    assert out.exists()


def test_dataframe_render_html_multi_col_writes_file(
    returns_panel: pd.DataFrame, tmp_path: Path
) -> None:
    """Multi-column frames (no weights) go through the multi-asset path."""
    out = tmp_path / "multi.html"
    returns_panel.fc.render_html(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    for col in returns_panel.columns:
        assert col in text


def test_dataframe_render_html_returns_string_when_no_path(
    returns_panel: pd.DataFrame,
) -> None:
    html = returns_panel.fc.render_html()
    assert isinstance(html, str)
    assert "<html" in html


def test_dataframe_render_html_with_string_benchmark(
    returns_panel: pd.DataFrame, tmp_path: Path
) -> None:
    """Passing a column name as benchmark drops that column from the iteration."""
    out = tmp_path / "bench_str.html"
    returns_panel.fc.render_html(out, benchmark="AAA")
    text = out.read_text(encoding="utf-8")
    # AAA was the benchmark — only BBB and CCC sections should appear.
    assert "AAA" in text  # mentioned as benchmark label
    assert "BBB" in text
    assert "CCC" in text


def test_dataframe_render_html_with_weights_combines(
    returns_panel: pd.DataFrame, tmp_path: Path
) -> None:
    """Explicit weights= combines columns into a single-strategy tear sheet."""
    out = tmp_path / "weighted.html"
    returns_panel.fc.render_html(out, weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    assert out.exists()


def test_dataframe_render_pdf_multi_col(returns_panel: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "multi.pdf"
    returns_panel.fc.render_pdf(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_dataframe_render_pdf_single_col(returns_panel: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "single.pdf"
    returns_panel[["AAA"]].fc.render_pdf(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_dataframe_render_excel_multi_col(returns_panel: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "multi.xlsx"
    returns_panel.fc.render_excel(out)
    assert out.exists()


def test_dataframe_render_excel_single_col(returns_panel: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "single.xlsx"
    returns_panel[["AAA"]].fc.render_excel(out)
    assert out.exists()


def test_render_html_rejects_self_benchmark_when_single_column(tmp_path: Path) -> None:
    """Regression: ``returns.fc.render_html(benchmark='NQ=F')`` used to hit
    a ``ZeroDivisionError`` in ``portfolio_from_frame`` when the returns
    frame had only the benchmark column. Now raises a clear ``ValueError``."""
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({"NQ=F": rng.normal(0, 0.01, 60)}, index=idx)
    with pytest.raises(ValueError, match="only column"):
        returns.fc.render_html(tmp_path / "x.html", benchmark="NQ=F")


# --------------------------------------------------------------------- DataFrame plots


def test_dataframe_plot_methods(returns_panel: pd.DataFrame, benchmark_series: pd.Series) -> None:
    df = returns_panel
    assert df.fc.plot_cumulative() is not None
    assert df.fc.plot_drawdown() is not None
    assert df.fc.plot_rolling_sharpe(window=30) is not None
    assert df.fc.plot_return_distribution() is not None
    # monthly_heatmap accepts a single column.
    assert df[["AAA"]].fc.plot_monthly_heatmap() is not None
    assert df.fc.plot_yearly_returns(benchmark=benchmark_series) is not None


def test_dataframe_plot_summary(returns_panel: pd.DataFrame) -> None:
    fig = returns_panel.fc.plot_summary()
    assert fig is not None


def test_dataframe_plot_summary_with_string_benchmark(returns_panel: pd.DataFrame) -> None:
    """Resolving a string benchmark drops that column from the per-asset view."""
    fig = returns_panel.fc.plot_summary(benchmark="AAA")
    assert fig is not None


def test_dataframe_plot_composition_with_weights() -> None:
    """plot_composition takes a weights frame, not a returns frame."""
    idx = pd.bdate_range("2024-01-02", periods=20)
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=idx)
    fig = weights.fc.plot_composition()
    assert fig is not None


# --------------------------------------------------------------------- DataFrame simulator


def test_dataframe_run_strategy(bars_panel: pd.DataFrame) -> None:
    from fundcloud.strategies import Hold

    result = bars_panel.fc.run_strategy(Hold(weights={"A": 1.0}), cash=100_000)
    assert result is not None
    assert hasattr(result, "trades")


def test_dataframe_run_weights(bars_panel: pd.DataFrame) -> None:
    targets = pd.DataFrame({"A": 0.6, "B": 0.4}, index=bars_panel.index[:5])
    result = bars_panel.fc.run_weights(targets, cash=100_000)
    assert result is not None


def test_dataframe_run_signals(bars_panel: pd.DataFrame) -> None:
    entries = pd.DataFrame(False, index=bars_panel.index, columns=["A", "B"])
    entries.iloc[5, 0] = True
    exits = pd.DataFrame(False, index=bars_panel.index, columns=["A", "B"])
    exits.iloc[10, 0] = True
    result = bars_panel.fc.run_signals(entries, exits, size=0.5, cash=100_000)
    assert result is not None


def test_dataframe_run_orders(bars_panel: pd.DataFrame) -> None:
    orders = pd.DataFrame({
        "ts": [bars_panel.index[3]],
        "asset": ["A"],
        "side": ["buy"],
        "qty": [10.0],
    })
    result = bars_panel.fc.run_orders(orders, cash=100_000)
    assert result is not None


def test_dataframe_run_hold(bars_panel: pd.DataFrame) -> None:
    result = bars_panel.fc.run_hold({"A": 0.6, "B": 0.4}, cash=100_000)
    assert result is not None


def test_dataframe_run_dca(bars_panel: pd.DataFrame) -> None:
    result = bars_panel.fc.run_dca(500, horizon="weekly", cash=100_000)
    assert result is not None


def test_dataframe_run_hold_no_weights(bars_panel: pd.DataFrame) -> None:
    """``run_hold()`` with no weights defaults to equal split across assets."""
    result = bars_panel.fc.run_hold(cash=100_000)
    assert result is not None
    # bars_panel has at least two assets — both should be traded. A trade
    # *count* check would also pass if one asset traded twice, so assert
    # the actual asset coverage instead.
    assert {"A", "B"}.issubset(set(result.trades["asset"]))


def test_dataframe_run_dca_amount_pct(bars_panel: pd.DataFrame) -> None:
    """``run_dca`` accepts ``amount_pct`` instead of ``amount``."""
    result = bars_panel.fc.run_dca(amount_pct=0.01, horizon="weekly", cash=100_000)
    assert result is not None
    assert len(result.trades) >= 1


def test_dataframe_simulate_dispatches_strategy(bars_panel: pd.DataFrame) -> None:
    from fundcloud.strategies import Hold

    result = bars_panel.fc.simulate(Hold(weights={"A": 1.0}), cash=100_000)
    assert result is not None


def test_dataframe_simulate_dispatches_orders(bars_panel: pd.DataFrame) -> None:
    orders = pd.DataFrame({
        "ts": [bars_panel.index[3]],
        "asset": ["A"],
        "side": ["buy"],
        "qty": [10.0],
    })
    result = bars_panel.fc.simulate(orders, cash=100_000)
    assert result is not None


def test_dataframe_simulate_dispatches_signals(bars_panel: pd.DataFrame) -> None:
    """A DataFrame of bool dispatches to run_signals with exits=~entries."""
    entries = pd.DataFrame(False, index=bars_panel.index, columns=["A", "B"])
    entries.iloc[5:8, 0] = True
    result = bars_panel.fc.simulate(entries, cash=100_000)
    assert result is not None


def test_dataframe_simulate_dispatches_weights(bars_panel: pd.DataFrame) -> None:
    """A DataFrame of floats dispatches to run_weights."""
    targets = pd.DataFrame({"A": 0.6, "B": 0.4}, index=bars_panel.index[:5])
    result = bars_panel.fc.simulate(targets, cash=100_000)
    assert result is not None


def test_dataframe_simulate_rejects_unknown_type(bars_panel: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="BaseStrategy or a DataFrame"):
        bars_panel.fc.simulate("not-a-strategy")


def test_run_strategy_rejects_non_bars_frame(returns_panel: pd.DataFrame) -> None:
    """run_* methods reject plain returns frames — they need OHLCV bars."""
    from fundcloud.strategies import Hold

    with pytest.raises(TypeError, match="Bars frame"):
        returns_panel.fc.run_strategy(Hold(weights={"AAA": 1.0}))


# --------------------------------------------------------------------- helpers


def test_portfolio_from_frame_rejects_zero_columns() -> None:
    """Direct callers hitting a 0-column frame get a clear error, not a
    cryptic ``ZeroDivisionError``."""
    from fundcloud.accessors._helpers import portfolio_from_frame

    empty = pd.DataFrame(index=pd.date_range("2024-01-02", periods=10, freq="B"))
    with pytest.raises(ValueError, match="zero columns"):
        portfolio_from_frame(empty)


def test_portfolio_from_frame_with_series_weights() -> None:
    """Series weights get reindexed to the frame's columns."""
    from fundcloud.accessors._helpers import portfolio_from_frame

    idx = pd.bdate_range("2024-01-02", periods=10)
    df = pd.DataFrame({"A": np.zeros(10), "B": np.ones(10) * 0.01}, index=idx)
    weights = pd.Series({"A": 0.5, "B": 0.5})
    pf = portfolio_from_frame(df, weights=weights)
    assert pf.returns.iloc[0] == pytest.approx(0.005)


def test_portfolio_from_frame_named_strategy() -> None:
    """A name= override propagates to the Portfolio."""
    from fundcloud.accessors._helpers import portfolio_from_frame

    s = pd.Series([0.01, 0.02, -0.01], name="orig")
    pf = portfolio_from_frame(s, name="custom")
    assert pf.name == "custom"


def test_portfolios_per_column_drops_leading_nan() -> None:
    from fundcloud.accessors._helpers import portfolios_per_column

    idx = pd.bdate_range("2024-01-02", periods=10)
    df = pd.DataFrame(
        {
            "early": np.linspace(0.001, 0.01, 10),
            "late": [np.nan] * 5 + list(np.linspace(0.001, 0.005, 5)),
        },
        index=idx,
    )
    pfs = portfolios_per_column(df)
    assert len(pfs) == 2
    early_pf = next(pf for name, pf in pfs if name == "early")
    late_pf = next(pf for name, pf in pfs if name == "late")
    assert len(early_pf.returns) == 10
    assert len(late_pf.returns) == 5  # leading NaN dropped


def test_is_bars_frame_true_for_ohlcv(ohlcv_panel: pd.DataFrame) -> None:
    from fundcloud.accessors._helpers import is_bars_frame

    assert is_bars_frame(ohlcv_panel) is True


def test_is_bars_frame_false_for_returns(returns_panel: pd.DataFrame) -> None:
    from fundcloud.accessors._helpers import is_bars_frame

    assert is_bars_frame(returns_panel) is False


def test_as_sim_kwargs_extracts_known_keys() -> None:
    from fundcloud.accessors._helpers import as_sim_kwargs

    kw = {"costs": "x", "slippage": "y", "execution": "z", "cash": 1000, "extra": "ignored"}
    out = as_sim_kwargs(kw)
    assert out == {"costs": "x", "slippage": "y", "execution": "z", "cash": 1000}


def test_resolve_benchmark_series_passthrough(benchmark_series: pd.Series) -> None:
    from fundcloud._benchmark import resolve_benchmark

    out = resolve_benchmark(None, benchmark_series)
    assert out is benchmark_series


def test_resolve_benchmark_string_lookup(returns_panel: pd.DataFrame) -> None:
    from fundcloud._benchmark import resolve_benchmark

    out = resolve_benchmark(returns_panel, "AAA")
    assert isinstance(out, pd.Series)
    assert out.name == "AAA"


def test_resolve_benchmark_unknown_string_raises(returns_panel: pd.DataFrame) -> None:
    from fundcloud._benchmark import resolve_benchmark

    with pytest.raises(ValueError, match="no matching column"):
        resolve_benchmark(returns_panel, "DOES_NOT_EXIST")


def test_resolve_benchmark_rejects_unknown_type() -> None:
    from fundcloud._benchmark import resolve_benchmark

    with pytest.raises(TypeError, match="must be"):
        resolve_benchmark(None, 42)  # type: ignore[arg-type]


def test_resolve_benchmark_none_returns_none() -> None:
    from fundcloud._benchmark import resolve_benchmark

    assert resolve_benchmark(None, None) is None
