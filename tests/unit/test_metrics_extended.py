"""Tests for the extended metric surface in :mod:`fundcloud.metrics`.

Focused checks on formula correctness and DataFrame/Series shape consistency.
The single-source-of-truth ``metrics()`` bundle is covered by a shape test
(every expected key is present) and a small cross-consistency set (a few
entries must equal the individual free-function calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.metrics import (
    alpha,
    avg_loss,
    avg_win,
    best,
    beta,
    cagr,
    capture_ratio,
    common_sense_ratio,
    consecutive_losses,
    consecutive_wins,
    down_capture,
    downside_volatility,
    drawdown_details,
    exposure,
    gain_to_pain_ratio,
    information_ratio,
    kelly_criterion,
    kurtosis,
    max_drawdown,
    metrics,
    monthly_returns,
    pain_index,
    payoff_ratio,
    positive_months,
    probabilistic_sharpe_ratio,
    profit_factor,
    r_squared,
    rolling_sharpe,
    rolling_volatility,
    sharpe,
    skew,
    smart_sharpe,
    tail_ratio,
    total_return,
    tracking_error,
    treynor_ratio,
    up_capture,
    value_at_risk,
    volatility,
    win_rate,
    worst,
    yearly_returns,
)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03", periods=756, freq="B")
    return pd.Series(rng.normal(0.0005, 0.012, 756), index=idx, name="demo")


@pytest.fixture
def benchmark() -> pd.Series:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-03", periods=756, freq="B")
    return pd.Series(rng.normal(0.0003, 0.010, 756), index=idx, name="SPY")


# --------------------------------------------------- return + risk


def test_total_return_matches_cumprod(returns: pd.Series) -> None:
    assert pytest.approx((1.0 + returns).prod() - 1.0) == total_return(returns)


def test_cagr_consistent_with_total_return(returns: pd.Series) -> None:
    t = (1.0 + returns).prod()
    expected = t ** (252 / len(returns)) - 1.0
    assert pytest.approx(expected, rel=1e-9) == cagr(returns, periods_per_year=252)


def test_volatility_annualises(returns: pd.Series) -> None:
    assert pytest.approx(returns.std(ddof=1) * np.sqrt(252)) == volatility(
        returns, periods_per_year=252
    )


def test_downside_volatility_lte_volatility(returns: pd.Series) -> None:
    vol = volatility(returns, periods_per_year=252)
    d_vol = downside_volatility(returns, periods_per_year=252)
    assert d_vol <= vol + 1e-9


def test_skew_and_kurtosis_match_pandas(returns: pd.Series) -> None:
    assert pytest.approx(returns.skew()) == skew(returns)
    assert pytest.approx(returns.kurtosis()) == kurtosis(returns)


def test_best_and_worst(returns: pd.Series) -> None:
    assert best(returns) == returns.max()
    assert worst(returns) == returns.min()


# --------------------------------------------------- win / loss stats


def test_win_rate_range(returns: pd.Series) -> None:
    wr = win_rate(returns)
    assert 0.0 <= wr <= 1.0


def test_payoff_profit_factor_positive_on_good_strategy() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.005, 0.015])
    assert payoff_ratio(r) > 1.0
    assert profit_factor(r) > 1.0


def test_exposure_on_sparse_series() -> None:
    r = pd.Series([0.0, 0.01, 0.0, 0.0, -0.02])
    assert pytest.approx(0.4) == exposure(r)


def test_avg_win_and_avg_loss_signs() -> None:
    r = pd.Series([0.02, -0.01, 0.03, -0.005])
    assert avg_win(r) > 0
    assert avg_loss(r) < 0


def test_kelly_criterion_bounded(returns: pd.Series) -> None:
    k = kelly_criterion(returns)
    assert -1.0 < k < 1.0


def test_consecutive_streaks() -> None:
    r = pd.Series([0.01, 0.02, 0.015, -0.01, -0.005, 0.01])
    assert consecutive_wins(r) == 3
    assert consecutive_losses(r) == 2


# --------------------------------------------------- tail + pain


def test_tail_ratio_is_positive(returns: pd.Series) -> None:
    assert tail_ratio(returns) > 0.0


def test_common_sense_ratio_equals_tail_times_profit(returns: pd.Series) -> None:
    t = tail_ratio(returns)
    p = profit_factor(returns)
    assert pytest.approx(t * p) == common_sense_ratio(returns)


def test_pain_index_nonnegative(returns: pd.Series) -> None:
    assert pain_index(returns) >= 0.0


def test_gain_to_pain_finite(returns: pd.Series) -> None:
    assert np.isfinite(gain_to_pain_ratio(returns))


def test_value_at_risk_below_zero_with_random_returns(returns: pd.Series) -> None:
    # 5% quantile of a roughly-zero-mean return stream is negative.
    assert value_at_risk(returns) < 0.0


def test_max_drawdown_is_nonpositive(returns: pd.Series) -> None:
    assert max_drawdown(returns) <= 0.0


# --------------------------------------------------- risk-adjusted


def test_probabilistic_sharpe_bounded(returns: pd.Series) -> None:
    p = probabilistic_sharpe_ratio(returns)
    assert 0.0 <= p <= 1.0


def test_smart_sharpe_related_to_sharpe(returns: pd.Series) -> None:
    # With weak autocorrelation, smart_sharpe should be near (but not equal to)
    # the plain Sharpe.
    s = sharpe(returns)
    ss = smart_sharpe(returns)
    assert abs(ss - s) < abs(s) * 0.5 + 0.5


# --------------------------------------------------- benchmark


def test_beta_is_close_to_one_when_self_benchmark(returns: pd.Series) -> None:
    assert pytest.approx(1.0, abs=1e-9) == beta(returns, returns)


def test_r_squared_equals_one_self(returns: pd.Series) -> None:
    assert pytest.approx(1.0, abs=1e-9) == r_squared(returns, returns)


def test_alpha_zero_self_benchmark(returns: pd.Series) -> None:
    assert pytest.approx(0.0, abs=1e-9) == alpha(returns, returns)


def test_information_ratio_vs_self_is_nan(returns: pd.Series) -> None:
    # Active returns identically zero => 0/0 => nan.
    assert np.isnan(information_ratio(returns, returns))


def test_tracking_error_vs_self_zero(returns: pd.Series) -> None:
    assert pytest.approx(0.0, abs=1e-12) == tracking_error(returns, returns)


def test_capture_ratio_agrees_with_components(returns: pd.Series, benchmark: pd.Series) -> None:
    up = up_capture(returns, benchmark)
    down = down_capture(returns, benchmark)
    cap = capture_ratio(returns, benchmark)
    if np.isfinite(cap):
        assert pytest.approx(up / down) == cap


def test_treynor_finite(returns: pd.Series, benchmark: pd.Series) -> None:
    t = treynor_ratio(returns, benchmark)
    assert np.isfinite(t)


# --------------------------------------------------- period tables


def test_monthly_returns_shape(returns: pd.Series) -> None:
    table = monthly_returns(returns)
    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert table.shape[1] == 12 or table.shape[1] <= 12


def test_yearly_returns_length(returns: pd.Series) -> None:
    y = yearly_returns(returns)
    # sample spans 3 calendar years.
    assert 2 <= len(y) <= 4


def test_positive_months_non_negative(returns: pd.Series) -> None:
    assert positive_months(returns) >= 0


# --------------------------------------------------- rolling


def test_rolling_sharpe_shape(returns: pd.Series) -> None:
    rs = rolling_sharpe(returns, window=63)
    assert isinstance(rs, pd.Series)
    assert len(rs) == len(returns)
    assert rs.iloc[:62].isna().all()
    assert rs.iloc[62:].notna().any()


def test_rolling_volatility_nonnegative(returns: pd.Series) -> None:
    rv = rolling_volatility(returns, window=21)
    assert (rv.dropna() >= 0).all()


# --------------------------------------------------- drawdown details


def test_drawdown_details_columns(returns: pd.Series) -> None:
    dd = drawdown_details(returns)
    for col in (
        "start",
        "valley",
        "recovery",
        "max_drawdown",
        "duration_days",
        "days_to_recover",
    ):
        assert col in dd.columns


def test_drawdown_details_first_row_is_deepest(returns: pd.Series) -> None:
    dd = drawdown_details(returns)
    assert dd["max_drawdown"].iloc[0] == dd["max_drawdown"].min()


# --------------------------------------------------- one-shot bundle


def test_metrics_series_bundle_has_all_keys(returns: pd.Series) -> None:
    m = metrics(returns)
    required = {
        "periods",
        "total_return",
        "cagr",
        "ann_volatility",
        "downside_volatility",
        "best",
        "worst",
        "win_rate",
        "payoff_ratio",
        "profit_factor",
        "exposure",
        "consecutive_wins",
        "consecutive_losses",
        "kelly_criterion",
        "risk_of_ruin",
        "skew",
        "kurtosis",
        "tail_ratio",
        "common_sense_ratio",
        "pain_index",
        "pain_ratio",
        "gain_to_pain_ratio",
        "max_drawdown",
        "ulcer_index",
        "ulcer_performance_index",
        "value_at_risk",
        "cvar",
        "sharpe",
        "sortino",
        "calmar",
        "omega",
        "adjusted_sortino",
        "probabilistic_sharpe",
        "smart_sharpe",
        "smart_sortino",
        "best_month",
        "worst_month",
        "best_year",
        "worst_year",
        "positive_months",
        "negative_months",
    }
    missing = required - set(m.index)
    assert not missing, f"metrics() missing expected keys: {missing}"


def test_metrics_with_benchmark_adds_benchmark_keys(
    returns: pd.Series, benchmark: pd.Series
) -> None:
    m = metrics(returns, benchmark=benchmark)
    for key in (
        "alpha",
        "beta",
        "r_squared",
        "information_ratio",
        "tracking_error",
        "up_capture",
        "down_capture",
        "capture_ratio",
        "treynor_ratio",
    ):
        assert key in m.index


def test_metrics_cross_consistency(returns: pd.Series) -> None:
    m = metrics(returns)
    assert pytest.approx(sharpe(returns)) == m["sharpe"]
    assert pytest.approx(max_drawdown(returns)) == m["max_drawdown"]
    assert pytest.approx(cagr(returns)) == m["cagr"]


def test_metrics_panel_returns_dataframe() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-03", periods=252, freq="B")
    panel = pd.DataFrame(
        {"a": rng.normal(0.0002, 0.010, 252), "b": rng.normal(0.0005, 0.014, 252)},
        index=idx,
    )
    m = metrics(panel)
    assert isinstance(m, pd.DataFrame)
    assert set(m.columns) == {"a", "b"}
    assert "sharpe" in m.index


def test_accessor_metrics_matches_free_function(returns: pd.Series) -> None:
    import fundcloud  # noqa: F401 — registers the accessor

    from_accessor = returns.fc.metrics()
    from_free_fn = metrics(returns)
    pd.testing.assert_series_equal(from_accessor, from_free_fn)


def test_portfolio_metrics_delegates() -> None:
    from fundcloud.portfolio import Portfolio

    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-03", periods=252, freq="B")
    r = pd.Series(rng.normal(0.0005, 0.01, 252), index=idx, name="p")
    p = Portfolio(returns=r, name="p")
    m = p.metrics()
    assert "sharpe" in m.index
    assert m["sharpe"] == pytest.approx(sharpe(r))
