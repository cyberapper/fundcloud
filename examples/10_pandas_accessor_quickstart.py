"""10 — The ``.fc`` pandas accessor: no simulator, no strategy.

Trader scenario: you already have your returns in a pandas ``Series`` or
``DataFrame`` — exported from a broker statement, a Google Sheet, or a
notebook experiment — and you want instant performance metrics without
setting up a ``Simulator``. ``import fundcloud`` registers the ``.fc``
accessor on both ``pd.Series`` and ``pd.DataFrame``; every metric in
``fundcloud.metrics.core`` is available as a one-liner.

Run:
    uv run python examples/10_pandas_accessor_quickstart.py
"""

from __future__ import annotations

from pathlib import Path

# Side-effect import: registers the ``.fc`` accessor on pandas.
import fundcloud  # noqa: F401
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def single_strategy_workflow() -> pd.Series:
    """Typical 'I have a returns Series' workflow."""
    _rule("1. One strategy, one Series")

    # Pretend this came from `pd.read_csv("my_returns.csv", index_col="date")["return"]`.
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=252)
    returns = pd.Series(
        rng.normal(0.0008, 0.012, len(idx)),
        index=idx,
        name="my_account",
    )

    print(f"Source:             pd.Series named {returns.name!r}  ({len(returns)} bars)")
    print()
    print(f"Sharpe:             {returns.fc.sharpe():>8.2f}")
    print(f"Sortino:            {returns.fc.sortino():>8.2f}")
    print(f"Calmar:             {returns.fc.calmar():>8.2f}")
    print(f"Omega (target 0%):  {returns.fc.omega():>8.2f}")
    print(f"Max drawdown:       {returns.fc.max_drawdown() * 100:>7.2f}%")
    print(f"Ulcer index:        {returns.fc.ulcer_index():>8.2f}")
    print(f"VaR (95%):          {returns.fc.value_at_risk() * 100:>7.2f}%")
    print(f"CVaR (95%):         {returns.fc.cvar() * 100:>7.2f}%")

    # The drawdown series is itself a Series — handy for plotting or
    # "when did I last hit a new high?" queries.
    dd = returns.fc.drawdown_series()
    deepest = dd.idxmin()
    print(f"\nDeepest drawdown:   {dd.min() * 100:>7.2f}%  on {deepest.date()}")

    # Every metric also works through the free-function form —
    # `fundcloud.metrics.core.sharpe(returns)` — but the accessor is more
    # readable inside notebooks.
    return returns


def multi_strategy_comparison() -> pd.DataFrame:
    """Typical 'I want to compare three variants side by side' workflow."""
    _rule("2. Three strategies, one DataFrame")

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-02", periods=252)
    df = pd.DataFrame(
        {
            "conservative": rng.normal(0.0003, 0.006, len(idx)),
            "balanced": rng.normal(0.0006, 0.010, len(idx)),
            "aggressive": rng.normal(0.0010, 0.018, len(idx)),
        },
        index=idx,
    )
    print(f"Source:   pd.DataFrame  (cols={list(df.columns)}, {len(df)} bars)")

    # Scalar-returning metrics become Series keyed by column when called on
    # a DataFrame — the panel version of the single-Series case above.
    print("\nSharpe per column:")
    print(df.fc.sharpe().to_string(float_format=lambda v: f"{v:>7.2f}"))

    print("\nMax drawdown per column:")
    print((df.fc.max_drawdown() * 100).to_string(float_format=lambda v: f"{v:>7.2f}%"))

    # One-shot metric-by-column table — the same shape a tear sheet prints.
    print("\nSummary (metric-by-strategy):\n")
    summary = df.fc.summary()
    rows = ["cagr", "ann_volatility", "sharpe", "sortino", "max_drawdown", "cvar"]
    print(summary.loc[rows].to_string(float_format=lambda v: f"{v:>10.4f}"))

    winner = str(summary.loc["sharpe"].idxmax())
    print(f"\nBest Sharpe:   {winner}")
    return df


def prices_to_returns_roundtrip() -> None:
    """Traders often have prices, not returns. ``.fc.to_returns()`` closes the gap."""
    _rule("3. Prices → returns (and back)")

    # Imagine loading a broker CSV of end-of-day prices.
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=60)
    prices = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.011, len(idx)))),
        index=idx,
        name="AAPL",
    )
    print(f"prices head:\n{prices.head().round(2).to_string()}")

    returns = prices.fc.to_returns()
    print(f"\nprices.fc.to_returns() head:\n{returns.head().round(5).to_string()}")
    print(f"\nSharpe from the derived returns:  {returns.fc.sharpe():.2f}")

    # Works on a DataFrame of prices too — lifts to per-column returns.
    multi_prices = pd.DataFrame({"AAPL": prices, "MSFT": prices * 1.1})
    multi_returns = multi_prices.fc.to_returns()
    print(f"\nmulti-asset pct-change shape:  {multi_returns.shape}")


def benchmark_relative_metrics() -> None:
    """Retail traders constantly ask 'did I beat SPY?' — here's the one-liner."""
    _rule("4. Strategy vs. benchmark")

    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2023-01-02", periods=252)
    strategy = pd.Series(rng.normal(0.0008, 0.013, len(idx)), index=idx, name="me")
    benchmark = pd.Series(rng.normal(0.0005, 0.010, len(idx)), index=idx, name="spy")

    alpha = strategy.fc.sharpe() - benchmark.fc.sharpe()
    print(f"My Sharpe:        {strategy.fc.sharpe():.2f}")
    print(f"SPY Sharpe:       {benchmark.fc.sharpe():.2f}")
    print(f"Sharpe gap:       {alpha:+.2f}")

    # Active returns are just subtraction — then .fc for the follow-up stats.
    active = (strategy - benchmark).rename("active")
    print(f"\nActive Sharpe (naïve):  {active.fc.sharpe():.2f}")
    print(f"Active max drawdown:    {active.fc.max_drawdown() * 100:.2f}%")


def tear_sheet_from_plain_series() -> None:
    """Passing ``.fc``-flavoured data into the tear sheet is one line away.

    ``Tearsheet`` takes a :class:`fundcloud.portfolio.Portfolio`, and
    building a Portfolio from a bare returns Series is trivial — no
    simulator involved.
    """
    _rule("5. One-liner tear sheet from a Series")

    from fundcloud.portfolio import Portfolio
    from fundcloud.reports import Tearsheet

    rng = np.random.default_rng(19)
    idx = pd.bdate_range("2022-01-03", periods=504)
    returns = pd.Series(rng.normal(0.0006, 0.011, len(idx)), index=idx, name="demo")

    portfolio = Portfolio(returns=returns, name="demo")
    out = OUT / "10_pandas_accessor_tearsheet.html"
    Tearsheet(portfolio, title="Quickstart tear sheet from a pd.Series").render_html(out)
    print(f"Wrote tear sheet:  {out.relative_to(HERE.parent)}")


def multi_asset_plot(df: pd.DataFrame) -> None:
    """Every builder in :mod:`fundcloud.plots` accepts multi-asset DataFrames.

    Reuses the DataFrame built in :func:`multi_strategy_comparison` —
    ``plots.cumulative`` overlays one wealth curve per column, and
    ``plots.summary`` composes the canonical panels into one figure.
    """
    _rule("6. plot a DataFrame of returns (one line per column)")

    from fundcloud import plots

    cumulative = plots.cumulative(df, annotations=True, title="All three strategies")
    summary = plots.summary(df, title="Three-strategy overview")

    cum_out = OUT / "10_cumulative_multi.html"
    sum_out = OUT / "10_summary_multi.html"
    cumulative.write_html(cum_out)
    summary.write_html(sum_out)
    print(f"Wrote {cum_out.relative_to(HERE.parent)}  (3 wealth curves)")
    print(f"Wrote {sum_out.relative_to(HERE.parent)}  (composed summary figure)")


def accessor_reports_eda_and_plots(df: pd.DataFrame) -> None:
    """The full ``.fc`` accessor surface — reports, EDA, plots, one-liners."""
    _rule("7. Reports / EDA / plots — all via .fc.*")

    # 7a — Describe: super-set of pandas describe + finance extras
    desc = df.fc.describe()
    print(f"df.fc.describe() shape:  {desc.shape}")
    print(f"  sharpe row present:   {'sharpe' in desc.columns}")
    print(f"  sample — 'balanced' sharpe = {desc.loc['balanced', 'sharpe']:.2f}")

    # 7b — ProfileReport (Python-first object with .stats / .alerts / .to_html)
    report = df.fc.profile(title="Multi-strategy profile")
    print(f"\n  df.fc.profile() -> {type(report).__name__}")
    print(f"  .alerts  : {len(report.alerts)} issues flagged")
    print(f"  .stats   : {report.stats.shape[0]} columns profiled")

    # 7c — Tear-sheet render from the DataFrame directly
    out = OUT / "10_accessor_tearsheet.html"
    df.fc.render_html(out, title="DataFrame → Tearsheet via .fc.render_html")
    print(
        f"\n  df.fc.render_html(...) -> {out.relative_to(HERE.parent)}  "
        f"({out.stat().st_size / 1024:.0f} KB)"
    )

    # 7d — Plots return plotly figures (render inline in Jupyter)
    cumulative_fig = df.fc.plot_cumulative(title="Three strategies (via accessor)")
    dd_fig = df.fc.plot_drawdown()
    print(f"\n  df.fc.plot_cumulative() -> {type(cumulative_fig).__name__}")
    print(f"  df.fc.plot_drawdown()   -> {type(dd_fig).__name__}")


def advanced_metrics_showcase(returns: pd.Series) -> None:
    """Trade statistics, advanced Sharpe variants, and tail-risk measures."""
    _rule("8. Advanced metrics — trade stats, smart ratios, tail risk")

    # Trade statistics — most informative for active strategies
    print("Trade statistics:")
    print(f"  Win rate:           {returns.fc.win_rate():>7.2%}")
    print(f"  Avg win:            {returns.fc.avg_win() * 100:>7.3f}%")
    print(f"  Avg loss:           {returns.fc.avg_loss() * 100:>7.3f}%")
    print(f"  Payoff ratio:       {returns.fc.payoff_ratio():>7.2f}")
    print(f"  Profit factor:      {returns.fc.profit_factor():>7.2f}")
    print(f"  Consecutive wins:   {returns.fc.consecutive_wins():>7d}")
    print(f"  Consecutive losses: {returns.fc.consecutive_losses():>7d}")
    print(f"  Exposure:           {returns.fc.exposure():>7.2%}")

    # Autocorrelation-adjusted Sharpe variants
    print("\nSharpe variants (correct for autocorrelation and sample size):")
    print(f"  Standard Sharpe:    {returns.fc.sharpe():>7.2f}")
    print(f"  Smart Sharpe:       {returns.fc.smart_sharpe():>7.2f}  (autocorrelation-adjusted)")
    print(f"  Smart Sortino:      {returns.fc.smart_sortino():>7.2f}  (autocorrelation-adjusted)")
    print(f"  Adjusted Sortino:   {returns.fc.adjusted_sortino():>7.2f}  (robust to small samples)")
    psr = returns.fc.probabilistic_sharpe(target_sharpe=0.5)
    print(f"  PSR (SR≥0.5):       {psr:>7.2%}  (probability Sharpe exceeds 0.5)")

    # Position sizing helpers
    print("\nPosition sizing:")
    print(f"  Kelly criterion:    {returns.fc.kelly_criterion():>7.2%}  (use 1/4–1/2 in practice)")
    ror = returns.fc.risk_of_ruin(ruin_level=0.5)
    print(f"  Risk of ruin (50%): {ror:>7.4f}")

    # Tail risk and pain measures
    print("\nTail risk & pain measures:")
    print(f"  Tail ratio:         {returns.fc.tail_ratio():>7.2f}  (right tail / left tail)")
    print(f"  Gain-to-pain:       {returns.fc.gain_to_pain_ratio():>7.2f}")
    print(f"  Pain index:         {returns.fc.pain_index():>7.4f}")
    print(f"  Pain ratio:         {returns.fc.pain_ratio():>7.2f}")
    print(f"  Common sense ratio: {returns.fc.common_sense_ratio():>7.2f}")
    print(f"  Ulcer perf. index:  {returns.fc.ulcer_performance_index():>7.2f}")


def main() -> None:
    returns = single_strategy_workflow()
    df = multi_strategy_comparison()
    prices_to_returns_roundtrip()
    benchmark_relative_metrics()
    tear_sheet_from_plain_series()
    multi_asset_plot(df)
    accessor_reports_eda_and_plots(df)
    advanced_metrics_showcase(returns)


if __name__ == "__main__":
    main()
