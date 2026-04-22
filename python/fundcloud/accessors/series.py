"""``.fc`` accessor on :class:`pandas.Series`.

Every method is a thin delegation to a canonical free function in
:mod:`fundcloud.metrics` or :mod:`fundcloud.data.bars`. No logic lives in
the accessor — if you find yourself writing any, move it to the free-fn
module and call that here.

Examples
--------
>>> import pandas as pd
>>> import fundcloud  # registers the accessor
>>> r = pd.Series([0.01, -0.005, 0.008, -0.010, 0.015])
>>> r.fc.sharpe(periods_per_year=252)  # doctest: +SKIP
>>> r.fc.metrics().head()             # 55-metric bundle  # doctest: +SKIP
"""

from __future__ import annotations

import pandas as pd

from fundcloud import metrics as _metrics
from fundcloud.data import bars as _bars

__all__ = ["SeriesAccessor"]


@pd.api.extensions.register_series_accessor("fc")
class SeriesAccessor:
    """Fundcloud accessor namespace for a ``pd.Series`` of returns."""

    def __init__(self, obj: pd.Series) -> None:
        self._obj = obj

    # --- risk-adjusted ------------------------------------------------------
    def sharpe(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> float:
        return _metrics.sharpe(self._obj, risk_free=risk_free, periods_per_year=periods_per_year)

    def sortino(self, *, target: float = 0.0, periods_per_year: int | None = None) -> float:
        return _metrics.sortino(self._obj, target=target, periods_per_year=periods_per_year)

    def calmar(self, *, periods_per_year: int | None = None) -> float:
        return _metrics.calmar(self._obj, periods_per_year=periods_per_year)

    def omega(self, *, target: float = 0.0) -> float:
        return _metrics.omega(self._obj, target=target)

    def adjusted_sortino(
        self, *, target: float = 0.0, periods_per_year: int | None = None
    ) -> float:
        return _metrics.adjusted_sortino(
            self._obj, target=target, periods_per_year=periods_per_year
        )

    def probabilistic_sharpe(
        self, *, target_sharpe: float = 0.0, periods_per_year: int | None = None
    ) -> float:
        return _metrics.probabilistic_sharpe_ratio(
            self._obj, target_sharpe=target_sharpe, periods_per_year=periods_per_year
        )

    def smart_sharpe(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> float:
        return _metrics.smart_sharpe(
            self._obj, risk_free=risk_free, periods_per_year=periods_per_year
        )

    def smart_sortino(self, *, target: float = 0.0, periods_per_year: int | None = None) -> float:
        return _metrics.smart_sortino(self._obj, target=target, periods_per_year=periods_per_year)

    # --- return / risk ------------------------------------------------------
    def total_return(self) -> float:
        return _metrics.total_return(self._obj)

    def cagr(self, *, periods_per_year: int | None = None) -> float:
        return _metrics.cagr(self._obj, periods_per_year=periods_per_year)

    def volatility(self, *, periods_per_year: int | None = None) -> float:
        return _metrics.volatility(self._obj, periods_per_year=periods_per_year)

    def downside_volatility(
        self, *, target: float = 0.0, periods_per_year: int | None = None
    ) -> float:
        return _metrics.downside_volatility(
            self._obj, target=target, periods_per_year=periods_per_year
        )

    def avg_return(self) -> float:
        return _metrics.avg_return(self._obj)

    def best(self) -> float:
        return _metrics.best(self._obj)

    def worst(self) -> float:
        return _metrics.worst(self._obj)

    def skew(self) -> float:
        return _metrics.skew(self._obj)

    def kurtosis(self) -> float:
        return _metrics.kurtosis(self._obj)

    def tail_ratio(self, *, alpha: float = 0.05) -> float:
        return _metrics.tail_ratio(self._obj, alpha=alpha)

    def common_sense_ratio(self) -> float:
        return _metrics.common_sense_ratio(self._obj)

    def gain_to_pain_ratio(self) -> float:
        return _metrics.gain_to_pain_ratio(self._obj)

    def pain_index(self) -> float:
        return _metrics.pain_index(self._obj)

    def pain_ratio(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> float:
        return _metrics.pain_ratio(
            self._obj, risk_free=risk_free, periods_per_year=periods_per_year
        )

    def ulcer_performance_index(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> float:
        return _metrics.ulcer_performance_index(
            self._obj, risk_free=risk_free, periods_per_year=periods_per_year
        )

    # --- trade stats --------------------------------------------------------
    def win_rate(self) -> float:
        return _metrics.win_rate(self._obj)

    def avg_win(self) -> float:
        return _metrics.avg_win(self._obj)

    def avg_loss(self) -> float:
        return _metrics.avg_loss(self._obj)

    def payoff_ratio(self) -> float:
        return _metrics.payoff_ratio(self._obj)

    def profit_factor(self) -> float:
        return _metrics.profit_factor(self._obj)

    def exposure(self) -> float:
        return _metrics.exposure(self._obj)

    def kelly_criterion(self) -> float:
        return _metrics.kelly_criterion(self._obj)

    def risk_of_ruin(self, *, starting_capital: float = 1.0, ruin_level: float = 0.0) -> float:
        return _metrics.risk_of_ruin(
            self._obj, starting_capital=starting_capital, ruin_level=ruin_level
        )

    def consecutive_wins(self) -> int:
        return _metrics.consecutive_wins(self._obj)

    def consecutive_losses(self) -> int:
        return _metrics.consecutive_losses(self._obj)

    # --- drawdown / tail ----------------------------------------------------
    def max_drawdown(self) -> float:
        return _metrics.max_drawdown(self._obj)

    def drawdown_series(self) -> pd.Series:
        return _metrics.drawdown_series(self._obj)

    def drawdown_details(self) -> pd.DataFrame:
        return _metrics.drawdown_details(self._obj)

    def ulcer_index(self) -> float:
        return _metrics.ulcer_index(self._obj)

    def cvar(self, *, alpha: float = 0.95) -> float:
        return _metrics.cvar(self._obj, alpha=alpha)

    def value_at_risk(self, *, alpha: float = 0.95) -> float:
        return _metrics.value_at_risk(self._obj, alpha=alpha)

    # --- benchmark-relative -------------------------------------------------
    def alpha(
        self,
        benchmark: pd.Series,
        *,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
    ) -> float:
        return _metrics.alpha(
            self._obj,
            benchmark,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
        )

    def beta(self, benchmark: pd.Series) -> float:
        return _metrics.beta(self._obj, benchmark)

    def r_squared(self, benchmark: pd.Series) -> float:
        return _metrics.r_squared(self._obj, benchmark)

    def information_ratio(self, benchmark: pd.Series) -> float:
        return _metrics.information_ratio(self._obj, benchmark)

    def tracking_error(self, benchmark: pd.Series, *, periods_per_year: int | None = None) -> float:
        return _metrics.tracking_error(self._obj, benchmark, periods_per_year=periods_per_year)

    def up_capture(self, benchmark: pd.Series) -> float:
        return _metrics.up_capture(self._obj, benchmark)

    def down_capture(self, benchmark: pd.Series) -> float:
        return _metrics.down_capture(self._obj, benchmark)

    def capture_ratio(self, benchmark: pd.Series) -> float:
        return _metrics.capture_ratio(self._obj, benchmark)

    def treynor_ratio(
        self,
        benchmark: pd.Series,
        *,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
    ) -> float:
        return _metrics.treynor_ratio(
            self._obj,
            benchmark,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
        )

    # --- calendar period ----------------------------------------------------
    def monthly_returns(self) -> pd.DataFrame:
        return _metrics.monthly_returns(self._obj)

    def yearly_returns(self, *, benchmark: pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """End-of-year compounded returns.

        Returns a :class:`pd.Series` when no benchmark is supplied, or a
        two-column :class:`pd.DataFrame` (``<benchmark>``, ``<self>``)
        when one is provided.
        """
        strategy = _metrics.yearly_returns(self._obj)
        if benchmark is None:
            return strategy
        strategy = strategy.rename(
            str(self._obj.name) if self._obj.name is not None else "strategy"
        )
        bench_name = str(benchmark.name) if benchmark.name is not None else "benchmark"
        bench_yearly = _metrics.yearly_returns(benchmark).rename(bench_name)
        return pd.concat([bench_yearly, strategy], axis=1)

    def best_month(self) -> float:
        return _metrics.best_month(self._obj)

    def worst_month(self) -> float:
        return _metrics.worst_month(self._obj)

    def best_year(self) -> float:
        return _metrics.best_year(self._obj)

    def worst_year(self) -> float:
        return _metrics.worst_year(self._obj)

    def positive_months(self) -> int:
        return _metrics.positive_months(self._obj)

    def negative_months(self) -> int:
        return _metrics.negative_months(self._obj)

    # --- rolling ------------------------------------------------------------
    def rolling_sharpe(
        self,
        *,
        window: int = 63,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
    ) -> pd.Series:
        return _metrics.rolling_sharpe(
            self._obj,
            window=window,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
        )

    def rolling_sortino(
        self,
        *,
        window: int = 63,
        target: float = 0.0,
        periods_per_year: int | None = None,
    ) -> pd.Series:
        return _metrics.rolling_sortino(
            self._obj,
            window=window,
            target=target,
            periods_per_year=periods_per_year,
        )

    def rolling_volatility(
        self, *, window: int = 63, periods_per_year: int | None = None
    ) -> pd.Series:
        return _metrics.rolling_volatility(
            self._obj, window=window, periods_per_year=periods_per_year
        )

    def rolling_beta(self, benchmark: pd.Series, *, window: int = 63) -> pd.Series:
        return _metrics.rolling_beta(self._obj, benchmark, window=window)

    def rolling_drawdown(self) -> pd.Series:
        return _metrics.rolling_drawdown(self._obj)

    # --- one-shot bundle ----------------------------------------------------
    def metrics(
        self,
        *,
        benchmark: pd.Series | None = None,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.Series:
        """Return ~55 standard metrics in a single :class:`pd.Series`.

        This is the canonical 'show me everything' call. Pass
        ``benchmark=`` for alpha, beta, capture ratios, etc.

        Examples
        --------
        >>> import pandas as pd, numpy as np
        >>> import fundcloud  # noqa: F401
        >>> rng = np.random.default_rng(0)
        >>> r = pd.Series(rng.normal(0, 0.01, 252),
        ...               index=pd.date_range("2024-01-02", periods=252, freq="B"))
        >>> m = r.fc.metrics()
        >>> set(["sharpe", "sortino", "max_drawdown", "cvar"]).issubset(m.index)
        True
        """
        return _metrics.metrics(
            self._obj,
            benchmark=benchmark,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        )

    def period_returns(
        self,
        *,
        benchmark: pd.Series | None = None,
        periods_per_year: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """MTD / 3M / 6M / YTD / 1Y / 3Y / 5Y / 10Y / All-time return table.

        See :func:`fundcloud.metrics.period_returns`.
        """
        return _metrics.period_returns(
            self._obj, benchmark=benchmark, periods_per_year=periods_per_year
        )

    # --- EDA ----------------------------------------------------------------
    def describe(
        self,
        *,
        percentiles: list[float] | None = None,
        include_finance: bool = True,
        output: str | object | None = None,
        title: str | None = None,
    ) -> pd.DataFrame:
        """Super-set of :meth:`pandas.Series.describe` + finance extras.

        See :func:`fundcloud.explore.describe`. Returns a DataFrame even for
        Series input so the shape stays consistent.
        """
        from fundcloud.explore import describe as _describe

        return _describe(
            self._obj,
            percentiles=percentiles,
            include_finance=include_finance,
            output=output,
            title=title,
        )

    def profile(self, **kw: object) -> object:
        """Build a :class:`~fundcloud.explore.ProfileReport` for this Series."""
        from fundcloud.explore import profile as _profile

        return _profile(self._obj.to_frame(), **kw)  # type: ignore[arg-type]

    # --- report renderers ---------------------------------------------------
    def render_html(
        self,
        path: object = None,
        *,
        benchmark: pd.Series | None = None,
        title: str | None = None,
        **ts_kwargs: object,
    ) -> object:
        """Render this returns Series as an HTML tear sheet."""
        from fundcloud.accessors._helpers import portfolio_from_frame
        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(self._obj, benchmark=benchmark)
        return Tearsheet(portfolio, benchmark=benchmark, title=title, **ts_kwargs).render_html(path)

    def render_pdf(
        self,
        path: object,
        *,
        engine: object = None,
        benchmark: pd.Series | None = None,
        title: str | None = None,
        **ts_kwargs: object,
    ) -> object:
        """Render this returns Series as a PDF tear sheet."""
        from fundcloud.accessors._helpers import portfolio_from_frame
        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(self._obj, benchmark=benchmark)
        return Tearsheet(portfolio, benchmark=benchmark, title=title, **ts_kwargs).render_pdf(
            path, engine=engine
        )

    def render_excel(
        self,
        path: object,
        *,
        benchmark: pd.Series | None = None,
        title: str | None = None,
        **ts_kwargs: object,
    ) -> object:
        """Render this returns Series as an Excel workbook with native charts."""
        from fundcloud.accessors._helpers import portfolio_from_frame
        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(self._obj, benchmark=benchmark)
        return Tearsheet(portfolio, benchmark=benchmark, title=title, **ts_kwargs).render_excel(
            path
        )

    # --- plots --------------------------------------------------------------
    def plot_cumulative(self, **kw: object) -> object:
        from fundcloud.plots import cumulative

        return cumulative(self._obj, **kw)

    def plot_drawdown(self, **kw: object) -> object:
        from fundcloud.plots import drawdown

        return drawdown(self._obj, **kw)

    def plot_rolling_sharpe(self, *, window: int = 63, **kw: object) -> object:
        from fundcloud.plots import rolling_sharpe

        return rolling_sharpe(self._obj, window=window, **kw)

    def plot_return_distribution(self, **kw: object) -> object:
        from fundcloud.plots import return_distribution

        return return_distribution(self._obj, **kw)

    def plot_monthly_heatmap(self, **kw: object) -> object:
        from fundcloud.plots import monthly_heatmap

        return monthly_heatmap(self._obj, **kw)

    def plot_yearly_returns(self, *, benchmark: pd.Series | None = None, **kw: object) -> object:
        """Plotly EOY-returns paired bar chart (strategy vs optional benchmark)."""
        from fundcloud.plots import yearly_returns_bars

        return yearly_returns_bars(self._obj, benchmark=benchmark, **kw)

    def plot_summary(
        self,
        *,
        benchmark: pd.Series | None = None,
        theme: str | None = None,
        title: str | None = None,
    ) -> object:
        """Composite summary figure (cumulative, drawdown, rolling Sharpe, …)
        for this Series; delegates to :func:`fundcloud.plots.summary`."""
        from fundcloud.plots import summary

        return summary(self._obj, benchmark=benchmark, theme=theme, title=title)

    # --- conversions --------------------------------------------------------
    def to_returns(self, *, method: str = "simple", dropna: bool = True) -> pd.Series:
        result = _bars.to_returns(self._obj, method=method, dropna=dropna)  # type: ignore[arg-type]
        if not isinstance(result, pd.Series):
            raise TypeError(f"Expected Series, got {type(result).__name__}")
        return result
