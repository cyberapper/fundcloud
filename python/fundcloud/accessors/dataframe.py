"""``.fc`` accessor on :class:`pandas.DataFrame`.

Six surfaces, all one-liners into the rest of Fundcloud:

* **Metrics** — ``sharpe``, ``sortino``, ``metrics``, ``summary`` …
* **EDA** — ``describe``, ``profile``, ``compare``
* **Report renderers** — ``render_html``, ``render_pdf``, ``render_excel``
* **Simulator** — ``run_strategy`` / ``run_weights`` / ``run_signals`` /
  ``run_orders`` + preset shortcuts ``run_hold`` / ``run_dca`` + the
  dispatching ``simulate(what)``
* **Plots** — ``plot_cumulative``, ``plot_drawdown``, ``plot_monthly_heatmap`` …
* **Conversions** — ``to_prices``, ``to_returns``

Examples
--------
>>> import pandas as pd
>>> import fundcloud  # registers the accessor
>>> returns = pd.DataFrame({"a": [0.01, -0.005, 0.008]})
>>> returns.fc.sharpe()                       # pd.Series per column   # doctest: +SKIP
>>> returns.fc.render_html("out.html")        # turns returns into tear sheet  # doctest: +SKIP
>>> bars.fc.run_dca(500, horizon="weekly",    # one-line DCA backtest  # doctest: +SKIP
...                  weights={"SPY": 1.0})    # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from fundcloud import metrics as _metrics
from fundcloud._benchmark import resolve_benchmark as _resolve_benchmark
from fundcloud.accessors._helpers import (
    as_sim_kwargs,
    portfolio_from_frame,
    portfolios_per_column,
    require_bars_frame,
)
from fundcloud.data import bars as _bars

if TYPE_CHECKING:  # pragma: no cover
    from fundcloud.explore import ProfileReport
    from fundcloud.sim.simulator import SimResult
    from fundcloud.strategies.base import BaseStrategy

__all__ = ["DataFrameAccessor"]


@pd.api.extensions.register_dataframe_accessor("fc")
class DataFrameAccessor:
    """Fundcloud accessor namespace for a ``pd.DataFrame``."""

    def __init__(self, obj: pd.DataFrame) -> None:
        self._obj = obj

    # ====================================================== metrics (scalar)
    def sharpe(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> pd.Series:
        return _metrics.sharpe(self._obj, risk_free=risk_free, periods_per_year=periods_per_year)

    def sortino(self, *, target: float = 0.0, periods_per_year: int | None = None) -> pd.Series:
        return _metrics.sortino(self._obj, target=target, periods_per_year=periods_per_year)

    def calmar(self, *, periods_per_year: int | None = None) -> pd.Series:
        return _metrics.calmar(self._obj, periods_per_year=periods_per_year)

    def omega(self, *, target: float = 0.0) -> pd.Series:
        return _metrics.omega(self._obj, target=target)

    def max_drawdown(self) -> pd.Series:
        return _metrics.max_drawdown(self._obj)

    def drawdown_series(self) -> pd.DataFrame:
        return _metrics.drawdown_series(self._obj)

    def ulcer_index(self) -> pd.Series:
        return _metrics.ulcer_index(self._obj)

    def cvar(self, *, alpha: float = 0.95) -> pd.Series:
        return _metrics.cvar(self._obj, alpha=alpha)

    def value_at_risk(self, *, alpha: float = 0.95) -> pd.Series:
        return _metrics.value_at_risk(self._obj, alpha=alpha)

    def summary(
        self,
        *,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.DataFrame:
        """Compact metric-by-strategy table (11 core rows)."""
        return _metrics.returns_stats(
            self._obj,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        )

    def metrics(
        self,
        *,
        benchmark: pd.Series | None = None,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.DataFrame:
        """Full metric-by-strategy table (~55 rows, every ``fundcloud.metrics`` field).

        Pass ``benchmark=`` for alpha / beta / capture ratios per column.
        Compare with :meth:`summary` when you only need the classic rows.
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

    def yearly_returns(self, *, benchmark: pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """End-of-year compounded returns, optionally alongside a benchmark column."""
        if isinstance(self._obj, pd.Series):
            strategy = _metrics.yearly_returns(self._obj)
        else:
            # DataFrame: one column per strategy.
            strategy = pd.concat(
                {str(c): _metrics.yearly_returns(self._obj[c]) for c in self._obj.columns},
                axis=1,
            )
        if benchmark is None:
            return strategy
        bench_name = str(benchmark.name) if benchmark.name is not None else "benchmark"
        bench_yearly = _metrics.yearly_returns(benchmark).rename(bench_name)
        if isinstance(strategy, pd.Series):
            return pd.concat([bench_yearly, strategy.rename(self._obj.name or "strategy")], axis=1)
        return pd.concat([bench_yearly, strategy], axis=1)

    # ========================================================== EDA
    def describe(
        self,
        *,
        percentiles: list[float] | None = None,
        include_finance: bool = True,
        output: str | Path | None = None,
        title: str | None = None,
    ) -> pd.DataFrame:
        """Super-set of :meth:`pandas.DataFrame.describe` + finance extras.

        See :func:`fundcloud.explore.describe` for details.
        """
        from fundcloud.explore import describe as _describe

        return _describe(
            self._obj,
            percentiles=percentiles,
            include_finance=include_finance,
            output=output,
            title=title,
        )

    def profile(
        self,
        *,
        output: str | Path | None = None,
        title: str | None = None,
        sample_rows: int = 5_000,
        embed_plotlyjs: bool = False,
    ) -> ProfileReport:
        """Build a :class:`~fundcloud.explore.ProfileReport`.

        See :func:`fundcloud.explore.profile`.
        """
        from fundcloud.explore import profile as _profile

        return _profile(
            self._obj,
            output=output,
            title=title,
            sample_rows=sample_rows,
            embed_plotlyjs=embed_plotlyjs,
        )

    def compare(
        self,
        other: pd.DataFrame,
        *,
        output: str | Path | None = None,
        names: tuple[str, str] = ("a", "b"),
        target: str | None = None,
        title: str | None = None,
        embed_plotlyjs: bool = False,
    ) -> Path | str:
        """Drift report between ``self`` and ``other``.

        See :func:`fundcloud.explore.compare`.
        """
        from fundcloud.explore import compare as _compare

        return _compare(
            self._obj,
            other,
            output=output,
            names=names,
            target=target,
            title=title,
            embed_plotlyjs=embed_plotlyjs,
        )

    # ========================================================== report renderers
    def _is_multi_asset(self, weights: Any) -> bool:
        """Return True when the frame should render one tear sheet per column.

        Multi-column + no explicit ``weights`` = user wants per-asset
        rendering. Explicit ``weights`` switches to the equal-weight-combine
        path so existing callers keep working.
        """
        return self._obj.shape[1] > 1 and weights is None

    def _prepare_frame_and_benchmark(
        self, benchmark: pd.Series | str | None
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Resolve a string benchmark to the column series and drop it from the
        per-asset iteration frame so ``render_*`` doesn't render ``SPY vs SPY``.
        """
        resolved = _resolve_benchmark(self._obj, benchmark)
        frame = self._obj
        if isinstance(benchmark, str) and benchmark in frame.columns:
            if frame.shape[1] == 1:
                msg = (
                    f"benchmark={benchmark!r} is the only column in this "
                    f"returns frame — there's no strategy series left to "
                    f"render against. Drop ``benchmark=`` to render "
                    f"{benchmark!r} as a stand-alone tear sheet, or pass a "
                    f"benchmark Series from a different source."
                )
                raise ValueError(msg)
            frame = frame.drop(columns=[benchmark])
        return frame, resolved

    def render_html(
        self,
        path: str | Path | None = None,
        *,
        benchmark: pd.Series | str | None = None,
        weights: pd.Series | Mapping[str, float] | None = None,
        title: str | None = None,
        **ts_kwargs: Any,
    ) -> str | Path:
        """Render an HTML tear sheet.

        * Single column or explicit ``weights=`` → one combined tear sheet.
        * Multi-column without ``weights=`` → tabbed report, one tear sheet
          per column.
        * ``benchmark=`` accepts a :class:`pandas.Series` or the name of a
          column in this DataFrame; passing a column name pulls it out and
          excludes it from the per-asset iteration.

        ``path`` is ``None`` → returns the HTML string; otherwise writes
        the file and returns the :class:`pathlib.Path`.
        """
        frame, bench = self._prepare_frame_and_benchmark(benchmark)
        if frame.shape[1] > 1 and weights is None:
            from fundcloud.reports import multi as _multi

            portfolios = portfolios_per_column(frame, benchmark=bench)
            return _multi.render_html(portfolios, title=title, benchmark=bench, path=path)

        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(frame, benchmark=bench, weights=weights)
        return Tearsheet(portfolio, benchmark=bench, title=title, **ts_kwargs).render_html(path)

    def render_pdf(
        self,
        path: str | Path,
        *,
        engine: Literal["matplotlib", "weasyprint"] | None = None,
        benchmark: pd.Series | str | None = None,
        weights: pd.Series | Mapping[str, float] | None = None,
        title: str | None = None,
        **ts_kwargs: Any,
    ) -> Path:
        """Render a PDF tear sheet. Multi-column frames get per-asset sections.

        ``benchmark=`` accepts a :class:`pandas.Series` or a column name;
        string resolution happens before the multi-asset split.
        """
        frame, bench = self._prepare_frame_and_benchmark(benchmark)
        if frame.shape[1] > 1 and weights is None:
            from fundcloud.reports import multi as _multi

            portfolios = portfolios_per_column(frame, benchmark=bench)
            return _multi.render_pdf(
                portfolios, path=path, title=title, benchmark=bench, engine=engine
            )

        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(frame, benchmark=bench, weights=weights)
        return Tearsheet(portfolio, benchmark=bench, title=title, **ts_kwargs).render_pdf(
            path, engine=engine
        )

    def render_excel(
        self,
        path: str | Path,
        *,
        benchmark: pd.Series | str | None = None,
        weights: pd.Series | Mapping[str, float] | None = None,
        title: str | None = None,
        **ts_kwargs: Any,
    ) -> Path:
        """Render an Excel workbook. Multi-column frames get a per-asset sheet pair.

        ``benchmark=`` accepts a :class:`pandas.Series` or a column name;
        passing a column name adds a ``Benchmark`` sheet and drops that
        column from the per-asset iteration.
        """
        frame, bench = self._prepare_frame_and_benchmark(benchmark)
        if frame.shape[1] > 1 and weights is None:
            from fundcloud.reports import multi as _multi

            portfolios = portfolios_per_column(frame, benchmark=bench)
            return _multi.render_excel(portfolios, path=path, title=title, benchmark=bench)

        from fundcloud.reports import Tearsheet

        portfolio = portfolio_from_frame(frame, benchmark=bench, weights=weights)
        return Tearsheet(portfolio, benchmark=bench, title=title, **ts_kwargs).render_excel(path)

    # ========================================================== simulator
    def run_strategy(self, strategy: BaseStrategy, **sim_kwargs: Any) -> SimResult:
        """Backtest any :class:`BaseStrategy` on this Bars frame."""
        require_bars_frame(self._obj, operation="run_strategy")
        from fundcloud.sim import Simulator

        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_strategy(strategy)

    def run_weights(self, target_weights: pd.DataFrame, **sim_kwargs: Any) -> SimResult:
        """Backtest a target-weights path on this Bars frame."""
        require_bars_frame(self._obj, operation="run_weights")
        from fundcloud.sim import Simulator

        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_weights(target_weights)

    def run_signals(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        *,
        size: float = 1.0,
        **sim_kwargs: Any,
    ) -> SimResult:
        """Backtest boolean ``entries`` / ``exits`` panels on this Bars frame."""
        require_bars_frame(self._obj, operation="run_signals")
        from fundcloud.sim import Simulator

        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_signals(
            entries, exits, size=size
        )

    def run_orders(self, orders: pd.DataFrame, **sim_kwargs: Any) -> SimResult:
        """Backtest an explicit orders log (``ts/asset/side/qty``) on this Bars frame."""
        require_bars_frame(self._obj, operation="run_orders")
        from fundcloud.sim import Simulator

        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_orders(orders)

    def run_hold(
        self,
        weights: Mapping[str, float] | pd.Series | Any | None = None,
        *,
        rebalance: Any = None,
        start: pd.Timestamp | str | None = None,
        **sim_kwargs: Any,
    ) -> SimResult:
        """Backtest a buy-and-hold strategy on this Bars frame.

        Builds a :class:`~fundcloud.strategies.Hold` instance and runs it
        through :class:`~fundcloud.sim.Simulator`. The simulator buys the
        target allocation on the first executable bar and (optionally)
        rebalances back to it on each cadence boundary.

        Parameters
        ----------
        weights
            Target allocation. Accepts a ``Mapping[str, float]``, a
            :class:`pandas.Series`, or a callable receiving the bars
            frame and returning such a mapping. Weights must sum to 1.
            Default ``None`` means **equal weights** across every asset
            in the bars frame — useful for a quick equal-weight
            baseline without enumerating the universe.
        rebalance
            Optional :class:`~fundcloud.strategies.RebalanceSpec`
            (``horizon``, ``tolerance``). When supplied, the simulator
            restores target weights at each cadence boundary, skipping
            bars where every asset is already within ``tolerance`` of
            its target.
        start
            Optional lock-out: don't place the first allocation before
            this timestamp.

        Other Parameters
        ----------------
        **sim_kwargs
            Forwarded to :class:`~fundcloud.sim.Simulator`. The defaults
            shown matter for any meaningful backtest:

            * ``cash`` — starting capital (default ``1_000_000.0``).
            * ``costs`` — :class:`~fundcloud.sim.CostModel`. Default is
              :class:`~fundcloud.sim.FixedBps` at 5 bps per fill.
            * ``slippage`` — :class:`~fundcloud.sim.SlippageModel`.
              Default :class:`~fundcloud.sim.NoSlippage`.
            * ``execution`` — :class:`~fundcloud.sim.ExecutionModel`.
              Default :class:`~fundcloud.sim.NextBarOpen` (orders fill
              at the next bar's open price).

        Returns
        -------
        SimResult
            Container with ``portfolio`` (full
            :class:`~fundcloud.portfolio.Portfolio`), ``equity_curve``,
            ``trades``, and ``orders``. Use ``result.pf`` as a shortcut
            for ``result.portfolio``.

        Notes
        -----
        ``self`` must be a Bars frame: a :class:`pandas.DataFrame` with
        ``(field, symbol)`` MultiIndex columns containing the OHLCV
        fields and a sorted :class:`pandas.DatetimeIndex`. Single-asset
        flat-column frames are also accepted.

        See Also
        --------
        run_dca : Dollar-cost-averaging preset.
        run_strategy : Run any custom :class:`BaseStrategy`.
        fundcloud.strategies.Hold : Underlying strategy class.
        fundcloud.strategies.RebalanceSpec : Rebalance cadence + tolerance.
        fundcloud.sim.Simulator : Backtest engine.

        Examples
        --------
        Equal-weight default — no ``weights`` argument needed:

        >>> bars.fc.run_hold()                                   # doctest: +SKIP

        Explicit 60/40 buy-and-hold on the default 1M cash pool:

        >>> bars.fc.run_hold({"SPY": 0.6, "AGG": 0.4})           # doctest: +SKIP

        Quarterly-rebalanced 60/40 with a 5 % drift tolerance and a
        100k starting balance:

        >>> from fundcloud.strategies import RebalanceSpec
        >>> bars.fc.run_hold(                                    # doctest: +SKIP
        ...     {"SPY": 0.6, "AGG": 0.4},
        ...     rebalance=RebalanceSpec("91D", 0.05),
        ...     cash=100_000,
        ... )
        """
        require_bars_frame(self._obj, operation="run_hold")
        from fundcloud.sim import Simulator
        from fundcloud.strategies import Hold

        strategy = Hold(weights=weights, rebalance=rebalance, start=start)
        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_strategy(strategy)

    def run_dca(
        self,
        amount: float | Mapping[str, float] | None = None,
        *,
        amount_pct: float | Mapping[str, float] | None = None,
        horizon: str = "monthly",
        weights: Mapping[str, float] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        sell_on_end: bool = False,
        **sim_kwargs: Any,
    ) -> SimResult:
        """Backtest a dollar-cost-averaging strategy on this Bars frame.

        Builds a :class:`~fundcloud.strategies.DCA` instance and runs it
        through :class:`~fundcloud.sim.Simulator`. On every fire of the
        cadence, DCA places buy orders sized either by a fixed dollar
        amount (``amount=``) or by a fraction of current equity
        (``amount_pct=``).

        Exactly one of ``amount`` or ``amount_pct`` must be supplied.

        Parameters
        ----------
        amount
            Dollars per fire. Either a scalar (split across ``weights``)
            or a mapping ``asset -> dollars`` (per-asset buckets bypass
            ``weights`` entirely). Mutually exclusive with
            ``amount_pct``.
        amount_pct
            Equity fraction per fire. Either a scalar in ``[0, 1]``
            (split across ``weights``) or a mapping
            ``asset -> fraction``. The dollar size is recomputed on
            every fire from the **current** ``Portfolio.equity_curve``,
            so the deposit grows or shrinks with the portfolio. On the
            very first fire (no equity history yet) it falls back to
            starting cash.
            Mutually exclusive with ``amount``.
        horizon
            Cadence — ``"daily"``, ``"weekly"`` (every 7 calendar days
            from the anchor), ``"monthly"`` (same day-of-month as the
            anchor, snapped to the most recent trading bar), or a
            :class:`~fundcloud.strategies.scheduler.Cadence` for
            arbitrary steps. Default ``"monthly"``.
        weights
            Optional. When omitted with a scalar ``amount`` /
            ``amount_pct``, DCA spreads the deposit equally across every
            asset in the bars frame. Provide an explicit mapping
            (fractions summing to 1) to weight unevenly.
        start, end
            Optional window inside which DCA fires.
        sell_on_end
            When ``True``, close all positions on the bar after the last
            fire (after ``end``). Useful for clean comparisons that need
            to liquidate at the window edge.

        Other Parameters
        ----------------
        **sim_kwargs
            Forwarded to :class:`~fundcloud.sim.Simulator`. The defaults
            shown matter for any meaningful backtest:

            * ``cash`` — starting capital (default ``1_000_000.0``).
            * ``costs`` — :class:`~fundcloud.sim.CostModel`. Default is
              :class:`~fundcloud.sim.FixedBps` at 5 bps per fill.
            * ``slippage`` — :class:`~fundcloud.sim.SlippageModel`.
              Default :class:`~fundcloud.sim.NoSlippage`.
            * ``execution`` — :class:`~fundcloud.sim.ExecutionModel`.
              Default :class:`~fundcloud.sim.NextBarOpen` (orders fill
              at the next bar's open price).

        Returns
        -------
        SimResult
            Container with ``portfolio`` (full
            :class:`~fundcloud.portfolio.Portfolio`), ``equity_curve``,
            ``trades``, and ``orders``. Use ``result.pf`` as a shortcut
            for ``result.portfolio``.

        Notes
        -----
        ``self`` must be a Bars frame: a :class:`pandas.DataFrame` with
        ``(field, symbol)`` MultiIndex columns containing the OHLCV
        fields and a sorted :class:`pandas.DatetimeIndex`. Single-asset
        flat-column frames are also accepted.

        See Also
        --------
        run_hold : Buy-and-hold preset.
        run_strategy : Run any custom :class:`BaseStrategy`.
        fundcloud.strategies.DCA : Underlying strategy class.
        fundcloud.sim.Simulator : Backtest engine.

        Examples
        --------
        Single-asset weekly DCA on the default 1M cash pool:

        >>> bars.fc.run_dca(500, horizon="weekly",               # doctest: +SKIP
        ...                   weights={"SPY": 1.0})

        Equal-weight split across every asset in the bars frame —
        no ``weights`` needed:

        >>> bars.fc.run_dca(500, horizon="weekly")               # doctest: +SKIP

        Percentage of current equity instead of fixed dollars — deploy
        1 % of the portfolio each month, scaling automatically as
        equity grows:

        >>> bars.fc.run_dca(amount_pct=0.01, horizon="monthly")  # doctest: +SKIP

        Per-asset dollar buckets (``weights`` ignored), 100k starting
        balance, 10 bps fixed-bps costs:

        >>> from fundcloud.sim import FixedBps
        >>> bars.fc.run_dca(                                     # doctest: +SKIP
        ...     amount={"SPY": 300, "AGG": 200},
        ...     horizon="monthly",
        ...     cash=100_000,
        ...     costs=FixedBps(10),
        ... )
        """
        require_bars_frame(self._obj, operation="run_dca")
        from fundcloud.sim import Simulator
        from fundcloud.strategies import DCA

        strategy = DCA(
            amount=amount,
            amount_pct=amount_pct,
            horizon=horizon,
            weights=weights,
            start=start,
            end=end,
            sell_on_end=sell_on_end,
        )
        return Simulator(self._obj, **as_sim_kwargs(sim_kwargs)).run_strategy(strategy)

    def simulate(self, what: Any, **sim_kwargs: Any) -> SimResult:
        """Type-dispatching shortcut for ``run_strategy`` / ``run_weights`` / ``run_orders``.

        * :class:`BaseStrategy` subclass → ``run_strategy``
        * :class:`pd.DataFrame` with ``(ts, asset, side, qty)`` columns → ``run_orders``
        * ``pd.DataFrame`` of bool → ``run_signals`` (with ``exits=~entries``)
        * Otherwise ``pd.DataFrame`` of floats → ``run_weights``
        """
        from fundcloud.strategies.base import BaseStrategy

        if isinstance(what, BaseStrategy):
            return self.run_strategy(what, **sim_kwargs)
        if isinstance(what, pd.DataFrame):
            orders_cols = {"ts", "asset", "side", "qty"}
            if orders_cols.issubset(set(what.columns)):
                return self.run_orders(what, **sim_kwargs)
            if what.dtypes.eq(bool).all():
                # exits inferred as ~entries when a bool DataFrame is passed
                return self.run_signals(what, ~what, **sim_kwargs)
            return self.run_weights(what, **sim_kwargs)
        msg = (
            "simulate() expects a BaseStrategy or a DataFrame (orders log / "
            "bool signals / target weights); got "
            f"{type(what).__name__}."
        )
        raise TypeError(msg)

    # ========================================================== patterns
    def detect_pattern(self, pattern: Any, **params: Any) -> pd.DataFrame:
        """Run a registered pattern detector on this Bars frame.

        Returns a wide signal panel — one column per asset — matching
        the indicator's :attr:`signal_mode` (``BREAKOUT`` by default).

        Examples
        --------
        >>> bars.fc.detect_pattern("head_and_shoulders").shape  # doctest: +SKIP
        >>> bars.fc.detect_pattern(Pattern.DOUBLE_BOTTOM, min_quality=70)  # doctest: +SKIP
        """
        require_bars_frame(self._obj, operation="detect_pattern")
        indicator = _resolve_pattern_indicator(pattern, params)
        return indicator.fit_transform(self._obj)

    def pattern_events(self, pattern: Any, **params: Any) -> pd.DataFrame:
        """Rich event log for the named pattern: timestamps, pivots, target,
        stop, quality, variant.

        See :data:`fundcloud.features.patterns.EVENTS_COLUMNS` for the
        canonical column order.
        """
        require_bars_frame(self._obj, operation="pattern_events")
        indicator = _resolve_pattern_indicator(pattern, params)
        return indicator.events(self._obj)

    def evaluate_pattern(
        self,
        pattern: Any,
        *,
        horizons: tuple[int, ...] = (5, 10, 20, 60),
        atr_window: int = 14,
        baseline: bool = True,
        trade_direction: str = "natural",
        condition: Any = None,
        **params: Any,
    ) -> pd.DataFrame:
        """Headline feature-quality panel for the pattern.

        Delegates to :func:`fundcloud.metrics.feature_quality.evaluate`
        after running the indicator. Use ``trade_direction='inverse'``
        to test fading the pattern. Pass ``condition`` (a
        :class:`PatternCondition`) to grade R-multiples against the
        condition's target / stop instead of the 1×ATR fallback.
        """
        require_bars_frame(self._obj, operation="evaluate_pattern")
        from fundcloud.metrics import feature_quality as fq

        indicator = _resolve_pattern_indicator(pattern, params)
        events = indicator.events(self._obj)
        return fq.evaluate(
            events,
            self._obj,
            horizons=horizons,
            atr_window=atr_window,
            baseline=baseline,
            trade_direction=trade_direction,
            condition=condition,
        )

    def list_patterns(self) -> list[Any]:
        """Registered pattern enum values, sorted by stable name."""
        from fundcloud.features.patterns import Pattern

        return sorted(Pattern, key=lambda p: p.value)

    def plot_pattern_event(
        self,
        event: Any,
        *,
        padding: int = 20,
        show_levels: bool = True,
        horizon: int | None = 20,
        theme: str | None = None,
    ) -> Any:
        """Render a single detected pattern as an annotated candlestick chart.

        ``event`` is a row from the events table (``pd.Series`` from
        ``events.iloc[i]`` or a dict). Pivots are connected into the
        formation shape; trend lines and entry / target / stop levels
        overlaid; the formation window is shaded; ``horizon`` (default
        20) marks ``breakout_ts + horizon`` so the metric grading
        window is visible.
        """
        require_bars_frame(self._obj, operation="plot_pattern_event")
        from fundcloud.plots.patterns import plot_pattern_event

        return plot_pattern_event(
            event,
            self._obj,
            padding=padding,
            show_levels=show_levels,
            horizon=horizon,
            theme=theme,
        )

    def plot_patterns(
        self,
        pattern: Any,
        *,
        asset: str,
        max_events: int | None = None,
        horizon: int | None = None,
        show_horizon_for_top: int = 10,
        theme: str | None = None,
        **params: Any,
    ) -> Any:
        """Render every detection of ``pattern`` on ``asset`` on one
        candlestick chart, with formation shapes drawn (pivots
        connected by a coloured polyline). Useful for spotting
        clustering and regime shifts.
        """
        require_bars_frame(self._obj, operation="plot_patterns")
        from fundcloud.plots.patterns import plot_patterns_overview

        indicator = _resolve_pattern_indicator(pattern, params)
        events = indicator.events(self._obj)
        return plot_patterns_overview(
            events,
            self._obj,
            asset,
            max_events=max_events,
            horizon=horizon,
            show_horizon_for_top=show_horizon_for_top,
            theme=theme,
        )

    def plot_asset_patterns(
        self,
        asset: str,
        *,
        patterns: Any = None,
        min_quality: float = 50.0,
        horizon: int | None = None,
        show_horizon_for_top: int = 10,
        theme: str | None = None,
    ) -> Any:
        """Single chart for ``asset`` with every pattern's detections drawn,
        legend-toggled per pattern.

        Click any pattern in the legend to hide / show its formations.
        Each detection is drawn as a coloured polyline through its pivots
        so the formation shape is identifiable at a glance. Horizon
        shading is *off* by default — passing ``horizon=20`` enables it
        for the most-recent ``show_horizon_for_top`` events only, so
        the chart doesn't end up barcoded with hundreds of overlapping
        bands.
        """
        require_bars_frame(self._obj, operation="plot_asset_patterns")
        from fundcloud.plots.patterns import plot_asset_patterns

        return plot_asset_patterns(
            self._obj,
            asset,
            patterns=patterns,
            min_quality=min_quality,
            horizon=horizon,
            show_horizon_for_top=show_horizon_for_top,
            theme=theme,
        )

    def run_pattern(
        self,
        pattern: Any,
        *,
        condition: Any = None,
        size: float = 0.1,
        inverse: bool = False,
        **params: Any,
    ) -> SimResult:
        """Backtest the named pattern via :class:`PatternStrategy`.

        Examples
        --------
        >>> bars.fc.run_pattern(Pattern.DOUBLE_BOTTOM,                  # doctest: +SKIP
        ...                     condition=PatternCondition(...))

        Trade fade-the-pattern on a bearish detector::

        >>> bars.fc.run_pattern("double_top", inverse=True)             # doctest: +SKIP
        """
        require_bars_frame(self._obj, operation="run_pattern")
        from fundcloud.sim import Simulator
        from fundcloud.strategies import PatternStrategy

        indicator = _resolve_pattern_indicator(pattern, params)
        strategy = PatternStrategy(
            indicator,
            condition=condition,
            size=size,
            inverse=inverse,
        )
        return Simulator(self._obj).run_strategy(strategy)

    # ========================================================== plots
    def plot_cumulative(self, **kw: Any) -> Any:
        """Plotly cumulative-returns figure (delegates to :func:`fundcloud.plots.cumulative`)."""
        from fundcloud.plots import cumulative

        return cumulative(self._obj, **kw)

    def plot_drawdown(self, **kw: Any) -> Any:
        """Plotly drawdown figure."""
        from fundcloud.plots import drawdown

        return drawdown(self._obj, **kw)

    def plot_rolling_sharpe(self, *, window: int = 63, **kw: Any) -> Any:
        """Plotly rolling-Sharpe figure."""
        from fundcloud.plots import rolling_sharpe

        return rolling_sharpe(self._obj, window=window, **kw)

    def plot_return_distribution(self, **kw: Any) -> Any:
        """Plotly histogram of return distribution."""
        from fundcloud.plots import return_distribution

        return return_distribution(self._obj, **kw)

    def plot_monthly_heatmap(self, **kw: Any) -> Any:
        """Plotly year × month heatmap. Requires a single column or Series."""
        from fundcloud.plots import monthly_heatmap

        return monthly_heatmap(self._obj, **kw)

    def plot_composition(self, **kw: Any) -> Any:
        """Plotly composition (stacked weights) figure."""
        from fundcloud.plots import composition

        return composition(self._obj, **kw)

    def plot_yearly_returns(self, *, benchmark: pd.Series | None = None, **kw: Any) -> Any:
        """Plotly EOY-returns paired bar chart (strategy vs optional benchmark)."""
        from fundcloud.plots import yearly_returns_bars

        return yearly_returns_bars(self._obj, benchmark=benchmark, **kw)

    def plot_summary(
        self,
        *,
        benchmark: pd.Series | str | None = None,
        weights: pd.DataFrame | None = None,
        theme: str | None = None,
        title: str | None = None,
        heatmap_asset: str | None = None,
    ) -> Any:
        """One-liner composite summary figure for the frame's returns.

        Delegates to :func:`fundcloud.plots.summary`. ``benchmark=`` accepts
        a :class:`pandas.Series` or the name of one of this DataFrame's
        columns.
        """
        from fundcloud.plots import summary

        bench = _resolve_benchmark(self._obj, benchmark)
        frame = self._obj
        if isinstance(benchmark, str) and benchmark in frame.columns:
            frame = frame.drop(columns=[benchmark])
        return summary(
            frame,
            benchmark=bench,
            weights=weights,
            theme=theme,
            title=title,
            heatmap_asset=heatmap_asset,
        )

    # ========================================================== conversions
    def to_prices(self, *, field: str = "close") -> pd.DataFrame:
        return _bars.to_prices(self._obj, field=field)  # type: ignore[arg-type]

    def to_returns(
        self, *, field: str = "close", method: str = "simple", dropna: bool = True
    ) -> pd.DataFrame:
        result = _bars.to_returns(self._obj, field=field, method=method, dropna=dropna)  # type: ignore[arg-type]
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(result).__name__}")
        return result


def _resolve_pattern_indicator(pattern: Any, params: Mapping[str, Any]) -> Any:
    """Map a Pattern enum / stable string name to a constructed indicator.

    Accepts either ``Pattern.HEAD_AND_SHOULDERS`` or the stable
    snake_case string ``"head_and_shoulders"``. Looks up the registered
    indicator class and instantiates it with ``**params``. Raises
    ``ValueError`` with a helpful message for unknown patterns.
    """
    from fundcloud.features.indicators.base import _REGISTRY
    from fundcloud.features.patterns import Pattern

    name = pattern.value if isinstance(pattern, Pattern) else str(pattern)
    try:
        Pattern(name)
    except ValueError as e:
        valid = ", ".join(p.value for p in Pattern)
        msg = f"unknown pattern: {pattern!r}; valid: {valid}"
        raise ValueError(msg) from e
    if name not in _REGISTRY:
        msg = f"pattern {name!r} is not registered with @register_indicator"
        raise LookupError(msg)
    cls = _REGISTRY[name]
    return cls(**params)
