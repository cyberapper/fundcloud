"""``Portfolio`` — unified live + analytics container.

A ``Portfolio`` can be used in two modes:

1. **Analytics** — construct with ``returns=``, ``weights=``, ``benchmark=``
   and call ``sharpe``, ``max_drawdown``, etc. This is the entry point
   ``EqualWeighted().fit(X).predict(X)`` returns and matches skfolio's
   ``Portfolio`` constructor shape.
2. **Live** — construct with ``cash=``, ``positions=``, apply ``Trade``
   objects via :meth:`apply`, advance time with :meth:`mark_to_market`, and
   call :meth:`snapshot` at the end to freeze a copy for reporting. This is
   the mode the ``Simulator`` drives during a backtest.

Either mode produces an object that lines up with skfolio's
``Portfolio``; :meth:`from_skfolio` / :meth:`to_skfolio` provide a cheap
round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fundcloud import metrics as _metrics
from fundcloud.metrics.core import ReturnMethod

__all__ = ["Portfolio", "Position"]


@dataclass(slots=True)
class Position:
    """Live position for a single asset.

    Attributes
    ----------
    qty
        Signed share count. Positive for long, negative for short, zero
        for closed.
    avg_cost
        Volume-weighted average cost per share. Updated only when adding
        to an existing direction (or opening a new one); closes leave
        ``avg_cost`` alone so reporting can compute realised P&L on the
        original basis.
    sl_level
        Absolute stop-loss price for this position, or ``None`` for
        positions without a stop. Set by the simulator when an entry
        :class:`~fundcloud.sim.Order` carries an ``sl_stop`` fraction:
        for longs the level becomes ``trade_price * (1 - sl_stop)``,
        for shorts ``trade_price * (1 + sl_stop)``. Anchored to the
        latest fill's price (not ``avg_cost``) so accumulating entries
        tighten the stop relative to current price — the conservative
        choice for risk management. Cleared when ``qty`` returns to
        zero. Preserved on partial closes that leave the direction
        unchanged.
    tp_level
        Absolute take-profit price for this position, or ``None`` for
        positions without one. Mirror of ``sl_level``: set by the
        simulator when an entry :class:`~fundcloud.sim.Order` carries a
        ``tp_stop`` fraction. Long: ``trade_price * (1 + tp_stop)`` —
        the simulator fires when a subsequent bar's *high* pierces it.
        Short: ``trade_price * (1 - tp_stop)`` against bar *low*.
        Anchored to the latest fill, cleared on close, preserved on
        partial close. Coexists with ``sl_level`` and the trail
        (bracket order); any stop (fixed or trailing) beats
        take-profit on the same bar.
    tsl_pct
        Trailing-stop fraction in ``(0, 1)`` for this position, or
        ``None`` for positions without a trailing stop. Set on the
        *first* entry that carries ``tsl_stop`` and held constant
        thereafter — accumulating entries do not reset it. Combined
        with :attr:`tsl_anchor` to derive the active trail level on
        each bar (long: ``tsl_anchor * (1 - tsl_pct)``; short:
        ``tsl_anchor * (1 + tsl_pct)``). Cleared on close.
    tsl_anchor
        Running high-water mark for the trailing stop (long: peak price
        seen since the first entry, ratchets up only; short: trough
        price, ratchets down only). Initially the first entry's fill
        price. Updated by the simulator's intra-bar exit check via a
        two-step ratchet around the trigger:

        1. Before the trigger check, ratchet against ``bar.open`` if
           favourable (gap-up for long, gap-down for short).
        2. After the trigger check (only if the trail didn't fire),
           ratchet against ``bar.high`` (long) / ``bar.low`` (short)
           so the next bar sees the new high-water mark.

        Splitting the ratchet means a single wide-range bar can't
        tighten the level mid-bar to something the open never traded
        against — the trigger uses the level that was in force when
        the bar started. Accumulating entries do not reset the anchor;
        the trail tracks the high-water mark from the original entry.
        Cleared on close.
    """

    qty: float = 0.0
    avg_cost: float = 0.0
    sl_level: float | None = None
    tp_level: float | None = None
    tsl_pct: float | None = None
    tsl_anchor: float | None = None


@dataclass(slots=True)
class _LiveState:
    """Live state maintained during :meth:`Portfolio.apply` calls."""

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    equity_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)
    trade_log: list[Any] = field(default_factory=list)
    # Last known mark-to-market price per asset. Lets the portfolio value a
    # held position even when the current bar is a non-trading day for that
    # asset (e.g., TSLA on a Saturday in a BTC+TSLA panel).
    last_prices: dict[str, float] = field(default_factory=dict)


class Portfolio:
    """Unified position + analytics container."""

    def __init__(
        self,
        *,
        returns: pd.DataFrame | pd.Series | None = None,
        weights: pd.DataFrame | pd.Series | None = None,
        benchmark: pd.Series | None = None,
        cash: float = 0.0,
        positions: dict[str, float] | None = None,
        name: str | None = None,
    ) -> None:
        self._name = name
        self._benchmark = benchmark
        self._returns: pd.Series | None = None
        self._weights_frame: pd.DataFrame | None = None

        if returns is not None:
            self._returns = _coerce_returns(returns)
        if weights is not None:
            self._weights_frame = _coerce_weights(weights)

        # Live state — only populated for live-mode portfolios.
        self._live = _LiveState(cash=float(cash))
        if positions:
            for asset, qty in positions.items():
                self._live.positions[asset] = Position(qty=float(qty))

    # ------------------------------------------------------------------ naming

    @property
    def name(self) -> str:
        """Human-readable label for this portfolio (default ``"strategy"``)."""
        return self._name or "strategy"

    def rename(self, name: str) -> Portfolio:
        """Rename in place and return ``self`` for chaining.

        Renames the underlying ``returns`` Series too when present, so
        downstream concatenations into a panel pick up the new name.

        Parameters
        ----------
        name
            New label.

        Returns
        -------
        Portfolio
            ``self``, for chaining.
        """
        self._name = name
        if self._returns is not None:
            self._returns = self._returns.rename(name)
        return self

    # ------------------------------------------------------------------ live API

    @property
    def cash(self) -> float:
        """Current uninvested cash (live mode only)."""
        return self._live.cash

    def position(self, asset: str) -> Position:
        """Return the live :class:`Position` for ``asset`` (creating one if missing).

        Parameters
        ----------
        asset
            Asset ticker.

        Returns
        -------
        Position
            The mutable :class:`Position` object — same identity across
            calls, so callers can inspect ``qty`` / ``avg_cost`` later.
        """
        return self._live.positions.setdefault(asset, Position())

    @property
    def positions(self) -> pd.Series:
        """Current open positions as a Series keyed by asset."""
        return pd.Series({asset: p.qty for asset, p in self._live.positions.items()}, dtype=float)

    def apply(self, trade: Any) -> None:
        """Apply a fill to live state — cash, positions, average cost, log.

        Mutates the portfolio in place: subtracts notional + fee from
        cash, adds the signed quantity to the position, updates the
        volume-weighted average cost on adds, and appends to the
        internal trade log.

        Parameters
        ----------
        trade
            Anything with the duck-typed attributes ``asset`` (str),
            ``qty`` (signed float), ``price`` (float), ``fee`` (float;
            optional). Typically a :class:`fundcloud.sim.Trade`, but
            the contract is duck-typed so this module doesn't import
            the simulator package.

        Notes
        -----
        Average cost is recomputed only when the trade adds to an
        existing direction (or opens a new position). Trades that close
        or partially close a position leave ``avg_cost`` unchanged so
        downstream reporting can compute realised P&L on the original
        basis.

        Bracket-order bookkeeping (when the trade's underlying
        :class:`~fundcloud.sim.Order` carries ``sl_stop`` / ``tp_stop``
        / ``tsl_stop``):

        * Fixed ``sl_level`` / ``tp_level`` re-anchor to *this fill's
          price* on every accumulating add — tightens the bracket as
          the position scales up.
        * The trailing stop is initialised on the *first* entry that
          carries ``tsl_stop`` and held thereafter — accumulating adds
          do **not** reset :attr:`Position.tsl_pct` or
          :attr:`Position.tsl_anchor`. The high-water mark continues
          to ratchet from the original entry's price.
        * All four bracket fields are cleared when the position closes
          (``qty == 0``).
        """
        asset = str(trade.asset)
        qty = float(trade.qty)
        price = float(trade.price)
        fee = float(getattr(trade, "fee", 0.0))
        pos = self.position(asset)
        notional = qty * price
        self._live.cash -= notional + fee
        # Weighted-average cost update for adds; leave avg_cost alone on closes.
        is_add = pos.qty == 0 or (pos.qty > 0) == (qty > 0)
        if is_add:
            total = pos.qty + qty
            if total != 0:
                pos.avg_cost = (pos.qty * pos.avg_cost + qty * price) / total
        pos.qty += qty

        # Bracket-order bookkeeping (stop-loss + take-profit + trailing
        # stop). Each fraction is carried on the originating Order; the
        # simulator translates them to position state here so the per-bar
        # intra-bar exit check has nothing else to compute. Any
        # combination may be set on the same Order.
        #
        # Fixed SL/TP levels are anchored to *this trade's fill price* —
        # not the running ``avg_cost`` — so on an accumulating position
        # each new entry tightens the stop / take-profit relative to
        # current price. This is the conservative choice for risk
        # management.
        #
        # The trailing stop is different: it has its own running anchor
        # that ratchets bar-by-bar in the favourable direction. Once the
        # trail is active (``tsl_pct`` is non-None), accumulating entries
        # do **not** reset it — the high-water mark continues to track
        # from the *first* entry's fill price regardless of subsequent
        # adds. If a user wants per-add re-anchoring they should close
        # and re-open instead of accumulating.
        #
        # All bracket state is cleared when the position closes
        # (``qty == 0``), regardless of whether the closing trade carried
        # its own bracket fractions. A trade without any bracket set
        # leaves the existing state alone — useful when only some
        # entries in a multi-entry position should re-anchor SL/TP.
        order = getattr(trade, "order", None)
        sl_stop = getattr(order, "sl_stop", None)
        tp_stop = getattr(order, "tp_stop", None)
        tsl_stop = getattr(order, "tsl_stop", None)
        if pos.qty == 0:
            pos.sl_level = None
            pos.tp_level = None
            pos.tsl_pct = None
            pos.tsl_anchor = None
        else:
            if sl_stop is not None and is_add and price > 0:
                pos.sl_level = price * (1.0 - sl_stop) if pos.qty > 0 else price * (1.0 + sl_stop)
            if tp_stop is not None and is_add and price > 0:
                pos.tp_level = price * (1.0 + tp_stop) if pos.qty > 0 else price * (1.0 - tp_stop)
            if tsl_stop is not None and is_add and price > 0 and pos.tsl_pct is None:
                # First entry that carries ``tsl_stop`` — initialise the
                # trail. Subsequent accumulating entries leave the
                # anchor in place; the high-water mark keeps ratcheting
                # from the original entry's price.
                pos.tsl_pct = tsl_stop
                pos.tsl_anchor = price

        self._live.trade_log.append(trade)

    def mark_to_market(
        self,
        prices: pd.Series,
        ts: pd.Timestamp,
    ) -> float:
        """Compute and record equity at timestamp ``ts``.

        Walks every open position, marks it at the current bar's price
        (with fallbacks for missing quotes — see Notes), sums into cash,
        appends the equity snapshot and resulting weights to the
        portfolio's history. The simulator calls this once per bar
        after :meth:`apply`-ing any fills.

        Parameters
        ----------
        prices
            Asset → price at this bar. May contain ``NaN`` for assets
            that didn't trade (mixed-frequency panels: equities on
            weekends, etc.).
        ts
            Bar timestamp; used as the index value when recording the
            snapshot.

        Returns
        -------
        float
            Total equity at ``ts`` (cash + sum of position values).

        Notes
        -----
        Missing-price fallback chain (in order): ``prices[asset]`` →
        the last finite price seen for ``asset`` (cached across calls)
        → the position's ``avg_cost``. If none is positive and finite,
        the position contributes zero to equity for this bar.
        Cash-only positions (``qty == 0``) are skipped.
        """
        # Refresh the last-known price cache from this bar's quotes.
        for asset, raw in prices.items():
            px = float(raw)
            if np.isfinite(px) and px > 0:
                self._live.last_prices[str(asset)] = px

        equity = self._live.cash
        per_asset_value: dict[str, float] = {}
        for asset, pos in self._live.positions.items():
            if pos.qty == 0:
                continue
            # Prefer the current bar's price; fall back to the last known
            # price for that asset; final fallback is the position's average
            # cost (covers the unusual case where we buy and immediately
            # need to mark-to-market on a NaN bar).
            raw_px = prices.get(asset, np.nan)
            px = float(raw_px) if raw_px is not None else float("nan")
            if not np.isfinite(px) or px <= 0:
                px = self._live.last_prices.get(
                    asset, pos.avg_cost if pos.avg_cost > 0 else float("nan")
                )
            if not np.isfinite(px) or px <= 0:
                continue
            value = pos.qty * px
            equity += value
            per_asset_value[asset] = value
        self._live.equity_history.append((ts, equity))
        if equity != 0:
            weights = {a: v / equity for a, v in per_asset_value.items()}
        else:
            weights = {a: 0.0 for a in per_asset_value}
        self._live.weights_history.append((ts, weights))
        return equity

    def snapshot(self) -> Portfolio:
        """Freeze live state into an analytics-mode copy.

        Builds ``returns`` from the equity curve and ``weights`` from
        the recorded weights history, then detaches live state so the
        returned instance behaves immutably for analytics. Used by
        :class:`~fundcloud.sim.Simulator` to produce the
        :class:`~fundcloud.sim.SimResult.portfolio` field.

        Returns
        -------
        Portfolio
            Analytics-mode copy with ``returns`` / ``weights`` populated
            and live state detached. Calling :meth:`apply` on the result
            won't affect the original.
        """
        equity = pd.Series(
            {ts: val for ts, val in self._live.equity_history},
            dtype=float,
        ).sort_index()
        returns = equity.pct_change().dropna() if len(equity) > 1 else pd.Series([], dtype=float)

        weights_frame: pd.DataFrame | None
        if self._live.weights_history:
            raw = pd.DataFrame.from_dict(
                {ts: w for ts, w in self._live.weights_history}, orient="index"
            ).sort_index()
            weights_frame = raw.fillna(0.0)
        else:
            weights_frame = None

        snap = Portfolio(
            returns=returns,
            weights=weights_frame,
            benchmark=self._benchmark,
            name=self._name,
        )
        return snap

    # ----------------------------------------------------------------- views

    @property
    def returns(self) -> pd.Series:
        if self._returns is None:
            msg = (
                "Portfolio has no recorded returns. Either construct with "
                "`returns=`, or call `snapshot()` on a live portfolio first."
            )
            raise ValueError(msg)
        return self._returns

    @property
    def weights(self) -> pd.DataFrame | None:
        return self._weights_frame

    @property
    def benchmark(self) -> pd.Series | None:
        return self._benchmark

    @property
    def equity_curve(self) -> pd.Series:
        """Running equity. For analytics-mode portfolios, cumulates ``returns``."""
        if self._live.equity_history:
            return pd.Series(
                {ts: val for ts, val in self._live.equity_history}, dtype=float
            ).sort_index()
        if self._returns is None:
            return pd.Series([], dtype=float)
        return (1.0 + self._returns).cumprod()

    # ------------------------------------------------------------------ analytics

    def sharpe(
        self, *, risk_free: float | None = None, periods_per_year: int | None = None
    ) -> float:
        return _metrics.sharpe(self.returns, risk_free=risk_free, periods_per_year=periods_per_year)

    def sortino(self, *, target: float = 0.0, periods_per_year: int | None = None) -> float:
        return _metrics.sortino(self.returns, target=target, periods_per_year=periods_per_year)

    def calmar(self, *, periods_per_year: int | None = None) -> float:
        return _metrics.calmar(self.returns, periods_per_year=periods_per_year)

    def omega(self, *, target: float = 0.0) -> float:
        return _metrics.omega(self.returns, target=target)

    def max_drawdown(self) -> float:
        return _metrics.max_drawdown(self.returns)

    def drawdown_series(self) -> pd.Series:
        return _metrics.drawdown_series(self.returns)

    def cvar(self, *, alpha: float = 0.95) -> float:
        return _metrics.cvar(self.returns, alpha=alpha)

    def value_at_risk(self, *, alpha: float = 0.95) -> float:
        return _metrics.value_at_risk(self.returns, alpha=alpha)

    def ulcer_index(self) -> float:
        return _metrics.ulcer_index(self.returns)

    def turnover(self) -> float:
        """Average one-way turnover across rebalance boundaries.

        Returns ``0.0`` when weights are constant or unknown.
        """
        w = self._weights_frame
        if w is None or len(w) < 2:
            return 0.0
        return float(w.diff().abs().sum(axis=1).iloc[1:].mean() / 2.0)

    def attribution(self) -> pd.DataFrame:
        """Asset-level return contribution = weights × returns (shifted).

        Requires a weights frame. Uses the current-bar weight × current-bar
        asset return, which is the standard backward-looking decomposition.
        """
        w = self._weights_frame
        if w is None:
            return pd.DataFrame()
        if self._returns is None or self._returns.empty:
            return pd.DataFrame(columns=w.columns)
        # If returns is a total-portfolio series (no per-asset info), attribution
        # is undefined beyond `weights * total_return`.
        contrib = w.reindex(self._returns.index).mul(self._returns, axis=0)
        return contrib

    def contribution(self) -> pd.Series:
        """Average per-asset contribution to total return."""
        attr = self.attribution()
        if attr.empty:
            return pd.Series(dtype=float)
        return attr.mean()

    def summary(
        self,
        *,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.Series:
        """Single-column summary of key metrics (rows = metric names).

        Compact 11-metric view. For the full ~55-metric bundle use
        :meth:`metrics`.
        """
        r = self.returns
        stats = _metrics.returns_stats(
            r,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        )
        return stats.iloc[:, 0].rename(self.name)

    def metrics(
        self,
        *,
        benchmark: pd.Series | None = None,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.Series:
        """Full portfolio-metrics bundle.

        Delegates to :func:`fundcloud.metrics.metrics`. When this Portfolio
        was constructed with ``benchmark=``, that benchmark is used by
        default; pass an explicit ``benchmark=`` to override.
        """
        r = self.returns
        bench = benchmark if benchmark is not None else self.benchmark
        return _metrics.metrics(
            r,
            benchmark=bench,
            risk_free=risk_free,
            periods_per_year=periods_per_year,
            cvar_alpha=cvar_alpha,
        ).rename(self.name)

    def drawdown_details(self) -> pd.DataFrame:
        """One row per drawdown episode: start / valley / recovery + durations.

        See :func:`fundcloud.metrics.drawdown_details` for the column
        definitions.
        """
        return _metrics.drawdown_details(self.returns)

    def runup_details(self) -> pd.DataFrame:
        """One row per runup (rally) episode between drawdowns.

        See :func:`fundcloud.metrics.runup_details` for the column
        definitions.
        """
        return _metrics.runup_details(self.returns)

    def worst_drawdowns(self, top: int = 10) -> pd.DataFrame:
        """Top-``top`` drawdown episodes, display-formatted.

        Columns: ``Started`` / ``Recovered`` / ``Drawdown`` / ``Days``.
        Episodes are sorted by depth (worst first).
        """
        dd = _metrics.drawdown_details(self.returns)
        if dd.empty:
            return pd.DataFrame(columns=["Started", "Recovered", "Drawdown", "Days"])
        view = (
            dd
            .head(top)[["start", "recovery", "max_drawdown", "duration_days"]]
            .rename(
                columns={
                    "start": "Started",
                    "recovery": "Recovered",
                    "max_drawdown": "Drawdown",
                    "duration_days": "Days",
                }
            )
            .reset_index(drop=True)
        )
        return view

    def worst_runups(self, top: int = 10) -> pd.DataFrame:
        """Top-``top`` runup episodes, display-formatted.

        Columns: ``Started`` / ``Peaked`` / ``Runup`` / ``Days``.
        Episodes are sorted by magnitude (biggest first).
        """
        ru = _metrics.runup_details(self.returns)
        if ru.empty:
            return pd.DataFrame(columns=["Started", "Peaked", "Runup", "Days"])
        view = (
            ru
            .head(top)[["start", "peak", "max_runup", "duration_days"]]
            .rename(
                columns={
                    "start": "Started",
                    "peak": "Peaked",
                    "max_runup": "Runup",
                    "duration_days": "Days",
                }
            )
            .reset_index(drop=True)
        )
        return view

    def period_returns(
        self,
        *,
        benchmark: pd.Series | None = None,
        periods_per_year: int | None = None,
    ) -> pd.Series | pd.DataFrame:
        """MTD / 3M / 6M / YTD / 1Y / 3Y / 5Y / 10Y / All-time bundle.

        When a benchmark is not passed and :attr:`benchmark` was set on
        construction, it's used as the default. See
        :func:`fundcloud.metrics.period_returns`.
        """
        bench = benchmark if benchmark is not None else self.benchmark
        return _metrics.period_returns(
            self.returns,
            benchmark=bench,
            periods_per_year=periods_per_year,
        )

    def yearly_returns(self, *, benchmark: pd.Series | None = None) -> pd.Series | pd.DataFrame:
        """End-of-year returns.

        Returns a :class:`pd.Series` when no benchmark is available, or a
        two-column :class:`pd.DataFrame` (``benchmark``, ``strategy``)
        when one is supplied (or set on construction).
        """
        bench = benchmark if benchmark is not None else self.benchmark
        strategy = _metrics.yearly_returns(self.returns).rename(self.name)
        if bench is None:
            return strategy
        bench_yearly = _metrics.yearly_returns(bench).rename(
            str(bench.name) if bench.name is not None else "benchmark"
        )
        return pd.concat([bench_yearly, strategy], axis=1)

    # --------------------------------------------------------------- from-NAV

    @classmethod
    def from_nav(
        cls,
        nav: pd.Series | pd.DataFrame,
        *,
        distributions: pd.Series | None = None,
        capital_flows: pd.Series | None = None,
        method: ReturnMethod = "total_return",
        trades: pd.DataFrame | None = None,
        positions: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        name: str | None = None,
    ) -> Portfolio:
        """Analytics-mode Portfolio built from a NAV series.

        Return computation is delegated to
        :func:`fundcloud.metrics.returns_from_nav` — see there for the
        four-method menu. The default (``total_return`` on per-share
        NAV with distributions added back) matches how public funds
        report performance: injections and withdrawals are
        NAV-per-share-invariant, and only ``DISTRIBUTION`` flows need
        a per-share add-back.

        Parameters
        ----------
        nav
            NAV timeseries. A :class:`pd.Series` is used directly; a
            :class:`pd.DataFrame` with a ``nav`` column (preferred) or
            a single-column frame is coerced to a Series.
        distributions, capital_flows, method
            Forwarded to :func:`returns_from_nav`. ``distributions``
            (per-share, aligned to ``nav``'s index) drives the
            ``total_return`` path; ``capital_flows`` (signed net
            inflow) drives ``modified_dietz`` / ``daily_twr``.
        trades, positions
            Stashed on the returned Portfolio as ``_source_trades`` /
            ``_source_positions`` for downstream introspection
            (attribution reports, reconciliation). Not used for
            return computation.
        benchmark, name
            Forwarded to :meth:`__init__`.

        Returns
        -------
        Portfolio
            Analytics-mode portfolio with ``returns`` populated.
        """
        nav_s = _coerce_nav_series(nav)
        returns = _metrics.returns_from_nav(
            nav_s,
            distributions=distributions,
            capital_flows=capital_flows,
            method=method,
        )
        if name:
            returns = returns.rename(name)
        pf = cls(returns=returns, benchmark=benchmark, name=name)
        pf._source_trades = trades  # type: ignore[attr-defined]
        pf._source_positions = positions  # type: ignore[attr-defined]
        return pf

    # --------------------------------------------------------------- skfolio

    @classmethod
    def from_skfolio(cls, portfolio: Any, *, benchmark: pd.Series | None = None) -> Portfolio:
        """Lift a skfolio ``Portfolio`` into a Fundcloud ``Portfolio``.

        Copies the returns series and the (per-period) weight vector if one is
        exposed. Compatible with skfolio >= 0.6.
        """
        returns = _safe_skfolio_returns(portfolio)
        weights = _safe_skfolio_weights(portfolio)
        name = getattr(portfolio, "name", None) or type(portfolio).__name__
        return cls(
            returns=returns,
            weights=weights,
            benchmark=benchmark,
            name=name,
        )

    def to_skfolio(self) -> Any:
        """Build a skfolio ``Portfolio`` mirror of this object.

        Requires the ``[pf]`` extra. The resulting object is an instance of
        :class:`skfolio.Portfolio`.
        """
        try:
            from skfolio import Portfolio as SkPortfolio  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "to_skfolio() requires skfolio; install with: uv add 'fundcloud[pf]'"
            raise ImportError(msg) from e
        # skfolio's Portfolio constructor expects an X/returns and a weights
        # vector. We pass a per-period weights frame when available.
        return SkPortfolio(
            X=self.returns.to_frame() if isinstance(self.returns, pd.Series) else self.returns,
            weights=self._weights_frame.iloc[-1].to_numpy()
            if self._weights_frame is not None and len(self._weights_frame) > 0
            else None,
            name=self.name,
        )

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        r = self._returns
        n = 0 if r is None else len(r)
        return f"Portfolio(name={self.name!r}, periods={n}, cash={self._live.cash:.2f})"


# -------------------------------------------------------------------- helpers


def _coerce_returns(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        msg = (
            "Portfolio.returns must be a Series (single strategy) or a "
            "one-column DataFrame; got a frame with multiple columns."
        )
        raise ValueError(msg)
    return x


def _coerce_nav_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Accept a NAV Series directly, or a DataFrame with a ``nav`` column."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if "nav" in x.columns:
            return x["nav"]
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        msg = (
            "Portfolio.from_nav(nav=) accepts a Series or a DataFrame with "
            "a 'nav' column (or a single column); got a DataFrame with "
            f"columns {list(x.columns)!r}."
        )
        raise ValueError(msg)
    msg = f"Portfolio.from_nav(nav=) expected Series or DataFrame; got {type(x).__name__}"
    raise TypeError(msg)


def _coerce_weights(x: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame().T
    return x


def _safe_skfolio_returns(portfolio: Any) -> pd.Series:
    # skfolio 0.6+ exposes `returns` as a Series.
    if hasattr(portfolio, "returns"):
        r = portfolio.returns
        if isinstance(r, np.ndarray):
            idx = getattr(portfolio, "observations", None)
            r = pd.Series(r, index=idx if idx is not None else range(len(r)))
        return r
    raise AttributeError("skfolio portfolio has no `returns` attribute")


def _safe_skfolio_weights(portfolio: Any) -> pd.DataFrame | None:
    w = getattr(portfolio, "weights", None)
    assets = getattr(portfolio, "assets", None)
    if w is None:
        return None
    if isinstance(w, np.ndarray) and assets is not None:
        idx = getattr(portfolio, "observations", None)
        if idx is None:
            return pd.DataFrame([w], columns=list(assets))
        # skfolio's weights are per-period when using cross_val_predict.
        if w.ndim == 2:
            return pd.DataFrame(w, index=idx, columns=list(assets))
        return pd.DataFrame([w], columns=list(assets))
    if isinstance(w, pd.DataFrame):
        return w
    if isinstance(w, pd.Series):
        return w.to_frame().T
    return None
