"""The :class:`Simulator` engine.

One live :class:`~fundcloud.portfolio.Portfolio`, four entry points
(:meth:`Simulator.run_strategy`, :meth:`Simulator.run_weights`,
:meth:`Simulator.run_signals`, :meth:`Simulator.run_orders`) — all
reusing the same core loop so strategies, weight paths, boolean signals,
and explicit orders produce the same post-run analytics surface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundcloud.data import Backend
from fundcloud.kernels import _sim as _sim_dispatcher
from fundcloud.kernels import _sim_pyfallback as _sim_pyfb
from fundcloud.portfolio import Portfolio
from fundcloud.sim import _py_kernel
from fundcloud.sim.costs import CostModel, FixedBps, NoCost, PerShare
from fundcloud.sim.execution import ExecutionModel, NextBarOpen, SameBarClose
from fundcloud.sim.orders import Order
from fundcloud.sim.slippage import HalfSpread, NoSlippage, SlippageModel
from fundcloud.sim.trades import Trade
from fundcloud.strategies.base import BaseStrategy, Context

_PerBarOrders = Callable[[Context], list[Order]]
_OnClose = Callable[[], None]

__all__ = ["SimResult", "Simulator"]


@dataclass(slots=True)
class SimResult:
    """Output of :meth:`Simulator.run_*`.

    Examples
    --------
    >>> # Given ``result = Simulator(bars).run_strategy(strategy)`` —
    >>> # ``result.pf`` is a shortcut for ``result.portfolio``:
    >>> # result.pf.sharpe()        # doctest: +SKIP
    >>> # result.pf.max_drawdown()  # doctest: +SKIP
    """

    portfolio: Portfolio
    trades: pd.DataFrame
    orders: pd.DataFrame
    equity_curve: pd.Series
    bars: pd.DataFrame = field(repr=False)

    @property
    def pf(self) -> Portfolio:
        """Shortcut: ``result.pf`` is the same object as ``result.portfolio``.

        Saves typing in interactive sessions where chains like
        ``result.pf.sharpe()`` or ``result.pf.metrics()`` are common.
        """
        return self.portfolio

    def summary(self) -> pd.Series:
        """Compact 11-metric view — delegates to :meth:`Portfolio.summary`."""
        return self.portfolio.summary()

    def metrics(self) -> pd.Series:
        """Full ~55-metric bundle — delegates to :meth:`Portfolio.metrics`."""
        return self.portfolio.metrics()


class Simulator:
    """Discrete-time backtest engine.

    Parameters
    ----------
    data
        A :class:`~fundcloud.data.Backend` (e.g. ``YF``, ``CSV``,
        ``Parquet``) or a raw ``Bars`` ``DataFrame`` with
        ``(field, symbol)`` MultiIndex columns.
    costs, slippage, execution
        Swap-in friction/execution models. Defaults: :class:`FixedBps` (5 bps),
        :class:`NoSlippage`, :class:`NextBarOpen`.
    cash
        Starting cash balance.

    Examples
    --------
    Drive a weekly DCA on a synthetic single-asset Bars frame:

    >>> import pandas as pd
    >>> from fundcloud.sim import Simulator
    >>> from fundcloud.strategies import DCA
    >>> bars = pd.DataFrame({  # tiny, flat-price dummy
    ...     ("open", "SPY"): 100.0,
    ...     ("high", "SPY"): 100.0,
    ...     ("low", "SPY"): 100.0,
    ...     ("close", "SPY"): 100.0,
    ...     ("volume", "SPY"): 1_000.0,
    ... }, index=pd.date_range("2024-01-02", periods=60, freq="B"))
    >>> bars.columns = pd.MultiIndex.from_tuples(bars.columns)
    >>> result = Simulator(bars, cash=10_000.0).run_strategy(
    ...     DCA(100.0, horizon="weekly", weights={"SPY": 1.0}),
    ... )
    >>> result.equity_curve.iloc[-1] >= 0.0
    True
    """

    def __init__(
        self,
        data: Backend | pd.DataFrame,
        *,
        costs: CostModel | None = None,
        slippage: SlippageModel | None = None,
        cash: float = 1_000_000.0,
        execution: ExecutionModel | None = None,
    ) -> None:
        self.bars: pd.DataFrame = _resolve_bars(data)
        self.costs: CostModel = costs if costs is not None else FixedBps(5.0)
        self.slippage: SlippageModel = slippage if slippage is not None else NoSlippage()
        self.cash: float = float(cash)
        self.execution: ExecutionModel = execution if execution is not None else NextBarOpen()

    # ------------------------------------------------------------------ API

    def run_strategy(self, strategy: BaseStrategy) -> SimResult:
        """Drive a :class:`BaseStrategy` bar by bar and return a :class:`SimResult`.

        The strategy sees one :class:`~fundcloud.strategies.base.Context` per
        bar and returns zero or more :class:`~fundcloud.sim.Order` instances.
        The simulator applies the execution / cost / slippage models and
        updates a single live :class:`~fundcloud.portfolio.Portfolio`.
        """
        portfolio = self._new_portfolio()
        strategy.init(self.bars, portfolio)
        pending: list[tuple[int, Order]] = []
        return self._drive(
            portfolio,
            pending,
            per_bar_orders=lambda ctx: strategy.decide(ctx),
            on_close=lambda: strategy.close(portfolio),
        )

    def run_weights(self, target_weights: pd.DataFrame) -> SimResult:
        """At each row of ``target_weights``, rebalance toward those weights.

        ``target_weights`` is a dense ``DataFrame`` indexed by timestamp with
        one column per asset; missing values are forward-filled.
        """
        cfg = _model_tags(self.costs, self.slippage, self.execution)
        if cfg is not None:
            return self._run_weights_fast(target_weights, cfg)

        aligned = target_weights.reindex(index=self.bars.index).ffill()
        portfolio = self._new_portfolio()

        def _orders_for(ctx: Context) -> list[Order]:
            if ctx.ts not in target_weights.index:
                return []
            weights = aligned.loc[ctx.ts].dropna().to_dict()
            if not weights:
                return []
            # Use the Hold helper's core logic, without the "first-bar" gate.
            from fundcloud.strategies.hold import _orders_to_reach_weights

            return _orders_to_reach_weights(ctx, weights)

        pending: list[tuple[int, Order]] = []
        return self._drive(portfolio, pending, per_bar_orders=_orders_for)

    def _run_weights_fast(
        self, target_weights: pd.DataFrame, cfg: _sim_pyfb.SimCfg
    ) -> SimResult:
        """Deterministic weights path via the Rust / NumPy dispatcher."""
        open_np, close_np, assets = _pack_panels(self.bars)
        asset_idx = {a: i for i, a in enumerate(assets)}
        # Align target-weight rows onto bars, keeping only rows whose timestamp
        # actually appears in the bars index.
        rows_ts = [ts for ts in target_weights.index if ts in self.bars.index]
        if not rows_ts:
            target_bar = np.zeros(0, dtype=np.intp)
            weights_np = np.zeros((0, len(assets)), dtype=float)
        else:
            bar_index_map = {ts: i for i, ts in enumerate(self.bars.index)}
            target_bar = np.asarray(
                [bar_index_map[ts] for ts in rows_ts], dtype=np.intp
            )
            weights_np = np.full((len(rows_ts), len(assets)), np.nan, dtype=float)
            for r, ts in enumerate(rows_ts):
                row = target_weights.loc[ts]
                for asset, w in row.dropna().items():
                    j = asset_idx.get(str(asset))
                    if j is not None:
                        weights_np[r, j] = float(w)
            weights_np = np.ascontiguousarray(weights_np)
        cfg = _sim_pyfb.SimCfg(
            cash=self.cash,
            cost_kind=cfg.cost_kind,
            cost_param1=cfg.cost_param1,
            cost_param2=cfg.cost_param2,
            slip_kind=cfg.slip_kind,
            slip_param1=cfg.slip_param1,
            exec_kind=cfg.exec_kind,
        )
        sim_out = _sim_dispatcher.run_weights(
            open_np, close_np, weights_np, target_bar, cfg
        )
        return _rehydrate_sim_result(sim_out, self.bars, assets, cash=self.cash)

    def run_signals(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        *,
        size: float = 1.0,
    ) -> SimResult:
        """Convert boolean entry/exit panels into market orders.

        ``size`` is a fraction of current cash to allocate per entry.
        """
        cfg = _model_tags(self.costs, self.slippage, self.execution)
        if cfg is not None:
            return self._run_signals_fast(entries, exits, size, cfg)

        en = entries.reindex(index=self.bars.index).fillna(False).astype(bool)
        ex = exits.reindex(index=self.bars.index).fillna(False).astype(bool)

        def _orders_for(ctx: Context) -> list[Order]:
            orders: list[Order] = []
            for asset in en.columns:
                if ctx.ts in en.index and en.loc[ctx.ts, asset]:
                    prices = _current_prices_map(ctx)
                    px = prices.get(asset)
                    if px is None or px <= 0:
                        continue
                    qty = max((ctx.portfolio.cash * size) / px, 0.0)
                    if qty > 0:
                        orders.append(Order(ts=ctx.ts, asset=asset, side="buy", qty=qty))
            for asset in ex.columns:
                if ctx.ts in ex.index and ex.loc[ctx.ts, asset]:
                    pos = ctx.portfolio._live.positions.get(asset)
                    if pos is not None and pos.qty > 0:
                        orders.append(Order(ts=ctx.ts, asset=asset, side="sell", qty=pos.qty))
            return orders

        portfolio = self._new_portfolio()
        pending: list[tuple[int, Order]] = []
        return self._drive(portfolio, pending, per_bar_orders=_orders_for)

    def _run_signals_fast(
        self,
        entries: pd.DataFrame,
        exits: pd.DataFrame,
        size: float,
        cfg: _sim_pyfb.SimCfg,
    ) -> SimResult:
        """Deterministic signals path via the Rust / NumPy dispatcher."""
        open_np, close_np, assets = _pack_panels(self.bars)
        n_bars = len(self.bars)
        n_assets = len(assets)
        asset_idx = {a: i for i, a in enumerate(assets)}

        en = entries.reindex(index=self.bars.index).fillna(False).astype(bool)
        ex = exits.reindex(index=self.bars.index).fillna(False).astype(bool)
        entries_np = np.zeros((n_bars, n_assets), dtype=bool)
        exits_np = np.zeros((n_bars, n_assets), dtype=bool)
        for col in en.columns:
            j = asset_idx.get(str(col))
            if j is not None:
                entries_np[:, j] = en[col].to_numpy(dtype=bool)
        for col in ex.columns:
            j = asset_idx.get(str(col))
            if j is not None:
                exits_np[:, j] = ex[col].to_numpy(dtype=bool)

        cfg = _sim_pyfb.SimCfg(
            cash=self.cash,
            cost_kind=cfg.cost_kind,
            cost_param1=cfg.cost_param1,
            cost_param2=cfg.cost_param2,
            slip_kind=cfg.slip_kind,
            slip_param1=cfg.slip_param1,
            exec_kind=cfg.exec_kind,
        )
        sim_out = _sim_dispatcher.run_signals(
            open_np, close_np, entries_np, exits_np, float(size), cfg
        )
        return _rehydrate_sim_result(sim_out, self.bars, assets, cash=self.cash)

    def run_orders(self, orders: pd.DataFrame) -> SimResult:
        """Execute an explicit long-format orders DataFrame."""
        required = {"ts", "asset", "side", "qty"}
        missing = required - set(orders.columns)
        if missing:
            msg = f"orders frame missing columns: {missing}"
            raise KeyError(msg)
        cfg = _model_tags(self.costs, self.slippage, self.execution)
        if cfg is not None:
            return self._run_orders_fast(orders, cfg)
        by_ts: dict[pd.Timestamp, list[Order]] = {}
        for row in orders.itertuples(index=False):
            ts = pd.Timestamp(row.ts)
            by_ts.setdefault(ts, []).append(
                Order(ts=ts, asset=str(row.asset), side=str(row.side), qty=float(row.qty))
            )

        def _orders_for(ctx: Context) -> list[Order]:
            return by_ts.get(ctx.ts, [])

        portfolio = self._new_portfolio()
        pending: list[tuple[int, Order]] = []
        return self._drive(portfolio, pending, per_bar_orders=_orders_for)

    def _run_orders_fast(
        self, orders: pd.DataFrame, cfg: _sim_pyfb.SimCfg
    ) -> SimResult:
        """Deterministic orders path via the Rust / NumPy dispatcher."""
        open_np, close_np, assets = _pack_panels(self.bars)
        asset_idx = {a: i for i, a in enumerate(assets)}
        bar_index_map = {ts: i for i, ts in enumerate(self.bars.index)}

        # Convert orders DataFrame → SoA arrays.
        bar_list: list[int] = []
        asset_list: list[int] = []
        side_list: list[int] = []
        qty_list: list[float] = []
        notional_list: list[float] = []
        kind_list: list[int] = []
        limit_list: list[float] = []
        for row in orders.itertuples(index=False):
            ts = pd.Timestamp(row.ts)
            i = bar_index_map.get(ts)
            if i is None:
                # Orders whose timestamp falls outside the bars are dropped —
                # same behaviour as the Python path (which lookups by ts and
                # finds nothing).
                continue
            j = asset_idx.get(str(row.asset))
            if j is None:
                continue
            qty = float(getattr(row, "qty", 0.0) or 0.0)
            notional = float(getattr(row, "notional", 0.0) or 0.0)
            kind_s = str(getattr(row, "kind", "market"))
            kind = _sim_pyfb.KIND_LIMIT if kind_s == "limit" else _sim_pyfb.KIND_MARKET
            limit_price = float(getattr(row, "limit_price", 0.0) or 0.0)
            side_s = str(row.side).lower()
            bar_list.append(i)
            asset_list.append(j)
            side_list.append(
                _sim_pyfb.SIDE_SELL if side_s == "sell" else _sim_pyfb.SIDE_BUY
            )
            qty_list.append(qty)
            notional_list.append(notional)
            kind_list.append(kind)
            limit_list.append(limit_price)

        cfg = _sim_pyfb.SimCfg(
            cash=self.cash,
            cost_kind=cfg.cost_kind,
            cost_param1=cfg.cost_param1,
            cost_param2=cfg.cost_param2,
            slip_kind=cfg.slip_kind,
            slip_param1=cfg.slip_param1,
            exec_kind=cfg.exec_kind,
        )
        sim_out = _sim_dispatcher.run_orders(
            open_np, close_np,
            np.asarray(bar_list, dtype=np.intp),
            np.asarray(asset_list, dtype=np.intp),
            np.asarray(side_list, dtype=np.intp),
            np.asarray(qty_list, dtype=float),
            np.asarray(notional_list, dtype=float),
            np.asarray(kind_list, dtype=np.intp),
            np.asarray(limit_list, dtype=float),
            cfg,
        )
        return _rehydrate_sim_result(sim_out, self.bars, assets, cash=self.cash)

    # --------------------------------------------------------------- internals

    def _new_portfolio(self) -> Portfolio:
        return Portfolio(cash=self.cash, name="strategy")

    def _drive(
        self,
        portfolio: Portfolio,
        pending: list[tuple[int, Order]],
        *,
        per_bar_orders: _PerBarOrders,
        on_close: _OnClose | None = None,
    ) -> SimResult:
        assets = _assets_from(self.bars)
        n_bars = len(self.bars)
        trades_rows: list[dict[str, object]] = []
        orders_rows: list[dict[str, object]] = []

        for i, ts, row in _py_kernel.iterate_bars(self.bars):
            # 1. Fill pending orders whose scheduled fill bar is `i` (produced
            #    by earlier bars under NextBarOpen-style execution).
            still_pending: list[tuple[int, Order]] = []
            for fill_idx, order in pending:
                if fill_idx == i:
                    fill = self._execute(order, i)
                    if fill is not None:
                        trades_rows.append(_trade_to_row(fill))
                        portfolio.apply(fill)
                else:
                    still_pending.append((fill_idx, order))
            pending[:] = still_pending

            # 2. Ask the caller for new orders, schedule them for their fill bar.
            ctx = Context(
                ts=ts,
                bar=row,
                history=self.bars.iloc[: i + 1],
                portfolio=portfolio,
                assets=assets,
            )
            new_orders = per_bar_orders(ctx) or []
            for order in new_orders:
                maybe_fill: int | None = self.execution.fill_at(
                    signal_index=i, bars_index_size=n_bars
                )
                if maybe_fill is None:
                    # Can't fill (e.g. last bar with NextBarOpen). Record but skip.
                    orders_rows.append(_order_to_row(order, filled=False))
                    continue
                orders_rows.append(_order_to_row(order, filled=True))
                if maybe_fill == i:
                    # SameBarClose-style: execute immediately so this bar's
                    # mark-to-market reflects the fill.
                    fill = self._execute(order, i)
                    if fill is not None:
                        trades_rows.append(_trade_to_row(fill))
                        portfolio.apply(fill)
                else:
                    pending.append((maybe_fill, order))

            # 3. Mark to market at bar close, after any same-bar fills.
            prices = _py_kernel.prices_at(self.bars, i, field="close")
            portfolio.mark_to_market(prices, ts)

        if on_close is not None:
            on_close()

        snapshot = portfolio.snapshot()
        snapshot.rename("strategy")
        trades_df = _rows_to_frame(trades_rows, _trade_columns())
        orders_df = _rows_to_frame(orders_rows, _order_columns())
        return SimResult(
            portfolio=snapshot,
            trades=trades_df,
            orders=orders_df,
            equity_curve=portfolio.equity_curve,
            bars=self.bars,
        )

    def _execute(self, order: Order, fill_idx: int) -> Trade | None:
        """Turn an ``Order`` into a ``Trade`` using reference price + slippage + cost."""
        ref_price = self.execution.reference_price(
            bars=self.bars,
            fill_index=fill_idx,
            asset=order.asset,
        )
        if not np.isfinite(ref_price) or ref_price <= 0:
            return None

        # Resolve qty if the order was notional-only.
        qty_abs = order.qty
        if qty_abs is None and order.notional is not None:
            qty_abs = abs(order.notional) / ref_price
        if qty_abs is None or qty_abs <= 0:
            return None

        signed_qty = qty_abs if order.side == "buy" else -qty_abs
        fill_price, slippage_bps = self.slippage.apply(price=ref_price, side=order.side)

        # Honour limit orders: skip if the limit can't be met.
        if order.kind == "limit" and order.limit_price is not None:
            if order.side == "buy" and fill_price > order.limit_price:
                return None
            if order.side == "sell" and fill_price < order.limit_price:
                return None

        fee = self.costs.fee(price=fill_price, qty=signed_qty)
        ts = self.bars.index[fill_idx]
        return Trade(
            order=order,
            ts=ts,
            asset=order.asset,
            qty=signed_qty,
            price=fill_price,
            fee=fee,
            slippage_bps=slippage_bps,
        )


# -------------------------------------------------------------------- helpers


_PRICE_FIELDS = frozenset({"open", "high", "low", "close"})


def _forward_fill_prices(bars: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill price columns so every bar has a reference price.

    On a mixed-frequency panel (e.g. 7-day crypto + 5-day equity), an
    equity's open/high/low/close are NaN on weekends. Without a fill, the
    simulator's next-bar-open execution can't price a fill that happens to
    land on a non-trading bar.

    We forward-fill only the price fields (``open``, ``high``, ``low``,
    ``close``); ``volume`` is left untouched — zero-volume on a closed
    bar is accurate and a non-zero ffill would be misleading. Leading
    NaNs (before the asset's first trading day) are preserved, so the
    simulator correctly refuses to buy an asset that has never traded.

    Returns a *copy*; the caller's input frame is never mutated.
    """
    if bars.empty:
        return bars.copy()
    out = bars.copy()
    if isinstance(out.columns, pd.MultiIndex):
        price_cols = [
            c for c in out.columns
            if str(c[0]).lower() in _PRICE_FIELDS
        ]
        if price_cols:
            out[price_cols] = out[price_cols].ffill()
    else:
        # Flat columns are interpreted as close-prices-per-asset.
        out = out.ffill()
    return out


def _resolve_bars(data: Backend | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return _forward_fill_prices(data.sort_index())
    if hasattr(data, "read"):
        bars = data.read()
        if not isinstance(bars, pd.DataFrame):
            msg = f"{type(data).__name__}.read() must return a DataFrame"
            raise TypeError(msg)
        return _forward_fill_prices(bars.sort_index())
    msg = f"Simulator.data must be a DataFrame or Backend, got {type(data).__name__}"
    raise TypeError(msg)


def _assets_from(bars: pd.DataFrame) -> tuple[str, ...]:
    if isinstance(bars.columns, pd.MultiIndex):
        return tuple(bars.columns.get_level_values(-1).unique())
    return tuple(str(c) for c in bars.columns)


def _current_prices_map(ctx: Context) -> dict[str, float]:
    bar = ctx.bar
    out: dict[str, float] = {}
    if isinstance(bar.index, pd.MultiIndex):
        for (field, asset), val in bar.items():
            if field == "close" and pd.notna(val):
                out[asset] = float(val)
    else:
        for asset, val in bar.items():
            if pd.notna(val):
                out[str(asset)] = float(val)
    return out


def _trade_columns() -> list[str]:
    return ["ts", "asset", "qty", "price", "fee", "slippage_bps", "notional"]


def _order_columns() -> list[str]:
    return ["ts", "asset", "side", "qty", "notional", "kind", "limit_price", "filled"]


def _trade_to_row(trade: Trade) -> dict[str, object]:
    return {
        "ts": trade.ts,
        "asset": trade.asset,
        "qty": trade.qty,
        "price": trade.price,
        "fee": trade.fee,
        "slippage_bps": trade.slippage_bps,
        "notional": trade.notional,
    }


def _order_to_row(order: Order, *, filled: bool) -> dict[str, object]:
    return {
        "ts": order.ts,
        "asset": order.asset,
        "side": order.side,
        "qty": order.qty,
        "notional": order.notional,
        "kind": order.kind,
        "limit_price": order.limit_price,
        "filled": filled,
    }


def _rows_to_frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows, columns=columns)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


# ============================================================================
# Fast-path helpers for the Rust / NumPy-panel dispatcher.
# ============================================================================


def _pack_panels(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract ``(open, close, asset_names)`` as C-contiguous float64 panels.

    The order of ``asset_names`` is the column order of both panels; the
    dispatcher uses this to translate asset-name lookups to integer index
    lookups.
    """
    assets = _assets_from(bars)
    n_bars = len(bars)
    n_assets = len(assets)
    open_np = np.full((n_bars, n_assets), np.nan, dtype=float)
    close_np = np.full((n_bars, n_assets), np.nan, dtype=float)
    if isinstance(bars.columns, pd.MultiIndex):
        for j, asset in enumerate(assets):
            if ("open", asset) in bars.columns:
                open_np[:, j] = bars[("open", asset)].to_numpy(dtype=float)
            if ("close", asset) in bars.columns:
                close_np[:, j] = bars[("close", asset)].to_numpy(dtype=float)
    else:
        # Flat columns: treat each as a close-price panel; open = close
        # under this convention (SameBarClose execution coalesces anyway).
        for j, asset in enumerate(assets):
            if asset in bars.columns:
                values = bars[asset].to_numpy(dtype=float)
                open_np[:, j] = values
                close_np[:, j] = values
    return (
        np.ascontiguousarray(open_np),
        np.ascontiguousarray(close_np),
        tuple(assets),
    )


def _model_tags(
    costs: CostModel, slippage: SlippageModel, execution: ExecutionModel
) -> _sim_pyfb.SimCfg | None:
    """Encode the three model objects as a scalar :class:`SimCfg`.

    Returns ``None`` when any of the three is a custom subclass — the
    caller falls back to the full Python ``_drive`` path in that case.
    """
    # Cost
    if isinstance(costs, NoCost):
        cost_kind = _sim_pyfb.COST_NONE
        cost_p1 = 0.0
        cost_p2 = 0.0
    elif isinstance(costs, FixedBps):
        cost_kind = _sim_pyfb.COST_FIXED_BPS
        cost_p1 = float(costs.bps)
        cost_p2 = float(costs.minimum)
    elif isinstance(costs, PerShare):
        cost_kind = _sim_pyfb.COST_PER_SHARE
        cost_p1 = float(costs.rate)
        cost_p2 = float(costs.minimum)
    else:
        return None
    # Slippage
    if isinstance(slippage, NoSlippage):
        slip_kind = _sim_pyfb.SLIP_NONE
        slip_p1 = 0.0
    elif isinstance(slippage, HalfSpread):
        slip_kind = _sim_pyfb.SLIP_HALF_SPREAD
        slip_p1 = float(slippage.spread_bps)
    else:
        return None
    # Execution
    if isinstance(execution, NextBarOpen):
        exec_kind = _sim_pyfb.EXEC_NEXT_BAR_OPEN
    elif isinstance(execution, SameBarClose):
        exec_kind = _sim_pyfb.EXEC_SAME_BAR_CLOSE
    else:
        return None
    return _sim_pyfb.SimCfg(
        cash=0.0,  # caller sets cash before dispatch
        cost_kind=cost_kind,
        cost_param1=cost_p1,
        cost_param2=cost_p2,
        slip_kind=slip_kind,
        slip_param1=slip_p1,
        exec_kind=exec_kind,
    )


def _rehydrate_sim_result(
    sim_out: dict[str, object],
    bars: pd.DataFrame,
    assets: tuple[str, ...],
    cash: float,
) -> SimResult:
    """Turn the dispatcher's SoA arrays into a :class:`SimResult`.

    Builds the trades / orders DataFrames, rebuilds the live-portfolio
    state (equity curve, per-bar weights, per-asset avg cost) by
    replaying the trade log, then takes a snapshot for analytics.
    """
    index = bars.index
    asset_by_idx = list(assets)

    # Trades DataFrame.
    n_trades = len(sim_out["trade_bar"])
    if n_trades:
        trades_df = pd.DataFrame({
            "ts": [index[int(i)] for i in sim_out["trade_bar"]],
            "asset": [asset_by_idx[int(a)] for a in sim_out["trade_asset"]],
            "qty": [float(q) for q in sim_out["trade_qty"]],
            "price": [float(p) for p in sim_out["trade_price"]],
            "fee": [float(f) for f in sim_out["trade_fee"]],
            "slippage_bps": [float(s) for s in sim_out["trade_slip_bps"]],
        })
        trades_df["notional"] = trades_df["qty"] * trades_df["price"]
    else:
        trades_df = pd.DataFrame(columns=_trade_columns())

    # Orders DataFrame.
    n_orders = len(sim_out["order_bar"])
    if n_orders:
        sides = ["buy" if int(s) == _sim_pyfb.SIDE_BUY else "sell" for s in sim_out["order_side"]]
        kinds = [
            "market" if int(k) == _sim_pyfb.KIND_MARKET else "limit"
            for k in sim_out["order_kind"]
        ]
        qtys = [float(q) if q != 0 else None for q in sim_out["order_qty"]]
        notionals = [float(n) if n != 0 else None for n in sim_out["order_notional"]]
        limits = [float(p) if p != 0 else None for p in sim_out["order_limit_price"]]
        orders_df = pd.DataFrame({
            "ts": [index[int(i)] for i in sim_out["order_bar"]],
            "asset": [asset_by_idx[int(a)] for a in sim_out["order_asset"]],
            "side": sides,
            "qty": qtys,
            "notional": notionals,
            "kind": kinds,
            "limit_price": limits,
            "filled": [bool(f) for f in sim_out["order_filled"]],
        })
    else:
        orders_df = pd.DataFrame(columns=_order_columns())

    # Build the Portfolio directly from the kernel's per-bar output.
    # Everything is vectorised — constructing a 2-D NumPy weights matrix
    # and wrapping it in a DataFrame once is ~100x faster than iterating
    # bar-by-bar and indexing into ``bars.index`` in Python.
    n_bars = len(index)
    n_assets = len(assets)
    equity_arr = np.asarray(sim_out["equity"], dtype=float)
    equity_series = pd.Series(equity_arr, index=index)

    # Trim leading zero-equity rows so the returns series starts at the
    # first bar with a non-zero portfolio value.
    nonzero_mask = equity_arr != 0.0
    if nonzero_mask.any():
        equity_series = equity_series[nonzero_mask]
    else:
        equity_series = pd.Series([cash], index=[index[0]])
    returns_series = equity_series.pct_change().dropna()

    weights_frame: pd.DataFrame | None = None
    if sim_out["weights_history"]:
        weights_np = np.zeros((n_bars, n_assets), dtype=float)
        seen = np.zeros(n_bars, dtype=bool)
        for bar_idx, pairs in sim_out["weights_history"]:
            d = dict(pairs)
            if not d:
                continue
            i = int(bar_idx)
            seen[i] = True
            for j, w in d.items():
                weights_np[i, int(j)] = float(w)
        if seen.any():
            idx_filter = np.where(seen)[0]
            # Construct without the DatetimeIndex first so pandas doesn't
            # box a Timestamp per (row, column) during column normalisation,
            # then attach the index as a single op. Brings a 10k-row
            # rehydration from ~500 ms down to ~10 ms.
            data_slice = np.ascontiguousarray(weights_np[idx_filter])
            weights_frame = pd.DataFrame(data_slice, columns=asset_by_idx)
            weights_frame.index = index[idx_filter]

    portfolio = Portfolio(
        returns=returns_series.rename("strategy") if not returns_series.empty else None,
        weights=weights_frame,
        name="strategy",
    )
    return SimResult(
        portfolio=portfolio,
        trades=trades_df,
        orders=orders_df,
        equity_curve=equity_series,
        bars=bars,
    )
