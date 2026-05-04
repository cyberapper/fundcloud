"""Pure-Python NumPy-panel simulator fallback.

Implements the three deterministic :class:`~fundcloud.sim.Simulator`
entry points (:meth:`run_weights`, :meth:`run_orders`, :meth:`run_signals`)
as straight-line loops over NumPy arrays — no pandas ``.iloc`` /
``iterrows`` per bar, no ``Portfolio`` object method dispatch. This is
the **parity reference** the Rust kernel matches to 1e-10; when
``fundcloud.kernels.HAS_RUST`` is ``True``, the Rust binding is used
instead, but the two produce identical outputs by construction.

The loops emit flat struct-of-arrays output that
:func:`fundcloud.sim.simulator._rehydrate_sim_result` turns into a
:class:`~fundcloud.sim.SimResult` with pandas frames and a
:class:`~fundcloud.portfolio.Portfolio` snapshot.

Bracket orders
--------------
Both entry points accept ``high_panel`` and ``low_panel`` so the
per-bar intra-bar exit check (:func:`_check_intrabar_exits`) can fire
stop-loss / take-profit forced exits when the bar's range pierces a
position's recorded level. ``run_orders_loop`` additionally accepts
``order_sl_stop`` / ``order_tp_stop`` arrays — non-zero entries
attach a stop fraction to that submission; the resulting absolute
level is computed at fill time and stored on
:class:`_LiveBook` so subsequent bars can check it.

The Rust kernel mirrors this layout 1:1; see
``crates/fundcloud-core/src/sim.rs`` for the parity reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Cost-model tags. Must match crates/fundcloud-core/src/sim/types.rs.
COST_NONE = 0
COST_FIXED_BPS = 1
COST_PER_SHARE = 2

# Slippage-model tags.
SLIP_NONE = 0
SLIP_HALF_SPREAD = 1

# Execution-model tags.
EXEC_NEXT_BAR_OPEN = 0
EXEC_NEXT_BAR_CLOSE = 1

# Order-side tags.
SIDE_BUY = 0
SIDE_SELL = 1

# Order-kind tags.
KIND_MARKET = 0
KIND_LIMIT = 1

# Trade-reason tags. Mirrors :data:`fundcloud.sim.TradeReason` and the
# ``REASON_*`` constants in the Rust kernel. Translated to Python strings
# at the dispatch boundary.
REASON_SIGNAL = 0
REASON_STOP_LOSS = 1
REASON_TAKE_PROFIT = 2
REASON_TRAILING_STOP = 3


@dataclass(slots=True, frozen=True)
class SimCfg:
    """Flat simulator config passed to every loop function.

    Mirrors the Rust :struct:`sim::types::SimCfg`. Scalar parameters are
    folded inline so the Rust binding can take them as primitive args
    without constructing a PyO3 class per call.
    """

    cash: float
    cost_kind: int
    cost_param1: float  # bps for FixedBps, rate for PerShare, 0 otherwise
    cost_param2: float  # minimum (both FixedBps and PerShare)
    slip_kind: int
    slip_param1: float  # bps for HalfSpread
    exec_kind: int


def _apply_slippage(price: float, side: int, slip_kind: int, slip_p1: float) -> tuple[float, float]:
    """Return ``(fill_price, slippage_bps)``."""
    if slip_kind == SLIP_HALF_SPREAD and price > 0:
        half = slip_p1 / 2.0
        adj = price * (half * 1e-4)
        return (price + adj if side == SIDE_BUY else price - adj, half)
    return price, 0.0


def _fee(price: float, qty: float, cost_kind: int, cost_p1: float, cost_p2: float) -> float:
    if cost_kind == COST_FIXED_BPS:
        notional = abs(price * qty)
        return max(cost_p2, notional * cost_p1 * 1e-4)
    if cost_kind == COST_PER_SHARE:
        return max(cost_p2, abs(qty) * cost_p1)
    return 0.0  # NO_COST


# -----------------------------------------------------------------------------
# Shared per-bar primitives.


def _exec_prices_at(
    open_panel: np.ndarray, close_panel: np.ndarray, exec_kind: int, fill_idx: int
) -> np.ndarray:
    """Row of prices used for fills at ``fill_idx`` under ``exec_kind``."""
    return open_panel[fill_idx] if exec_kind == EXEC_NEXT_BAR_OPEN else close_panel[fill_idx]


def _fill_idx_for(signal_idx: int, n_bars: int, exec_kind: int) -> int:
    """Return the bar index at which an order submitted at ``signal_idx``
    fills, or -1 if the order cannot fill (last bar under any forward-only
    model). Both built-in execution models fill on bar ``signal_idx + 1``;
    the price column they pull from differs (open vs close).
    """
    del exec_kind  # both EXEC_NEXT_BAR_OPEN and EXEC_NEXT_BAR_CLOSE fill at t+1
    nxt = signal_idx + 1
    return nxt if nxt < n_bars else -1


@dataclass(slots=True)
class _LiveBook:
    """Mutable live state carried across bars.

    ``sl_level`` and ``tp_level`` are absolute prices; ``tsl_pct`` is a
    fraction in ``(0, 1)`` and ``tsl_anchor`` is the running high-water
    mark (long: ratchets up; short: ratchets down). All four use
    ``0.0`` as the wire-format sentinel for "no stop" — matches the
    Rust kernel layout. Fixed SL/TP re-anchor to the latest fill on
    accumulating entries; the trailing stop is initialised on the
    *first* entry that carries ``tsl_stop`` and held thereafter, so its
    high-water mark tracks from the original entry. Cleared whenever
    the position closes.
    """

    cash: float
    qty: np.ndarray  # (n_assets,) float64
    avg_cost: np.ndarray  # (n_assets,) float64 — volume-weighted average cost
    last_price: np.ndarray  # (n_assets,) float64 — last seen finite price
    sl_level: np.ndarray  # (n_assets,) float64 — 0.0 = no stop-loss
    tp_level: np.ndarray  # (n_assets,) float64 — 0.0 = no take-profit
    tsl_pct: np.ndarray  # (n_assets,) float64 — 0.0 = no trailing stop
    tsl_anchor: np.ndarray  # (n_assets,) float64 — high-water-mark price


def _new_book(cash: float, n_assets: int) -> _LiveBook:
    return _LiveBook(
        cash=cash,
        qty=np.zeros(n_assets, dtype=float),
        avg_cost=np.zeros(n_assets, dtype=float),
        last_price=np.zeros(n_assets, dtype=float),
        sl_level=np.zeros(n_assets, dtype=float),
        tp_level=np.zeros(n_assets, dtype=float),
        tsl_pct=np.zeros(n_assets, dtype=float),
        tsl_anchor=np.zeros(n_assets, dtype=float),
    )


def _execute_one(
    book: _LiveBook,
    asset_idx: int,
    side: int,
    qty_req: float,
    notional_req: float,
    kind: int,
    limit_price: float,
    prices_row: np.ndarray,
    cost_kind: int,
    cost_p1: float,
    cost_p2: float,
    slip_kind: int,
    slip_p1: float,
    sl_stop: float = 0.0,
    tp_stop: float = 0.0,
    tsl_stop: float = 0.0,
) -> tuple[bool, float, float, float, float]:
    """Try to execute a single order against the given price row.

    Returns ``(filled, signed_qty, fill_price, fee, slippage_bps)`` — but
    packed as a 5-tuple because Python dataclass allocation per order
    would dominate the loop. ``filled`` is False when the fill can't
    happen (price NaN / zero, limit violated, zero qty).

    When ``sl_stop``, ``tp_stop`` and/or ``tsl_stop`` are non-zero, the
    bracket state is written into the corresponding ``book.*`` arrays
    after the fill — matching :meth:`fundcloud.portfolio.Portfolio.apply`.
    Fixed SL/TP levels are anchored to *this fill's price* (latest-fill
    convention) on every accumulating entry; the trailing stop is
    initialised on the *first* entry that carries ``tsl_stop`` and
    held thereafter, with the anchor ratcheted bar-by-bar by
    :func:`_check_intrabar_exits`. Position closes (qty → 0) clear
    all four bracket arrays.
    """
    ref = float(prices_row[asset_idx])
    if not np.isfinite(ref) or ref <= 0:
        return False, 0.0, 0.0, 0.0, 0.0

    qty_abs = qty_req
    if qty_abs <= 0 and notional_req > 0:
        qty_abs = notional_req / ref
    if qty_abs <= 0:
        return False, 0.0, 0.0, 0.0, 0.0

    fill_price, slip_bps = _apply_slippage(ref, side, slip_kind, slip_p1)

    # Limit-order gating (matches Simulator._execute).
    if kind == KIND_LIMIT and np.isfinite(limit_price) and limit_price > 0:
        if side == SIDE_BUY and fill_price > limit_price:
            return False, 0.0, 0.0, 0.0, 0.0
        if side == SIDE_SELL and fill_price < limit_price:
            return False, 0.0, 0.0, 0.0, 0.0

    signed = qty_abs if side == SIDE_BUY else -qty_abs
    fee = _fee(fill_price, signed, cost_kind, cost_p1, cost_p2)
    # Apply to book.
    prev_qty = book.qty[asset_idx]
    prev_cost = book.avg_cost[asset_idx]
    new_qty = prev_qty + signed
    is_add = prev_qty == 0 or (prev_qty > 0) == (signed > 0)
    # Volume-weighted avg cost update — matches Portfolio.apply logic.
    if signed > 0 and new_qty > 0:
        book.avg_cost[asset_idx] = (prev_qty * prev_cost + signed * fill_price) / new_qty
    elif new_qty == 0:
        book.avg_cost[asset_idx] = 0.0
    book.qty[asset_idx] = new_qty
    book.cash -= signed * fill_price + fee
    book.last_price[asset_idx] = fill_price

    # Bracket-level bookkeeping. Mirrors Portfolio.apply.
    if new_qty == 0:
        book.sl_level[asset_idx] = 0.0
        book.tp_level[asset_idx] = 0.0
        book.tsl_pct[asset_idx] = 0.0
        book.tsl_anchor[asset_idx] = 0.0
    elif is_add and fill_price > 0:
        if sl_stop > 0:
            book.sl_level[asset_idx] = (
                fill_price * (1.0 - sl_stop) if new_qty > 0 else fill_price * (1.0 + sl_stop)
            )
        if tp_stop > 0:
            book.tp_level[asset_idx] = (
                fill_price * (1.0 + tp_stop) if new_qty > 0 else fill_price * (1.0 - tp_stop)
            )
        if tsl_stop > 0 and book.tsl_pct[asset_idx] == 0.0:
            # First entry that carries ``tsl_stop`` — initialise the
            # trail. Subsequent accumulating entries leave the anchor
            # in place so the high-water mark keeps ratcheting from the
            # original entry's price. Mirrors Portfolio.apply.
            book.tsl_pct[asset_idx] = tsl_stop
            book.tsl_anchor[asset_idx] = fill_price

    return True, signed, fill_price, fee, slip_bps


def _check_intrabar_exits(
    bar_idx: int,
    open_row: np.ndarray,
    high_row: np.ndarray,
    low_row: np.ndarray,
    book: _LiveBook,
    out: dict[str, Any],
    cost_kind: int,
    cost_p1: float,
    cost_p2: float,
    slip_kind: int,
    slip_p1: float,
) -> None:
    """Synthesise intra-bar SL / TP / trailing-stop exits.

    Mirrors :meth:`fundcloud.sim.simulator.Simulator._check_intrabar_exits`
    exactly, including the gap-vs-ratchet ordering for trailing stops
    and the arbitration rule (stops beat take-profit; between fixed SL
    and trailing SL the *tighter fill* wins). Called at the top of each
    bar in every ``run_*_loop``, before pending fills are drained, so
    bracket exits beat same-bar discretionary fills.

    Trailing-stop semantics — two-step ratchet within a single bar:

    1. **Pre-trigger ratchet** — anchor moves to ``bar.open`` if
       favourable (long: ``max(anchor, bar.open)``; short:
       ``min(anchor, bar.open)``). Most bars are a no-op; only gap-up
       (long) or gap-down (short) bars move the anchor here.
    2. **Trigger check** — compute the level from the post-open
       anchor; gap-fire at ``bar.open`` if the open already pierced
       it, otherwise fire at the level if the bar's unfavourable-side
       extreme reaches it.
    3. **Post-trigger ratchet** — if the trail didn't fire, ratchet
       against the favourable extreme (long: ``max(anchor, bar.high)``;
       short: ``min(anchor, bar.low)``) so subsequent bars see the new
       high-water mark.

    Splitting the ratchet across the trigger check (open-side before,
    full-extreme after) means a single wide-range bar can't tighten
    the level mid-bar to something the open never traded against —
    the trigger uses the level that was in force when the bar
    started.
    """
    n_assets = book.qty.shape[0]
    for j in range(n_assets):
        q = book.qty[j]
        if q == 0:
            continue
        sl = book.sl_level[j]
        tp = book.tp_level[j]
        tsl_pct = book.tsl_pct[j]
        if sl == 0.0 and tp == 0.0 and tsl_pct == 0.0:
            continue
        bar_open = float(open_row[j])
        bar_high = float(high_row[j])
        bar_low = float(low_row[j])
        if not (np.isfinite(bar_open) and np.isfinite(bar_high) and np.isfinite(bar_low)):
            continue

        is_long = q > 0

        # Compute the fill price each potential exit would land at this
        # bar (or 0.0 = "no fire"). This separates "did it fire?" from
        # "where did it fill?" so the gap rule can be applied correctly
        # per source — particularly important for the trail, whose level
        # changes mid-bar via ratchet.

        # Fixed stop-loss
        sl_fires_at = 0.0
        if sl > 0:
            if is_long and bar_low <= sl:
                sl_fires_at = min(bar_open, sl) if bar_open <= sl else sl
            elif (not is_long) and bar_high >= sl:
                sl_fires_at = max(bar_open, sl) if bar_open >= sl else sl

        # Trailing stop — two-step ratchet matching
        # `Simulator._check_intrabar_exits`:
        #   1. Before trigger check: ratchet anchor to bar.open.
        #   2. Trigger check uses that level.
        #   3. After trigger: ratchet to bar.high (long) / bar.low
        #      (short) for the next bar's check.
        tsl_fires_at = 0.0
        if tsl_pct > 0 and book.tsl_anchor[j] > 0:
            # Step 1 — ratchet to bar.open before trigger check.
            if (is_long and bar_open > book.tsl_anchor[j]) or (
                (not is_long) and bar_open < book.tsl_anchor[j]
            ):
                book.tsl_anchor[j] = bar_open
            anchor = book.tsl_anchor[j]
            level = anchor * (1.0 - tsl_pct) if is_long else anchor * (1.0 + tsl_pct)
            # Step 2 — trigger check.
            if is_long:
                if bar_open <= level:
                    tsl_fires_at = bar_open
                elif bar_low <= level:
                    tsl_fires_at = level
            else:
                if bar_open >= level:
                    tsl_fires_at = bar_open
                elif bar_high >= level:
                    tsl_fires_at = level
            # Step 3 — post-trigger ratchet for next bar
            # (skip when the trail fired; the position is closing).
            if tsl_fires_at == 0.0:
                if is_long and bar_high > book.tsl_anchor[j]:
                    book.tsl_anchor[j] = bar_high
                elif not is_long and bar_low < book.tsl_anchor[j]:
                    book.tsl_anchor[j] = bar_low

        # Take-profit
        tp_fires_at = 0.0
        if tp > 0:
            if is_long and bar_high >= tp:
                tp_fires_at = max(bar_open, tp) if bar_open >= tp else tp
            elif (not is_long) and bar_low <= tp:
                tp_fires_at = min(bar_open, tp) if bar_open <= tp else tp

        # Arbitrate. Stops beat take-profit. Between fixed SL and TSL
        # pick the tighter fill (higher price for long → less loss;
        # lower price for short → less loss).
        if sl_fires_at > 0 and tsl_fires_at > 0:
            if is_long:
                if tsl_fires_at >= sl_fires_at:
                    ref_price = tsl_fires_at
                    reason = REASON_TRAILING_STOP
                else:
                    ref_price = sl_fires_at
                    reason = REASON_STOP_LOSS
            else:
                if tsl_fires_at <= sl_fires_at:
                    ref_price = tsl_fires_at
                    reason = REASON_TRAILING_STOP
                else:
                    ref_price = sl_fires_at
                    reason = REASON_STOP_LOSS
        elif sl_fires_at > 0:
            ref_price = sl_fires_at
            reason = REASON_STOP_LOSS
        elif tsl_fires_at > 0:
            ref_price = tsl_fires_at
            reason = REASON_TRAILING_STOP
        elif tp_fires_at > 0:
            ref_price = tp_fires_at
            reason = REASON_TAKE_PROFIT
        else:
            continue

        exit_side = SIDE_SELL if is_long else SIDE_BUY
        qty_abs = abs(q)
        fill_price, slip = _apply_slippage(ref_price, exit_side, slip_kind, slip_p1)
        signed = qty_abs if exit_side == SIDE_BUY else -qty_abs
        fee = _fee(fill_price, signed, cost_kind, cost_p1, cost_p2)
        # Apply to book directly — bypasses _execute_one because the fill
        # always goes through (no notional / limit gating for stop fills).
        book.cash -= signed * fill_price + fee
        book.qty[j] = 0.0
        book.last_price[j] = fill_price
        book.avg_cost[j] = 0.0
        book.sl_level[j] = 0.0
        book.tp_level[j] = 0.0
        book.tsl_pct[j] = 0.0
        book.tsl_anchor[j] = 0.0
        _record_trade(out, bar_idx, j, signed, fill_price, fee, slip, reason)


def _mark_to_market(book: _LiveBook, close_row: np.ndarray) -> tuple[float, np.ndarray]:
    """Return ``(equity, per_asset_value)`` at bar close.

    Falls back to ``book.last_price`` when a close price is NaN — mirrors
    :meth:`Portfolio.mark_to_market`'s defensive behaviour on
    market-holiday bars.
    """
    n = book.qty.shape[0]
    per_asset = np.zeros(n, dtype=float)
    # Refresh last-price cache.
    for j in range(n):
        px = close_row[j]
        if np.isfinite(px) and px > 0:
            book.last_price[j] = px
    equity = book.cash
    for j in range(n):
        if book.qty[j] == 0:
            continue
        px = close_row[j]
        if not np.isfinite(px) or px <= 0:
            px = book.last_price[j]
            if (not np.isfinite(px) or px <= 0) and book.avg_cost[j] > 0:
                px = book.avg_cost[j]
            if not np.isfinite(px) or px <= 0:
                continue
        per_asset[j] = book.qty[j] * px
        equity += per_asset[j]
    return equity, per_asset


# -----------------------------------------------------------------------------
# Public entry points. Each returns a dict of SoA arrays the caller can
# drop straight into ``_rehydrate_sim_result``.


def _weights_to_orders_at_bar(
    book: _LiveBook,
    bar_close: np.ndarray,
    targets: np.ndarray,  # (n_assets,) NaN where asset not in weights; else target weight
    tolerance: float,
) -> list[tuple[int, int, float]]:
    """Per-bar weight→order translation. Returns ``[(asset_idx, side, qty), ...]``.

    Mirrors :func:`fundcloud.strategies.hold._orders_to_reach_weights`.
    """
    n = targets.shape[0]
    equity = book.cash
    for j in range(n):
        q = book.qty[j]
        if q == 0:
            continue
        px = bar_close[j]
        if not np.isfinite(px) or px <= 0:
            px = book.last_price[j]
        if np.isfinite(px) and px > 0:
            equity += q * px
    if equity <= 0:
        return []

    orders: list[tuple[int, int, float]] = []
    for j in range(n):
        tw = targets[j]
        if not np.isfinite(tw):
            continue
        px = bar_close[j]
        if not np.isfinite(px) or px <= 0:
            continue
        target_qty = (equity * tw) / px
        delta = target_qty - book.qty[j]
        if abs(delta) * px < tolerance * equity:
            continue
        if delta == 0:
            continue
        orders.append((j, SIDE_BUY if delta > 0 else SIDE_SELL, abs(delta)))
    return orders


def _empty_output(n_bars: int) -> dict[str, Any]:
    """Result scaffolding used by all three loops."""
    return {
        "equity": np.zeros(n_bars, dtype=float),
        "weights_history": [],  # list[(bar_idx, {asset_idx: weight})]
        "trade_bar": [],
        "trade_asset": [],
        "trade_qty": [],
        "trade_price": [],
        "trade_fee": [],
        "trade_slip_bps": [],
        "trade_reason": [],  # int codes; translated by the dispatcher
        "order_bar": [],
        "order_asset": [],
        "order_side": [],
        "order_qty": [],
        "order_notional": [],
        "order_kind": [],
        "order_limit_price": [],
        "order_filled": [],
    }


def _record_trade(
    out: dict[str, Any],
    bar: int,
    asset: int,
    signed_qty: float,
    price: float,
    fee: float,
    slip_bps: float,
    reason: int = REASON_SIGNAL,
) -> None:
    out["trade_bar"].append(bar)
    out["trade_asset"].append(asset)
    out["trade_qty"].append(signed_qty)
    out["trade_price"].append(price)
    out["trade_fee"].append(fee)
    out["trade_slip_bps"].append(slip_bps)
    out["trade_reason"].append(reason)


def _record_order(
    out: dict[str, Any],
    bar: int,
    asset: int,
    side: int,
    qty: float,
    notional: float,
    kind: int,
    limit_price: float,
    filled: bool,
) -> None:
    out["order_bar"].append(bar)
    out["order_asset"].append(asset)
    out["order_side"].append(side)
    out["order_qty"].append(qty)
    out["order_notional"].append(notional)
    out["order_kind"].append(kind)
    out["order_limit_price"].append(limit_price)
    out["order_filled"].append(filled)


# -----------------------------------------------------------------------------
# run_weights


def run_weights_loop(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    target_weights: np.ndarray,  # (n_target_rows, n_assets); NaN where asset absent
    target_bar_indices: np.ndarray,  # (n_target_rows,) int — bar index per weights row
    cfg: SimCfg,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Deterministic weights-path backtest. No Python callbacks.

    ``high_panel`` / ``low_panel`` exist for parity with the bracket-order
    path but are only consulted by :func:`_check_intrabar_exits` — and the
    weights path can't attach brackets, so the check is a no-op for any
    position opened by this loop.
    """
    n_bars, n_assets = close_panel.shape
    book = _new_book(cfg.cash, n_assets)
    out = _empty_output(n_bars)

    # Bars that trigger a rebalance — exactly the ones the caller passed in
    # ``target_bar_indices``. Matches the original Python simulator, which
    # only rebalances when ``ctx.ts`` is a row in the target-weights frame.
    # Forward-filling would re-rebalance every bar and explode the orders log.
    rebalance_at: dict[int, np.ndarray] = {}
    for r in range(target_bar_indices.size):
        rebalance_at[int(target_bar_indices[r])] = target_weights[r]

    # Pending orders: list[(fill_idx, asset, side, qty, notional, kind, limit_price,
    #                       order_idx, sl_stop, tp_stop, tsl_stop)].
    pending: list[tuple[int, int, int, float, float, int, float, int, float, float, float]] = []

    for i in range(n_bars):
        # 0. Intra-bar bracket check on positions opened by previous bars.
        _check_intrabar_exits(
            i,
            open_panel[i],
            high_panel[i],
            low_panel[i],
            book,
            out,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
        )

        # 1. Drain pending fills scheduled for this bar.
        remaining: list[
            tuple[int, int, int, float, float, int, float, int, float, float, float]
        ] = []
        for entry in pending:
            (
                fill_idx,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                order_idx,
                sl_stop,
                tp_stop,
                tsl_stop,
            ) = entry
            if fill_idx != i:
                remaining.append(entry)
                continue
            prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
            ok = _execute_one(
                book,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                prices_row,
                cfg.cost_kind,
                cfg.cost_param1,
                cfg.cost_param2,
                cfg.slip_kind,
                cfg.slip_param1,
                sl_stop,
                tp_stop,
                tsl_stop,
            )
            if ok[0]:
                _, signed, price, fee, slip = ok
                _record_trade(out, i, asset, signed, price, fee, slip, REASON_SIGNAL)
                out["order_filled"][order_idx] = True
        pending = remaining

        # 2. Produce new orders only on user-specified rebalance bars.
        bar_weights = rebalance_at.get(i)
        if bar_weights is not None:
            orders_this_bar = _weights_to_orders_at_bar(
                book, close_panel[i], bar_weights, tolerance
            )
            fill_idx = _fill_idx_for(i, n_bars, cfg.exec_kind)
            for asset, side, qty in orders_this_bar:
                if fill_idx < 0:
                    _record_order(out, i, asset, side, qty, 0.0, KIND_MARKET, 0.0, False)
                    continue
                _record_order(out, i, asset, side, qty, 0.0, KIND_MARKET, 0.0, False)
                if fill_idx == i:
                    prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
                    ok = _execute_one(
                        book,
                        asset,
                        side,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        prices_row,
                        cfg.cost_kind,
                        cfg.cost_param1,
                        cfg.cost_param2,
                        cfg.slip_kind,
                        cfg.slip_param1,
                    )
                    if ok[0]:
                        _, signed, price, fee, slip = ok
                        _record_trade(out, i, asset, signed, price, fee, slip, REASON_SIGNAL)
                        out["order_filled"][-1] = True
                else:
                    pending.append((
                        fill_idx,
                        asset,
                        side,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        len(out["order_filled"]) - 1,
                        0.0,
                        0.0,
                        0.0,
                    ))

        # 3. Mark-to-market.
        equity, per_asset = _mark_to_market(book, close_panel[i])
        out["equity"][i] = equity
        weights_snapshot = {}
        if equity != 0:
            for j in range(n_assets):
                v = per_asset[j]
                if v != 0:
                    weights_snapshot[j] = v / equity
        out["weights_history"].append((i, weights_snapshot))

    return out


# -----------------------------------------------------------------------------
# run_orders


def run_orders_loop(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    order_bar: np.ndarray,  # (n_orders,) int
    order_asset: np.ndarray,  # (n_orders,) int
    order_side: np.ndarray,  # (n_orders,) int
    order_qty: np.ndarray,  # (n_orders,) float (NaN for notional mode)
    order_notional: np.ndarray,  # (n_orders,) float (NaN for qty mode)
    order_kind: np.ndarray,  # (n_orders,) int
    order_limit_price: np.ndarray,  # (n_orders,) float (NaN for market)
    order_sl_stop: np.ndarray,  # (n_orders,) float (0.0 = no SL)
    order_tp_stop: np.ndarray,  # (n_orders,) float (0.0 = no TP)
    order_tsl_stop: np.ndarray,  # (n_orders,) float (0.0 = no trailing SL)
    cfg: SimCfg,
) -> dict[str, Any]:
    """Deterministic orders-path backtest with bracket-order support."""
    n_bars, n_assets = close_panel.shape
    book = _new_book(cfg.cash, n_assets)
    out = _empty_output(n_bars)

    # Group orders by submission bar for O(1) per-bar lookup.
    by_bar: dict[int, list[int]] = {}
    for k in range(order_bar.shape[0]):
        by_bar.setdefault(int(order_bar[k]), []).append(k)

    pending: list[tuple[int, int, int, float, float, int, float, int, float, float, float]] = []

    for i in range(n_bars):
        # 0. Intra-bar bracket check.
        _check_intrabar_exits(
            i,
            open_panel[i],
            high_panel[i],
            low_panel[i],
            book,
            out,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
        )

        # 1. Drain pending.
        remaining: list[
            tuple[int, int, int, float, float, int, float, int, float, float, float]
        ] = []
        for entry in pending:
            (
                fill_idx,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                order_idx,
                sl_stop,
                tp_stop,
                tsl_stop,
            ) = entry
            if fill_idx != i:
                remaining.append(entry)
                continue
            prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
            ok = _execute_one(
                book,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                prices_row,
                cfg.cost_kind,
                cfg.cost_param1,
                cfg.cost_param2,
                cfg.slip_kind,
                cfg.slip_param1,
                sl_stop,
                tp_stop,
                tsl_stop,
            )
            if ok[0]:
                _, signed, price, fee, slip = ok
                _record_trade(out, i, asset, signed, price, fee, slip, REASON_SIGNAL)
                out["order_filled"][order_idx] = True
        pending = remaining

        # 2. Submit new orders from the explicit log.
        fill_idx = _fill_idx_for(i, n_bars, cfg.exec_kind)
        for k in by_bar.get(i, ()):
            qty = float(order_qty[k])
            if not np.isfinite(qty):
                qty = 0.0
            notional = float(order_notional[k])
            if not np.isfinite(notional):
                notional = 0.0
            kind = int(order_kind[k])
            limit_px = float(order_limit_price[k])
            if not np.isfinite(limit_px):
                limit_px = 0.0
            sl_stop = float(order_sl_stop[k]) if order_sl_stop.size else 0.0
            if not np.isfinite(sl_stop) or sl_stop < 0:
                sl_stop = 0.0
            tp_stop = float(order_tp_stop[k]) if order_tp_stop.size else 0.0
            if not np.isfinite(tp_stop) or tp_stop < 0:
                tp_stop = 0.0
            tsl_stop = float(order_tsl_stop[k]) if order_tsl_stop.size else 0.0
            if not np.isfinite(tsl_stop) or tsl_stop < 0:
                tsl_stop = 0.0
            side = int(order_side[k])
            asset = int(order_asset[k])
            if fill_idx < 0:
                _record_order(out, i, asset, side, qty, notional, kind, limit_px, False)
                continue
            _record_order(out, i, asset, side, qty, notional, kind, limit_px, False)
            if fill_idx == i:
                prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
                ok = _execute_one(
                    book,
                    asset,
                    side,
                    qty,
                    notional,
                    kind,
                    limit_px,
                    prices_row,
                    cfg.cost_kind,
                    cfg.cost_param1,
                    cfg.cost_param2,
                    cfg.slip_kind,
                    cfg.slip_param1,
                    sl_stop,
                    tp_stop,
                    tsl_stop,
                )
                if ok[0]:
                    _, signed, price, fee, slip = ok
                    _record_trade(out, i, asset, signed, price, fee, slip, REASON_SIGNAL)
                    out["order_filled"][-1] = True
            else:
                pending.append((
                    fill_idx,
                    asset,
                    side,
                    qty,
                    notional,
                    kind,
                    limit_px,
                    len(out["order_filled"]) - 1,
                    sl_stop,
                    tp_stop,
                    tsl_stop,
                ))

        # 3. Mark-to-market.
        equity, per_asset = _mark_to_market(book, close_panel[i])
        out["equity"][i] = equity
        weights_snapshot = {}
        if equity != 0:
            for j in range(n_assets):
                v = per_asset[j]
                if v != 0:
                    weights_snapshot[j] = v / equity
        out["weights_history"].append((i, weights_snapshot))

    return out


# -----------------------------------------------------------------------------
# run_signals


def run_signals_loop(
    open_panel: np.ndarray,
    close_panel: np.ndarray,
    high_panel: np.ndarray,
    low_panel: np.ndarray,
    entries: np.ndarray,  # (n_bars, n_assets) bool
    exits: np.ndarray,  # (n_bars, n_assets) bool
    size: float,
    cfg: SimCfg,
) -> dict[str, Any]:
    """Deterministic signals-path backtest: boolean ``entries`` buy a
    ``size`` fraction of current cash; ``exits`` close the position.

    Like :func:`run_weights_loop`, the high/low panels are accepted for
    bracket-check parity but the signals path doesn't attach brackets to
    the orders it synthesises — so :func:`_check_intrabar_exits` is a
    no-op when driven from this entry point.
    """
    n_bars, n_assets = close_panel.shape
    book = _new_book(cfg.cash, n_assets)
    out = _empty_output(n_bars)

    pending: list[tuple[int, int, int, float, float, int, float, int, float, float, float]] = []

    for i in range(n_bars):
        # 0. Intra-bar bracket check (no-op for signals path).
        _check_intrabar_exits(
            i,
            open_panel[i],
            high_panel[i],
            low_panel[i],
            book,
            out,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
        )

        # 1. Drain pending.
        remaining: list[
            tuple[int, int, int, float, float, int, float, int, float, float, float]
        ] = []
        for entry in pending:
            (
                fill_idx,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                order_idx,
                sl_stop,
                tp_stop,
                tsl_stop,
            ) = entry
            if fill_idx != i:
                remaining.append(entry)
                continue
            prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
            ok = _execute_one(
                book,
                asset,
                side,
                qty,
                notional,
                kind,
                limit_px,
                prices_row,
                cfg.cost_kind,
                cfg.cost_param1,
                cfg.cost_param2,
                cfg.slip_kind,
                cfg.slip_param1,
                sl_stop,
                tp_stop,
                tsl_stop,
            )
            if ok[0]:
                _, signed, price, fee, slip = ok
                _record_trade(out, i, asset, signed, price, fee, slip, REASON_SIGNAL)
                out["order_filled"][order_idx] = True
        pending = remaining

        # 2. Emit entry / exit orders for this bar.
        fill_idx = _fill_idx_for(i, n_bars, cfg.exec_kind)
        close_row = close_panel[i]
        # Entries: allocate ``size`` fraction of CURRENT CASH per asset signalled.
        for j in range(n_assets):
            if entries[i, j]:
                px = close_row[j]
                if not np.isfinite(px) or px <= 0:
                    continue
                qty = max((book.cash * size) / px, 0.0)
                if qty <= 0:
                    continue
                if fill_idx < 0:
                    _record_order(out, i, j, SIDE_BUY, qty, 0.0, KIND_MARKET, 0.0, False)
                    continue
                _record_order(out, i, j, SIDE_BUY, qty, 0.0, KIND_MARKET, 0.0, False)
                if fill_idx == i:
                    prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
                    ok = _execute_one(
                        book,
                        j,
                        SIDE_BUY,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        prices_row,
                        cfg.cost_kind,
                        cfg.cost_param1,
                        cfg.cost_param2,
                        cfg.slip_kind,
                        cfg.slip_param1,
                    )
                    if ok[0]:
                        _, signed, price, fee, slip = ok
                        _record_trade(out, i, j, signed, price, fee, slip, REASON_SIGNAL)
                        out["order_filled"][-1] = True
                else:
                    pending.append((
                        fill_idx,
                        j,
                        SIDE_BUY,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        len(out["order_filled"]) - 1,
                        0.0,
                        0.0,
                        0.0,
                    ))
        # Exits: flatten any held position.
        for j in range(n_assets):
            if exits[i, j] and book.qty[j] > 0:
                qty = float(book.qty[j])
                if fill_idx < 0:
                    _record_order(out, i, j, SIDE_SELL, qty, 0.0, KIND_MARKET, 0.0, False)
                    continue
                _record_order(out, i, j, SIDE_SELL, qty, 0.0, KIND_MARKET, 0.0, False)
                if fill_idx == i:
                    prices_row = _exec_prices_at(open_panel, close_panel, cfg.exec_kind, i)
                    ok = _execute_one(
                        book,
                        j,
                        SIDE_SELL,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        prices_row,
                        cfg.cost_kind,
                        cfg.cost_param1,
                        cfg.cost_param2,
                        cfg.slip_kind,
                        cfg.slip_param1,
                    )
                    if ok[0]:
                        _, signed, price, fee, slip = ok
                        _record_trade(out, i, j, signed, price, fee, slip, REASON_SIGNAL)
                        out["order_filled"][-1] = True
                else:
                    pending.append((
                        fill_idx,
                        j,
                        SIDE_SELL,
                        qty,
                        0.0,
                        KIND_MARKET,
                        0.0,
                        len(out["order_filled"]) - 1,
                        0.0,
                        0.0,
                        0.0,
                    ))

        # 3. Mark-to-market.
        equity, per_asset = _mark_to_market(book, close_panel[i])
        out["equity"][i] = equity
        weights_snapshot = {}
        if equity != 0:
            for k in range(n_assets):
                v = per_asset[k]
                if v != 0:
                    weights_snapshot[k] = v / equity
        out["weights_history"].append((i, weights_snapshot))

    return out
