//! Deterministic simulator kernels — Rust parity reference for
//! [`fundcloud.kernels._sim_pyfallback`].
//!
//! Three entry points: [`run_weights`], [`run_orders`], [`run_signals`].
//! Each one consumes NumPy panels + a [`SimCfg`] and returns a [`SimOutput`]
//! struct-of-arrays. The PyO3 bindings in the sibling `fundcloud-py` crate
//! thin-wrap these.
//!
//! Bracket orders
//! --------------
//! All three entry points now accept `high` and `low` panels. Per-bar,
//! *after* pending fills drain, [`check_intrabar_exits`] tests every
//! open position carrying a non-zero [`LiveBook::sl_level`] /
//! [`LiveBook::tp_level`] / [`LiveBook::tsl_pct`] against the bar's
//! range. A breach synthesises a forced exit at the stop price (or the
//! bar's open on a gap), tagged with [`REASON_STOP_LOSS`] /
//! [`REASON_TAKE_PROFIT`] / [`REASON_TRAILING_STOP`] on the resulting
//! trade. Arbitration: any stop (fixed or trailing) beats take-profit
//! on the same bar; between the fixed SL and the trailing SL the
//! tighter fill wins (higher price for long, lower for short).
//!
//! [`run_orders`] additionally accepts per-order `order_sl_stop` /
//! `order_tp_stop` / `order_tsl_stop` arrays. Non-zero entries attach
//! the corresponding bracket fraction to that order; the resulting
//! absolute level (or, for trailing, the running anchor + fraction) is
//! recorded on [`LiveBook`] when the order fills, then checked on
//! subsequent bars. [`run_weights`] and [`run_signals`] don't take
//! per-order brackets (their APIs synthesise orders internally), but
//! they still accept the high/low panels so the per-bar check is
//! wired uniformly.
//!
//! The inner loop holds no Python objects, so the PyO3 bindings run
//! everything under `py.allow_threads` for GIL-released execution.

use ndarray::{Array1, ArrayView1, ArrayView2};

// --------------------------------------------------------------------------
// Config / enum tags — must stay numerically identical to the Python
// constants in `fundcloud.kernels._sim_pyfallback`.

/// Configuration passed to every simulator entry-point.
///
/// All numeric discriminants must stay numerically identical to the Python
/// constants in `fundcloud.kernels._sim_pyfallback`.
#[derive(Copy, Clone, Debug)]
pub struct SimCfg {
    /// Starting cash in account-currency units.
    pub cash: f64,
    /// Cost-model discriminant: `COST_NONE` / `COST_FIXED_BPS` / `COST_PER_SHARE`.
    pub cost_kind: u8,
    /// Primary cost parameter: bps for `COST_FIXED_BPS`, rate for `COST_PER_SHARE`.
    pub cost_param1: f64,
    /// Minimum cost floor (currency units), applied after the primary calculation.
    pub cost_param2: f64,
    /// Slippage-model discriminant: `SLIP_NONE` / `SLIP_HALF_SPREAD`.
    pub slip_kind: u8,
    /// Primary slippage parameter: half-spread in bps for `SLIP_HALF_SPREAD`.
    pub slip_param1: f64,
    /// Execution timing: `EXEC_NEXT_BAR_OPEN` / `EXEC_NEXT_BAR_CLOSE`.
    pub exec_kind: u8,
}

/// No transaction cost applied.
pub const COST_NONE: u8 = 0;
/// Fixed-bps commission on notional, subject to a minimum floor (`cost_param2`).
pub const COST_FIXED_BPS: u8 = 1;
/// Per-share commission, subject to a minimum floor (`cost_param2`).
pub const COST_PER_SHARE: u8 = 2;

/// No slippage model applied.
pub const SLIP_NONE: u8 = 0;
/// Bid-ask half-spread slippage: cost = price × (bps / 2 × 1e-4).
pub const SLIP_HALF_SPREAD: u8 = 1;

/// Execute at the open of the bar following the signal bar.
pub const EXEC_NEXT_BAR_OPEN: u8 = 0;
/// Execute at the close of the bar following the signal bar.
pub const EXEC_NEXT_BAR_CLOSE: u8 = 1;

/// Buy-side order direction.
pub const SIDE_BUY: u8 = 0;
/// Sell-side order direction.
pub const SIDE_SELL: u8 = 1;

/// Market order — fill at best available price.
pub const KIND_MARKET: u8 = 0;
/// Limit order — fill only if the reference price crosses the limit.
pub const KIND_LIMIT: u8 = 1;

/// Strategy-emitted fill (the default).
pub const REASON_SIGNAL: u8 = 0;
/// Forced exit synthesised by the intra-bar stop-loss check.
pub const REASON_STOP_LOSS: u8 = 1;
/// Forced exit synthesised by the intra-bar take-profit check.
pub const REASON_TAKE_PROFIT: u8 = 2;
/// Forced exit synthesised by the intra-bar trailing-stop check.
pub const REASON_TRAILING_STOP: u8 = 3;

// --------------------------------------------------------------------------
// Output struct-of-arrays. Every field is `Vec<...>` in trade / order order
// and `Array1<...>` for the per-bar equity / weights history.

/// Struct-of-arrays returned by every simulator entry-point.
///
/// Per-bar equity and weights history use `Array1` / `Vec` respectively;
/// all trade and order log fields are parallel `Vec`s indexed in fill or
/// submission order.
#[derive(Debug, Default)]
pub struct SimOutput {
    /// Per-bar portfolio equity in account-currency units (`n_bars`).
    pub equity: Array1<f64>,
    /// Per-bar weight snapshot: `(bar_idx, [(asset_idx, weight)])`.
    pub weights_history: Vec<(usize, Vec<(usize, f64)>)>,
    /// Bar index of each executed fill.
    pub trade_bar: Vec<usize>,
    /// Asset index of each executed fill.
    pub trade_asset: Vec<usize>,
    /// Signed fill quantity (positive = buy, negative = sell).
    pub trade_qty: Vec<f64>,
    /// Fill price after slippage.
    pub trade_price: Vec<f64>,
    /// Transaction fee charged at fill time.
    pub trade_fee: Vec<f64>,
    /// Realised slippage in basis points for each fill.
    pub trade_slip_bps: Vec<f64>,
    /// Why the trade fired — `REASON_SIGNAL` / `REASON_STOP_LOSS` /
    /// `REASON_TAKE_PROFIT` / `REASON_TRAILING_STOP`. Translated to
    /// Python strings at the PyO3 boundary.
    pub trade_reason: Vec<u8>,
    /// Bar index of each submitted order.
    pub order_bar: Vec<usize>,
    /// Asset index of each submitted order.
    pub order_asset: Vec<usize>,
    /// Side of each submitted order (`SIDE_BUY` / `SIDE_SELL`).
    pub order_side: Vec<u8>,
    /// Requested quantity for each order.
    pub order_qty: Vec<f64>,
    /// Notional target; non-zero overrides `order_qty` when qty is zero.
    pub order_notional: Vec<f64>,
    /// Order kind (`KIND_MARKET` / `KIND_LIMIT`).
    pub order_kind: Vec<u8>,
    /// Limit price; ignored for market orders.
    pub order_limit_price: Vec<f64>,
    /// Whether each order was filled within the simulation window.
    pub order_filled: Vec<bool>,
}

// --------------------------------------------------------------------------
// Shared primitives.

fn apply_slippage(price: f64, side: u8, slip_kind: u8, slip_p1: f64) -> (f64, f64) {
    if slip_kind == SLIP_HALF_SPREAD && price > 0.0 {
        let half = slip_p1 / 2.0;
        let adj = price * (half * 1e-4);
        let px = if side == SIDE_BUY {
            price + adj
        } else {
            price - adj
        };
        (px, half)
    } else {
        (price, 0.0)
    }
}

fn fee(price: f64, qty: f64, cost_kind: u8, p1: f64, p2: f64) -> f64 {
    if cost_kind == COST_FIXED_BPS {
        let notional = (price * qty).abs();
        let computed = notional * p1 * 1e-4;
        if computed > p2 {
            computed
        } else {
            p2
        }
    } else if cost_kind == COST_PER_SHARE {
        let computed = qty.abs() * p1;
        if computed > p2 {
            computed
        } else {
            p2
        }
    } else {
        0.0
    }
}

fn exec_price_at(
    open: &ArrayView2<'_, f64>,
    close: &ArrayView2<'_, f64>,
    exec_kind: u8,
    fill_idx: usize,
    asset_idx: usize,
) -> f64 {
    if exec_kind == EXEC_NEXT_BAR_OPEN {
        open[[fill_idx, asset_idx]]
    } else {
        close[[fill_idx, asset_idx]]
    }
}

fn fill_idx_for(signal_idx: usize, n_bars: usize, _exec_kind: u8) -> i64 {
    // Both built-in execution models (EXEC_NEXT_BAR_OPEN and
    // EXEC_NEXT_BAR_CLOSE) fill on bar `signal_idx + 1`; only the
    // price column they pull from differs (handled in `exec_price_at`).
    let nxt = signal_idx + 1;
    if nxt < n_bars {
        nxt as i64
    } else {
        -1
    }
}

// --------------------------------------------------------------------------
// LiveBook mirrors _sim_pyfallback._LiveBook.

/// Mutable per-asset state carried across bars.
///
/// `sl_level` / `tp_level` are absolute prices; `tsl_pct` is a fraction
/// in `(0, 1)` and `tsl_anchor` is the running high-water-mark price
/// (long: ratchets up; short: ratchets down). All four use `0.0` as
/// the wire-format sentinel meaning "no stop". Fixed SL/TP re-anchor
/// to the latest fill on accumulating entries; the trailing stop is
/// initialised on the *first* entry that carries `tsl_stop` and held
/// thereafter, so its anchor tracks the high-water mark from the
/// original entry. Cleared whenever the position closes (qty → 0),
/// regardless of whether the closing trade carried fresh stop
/// fractions.
struct LiveBook {
    cash: f64,
    qty: Array1<f64>,
    avg_cost: Array1<f64>,
    last_price: Array1<f64>,
    sl_level: Array1<f64>,
    tp_level: Array1<f64>,
    tsl_pct: Array1<f64>,
    tsl_anchor: Array1<f64>,
}

impl LiveBook {
    fn new(cash: f64, n_assets: usize) -> Self {
        Self {
            cash,
            qty: Array1::zeros(n_assets),
            avg_cost: Array1::zeros(n_assets),
            last_price: Array1::zeros(n_assets),
            sl_level: Array1::zeros(n_assets),
            tp_level: Array1::zeros(n_assets),
            tsl_pct: Array1::zeros(n_assets),
            tsl_anchor: Array1::zeros(n_assets),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_one(
    book: &mut LiveBook,
    asset_idx: usize,
    side: u8,
    qty_req: f64,
    notional_req: f64,
    kind: u8,
    limit_price: f64,
    ref_price: f64,
    cost_kind: u8,
    cost_p1: f64,
    cost_p2: f64,
    slip_kind: u8,
    slip_p1: f64,
    sl_stop: f64,
    tp_stop: f64,
    tsl_stop: f64,
) -> Option<(f64, f64, f64, f64)> {
    // Returns (signed_qty, fill_price, fee, slip_bps) on success, None on drop.
    if !ref_price.is_finite() || ref_price <= 0.0 {
        return None;
    }
    let mut qty_abs = qty_req;
    if qty_abs <= 0.0 && notional_req > 0.0 {
        qty_abs = notional_req / ref_price;
    }
    if qty_abs <= 0.0 {
        return None;
    }
    let (fill_price, slip_bps) = apply_slippage(ref_price, side, slip_kind, slip_p1);
    if kind == KIND_LIMIT && limit_price.is_finite() && limit_price > 0.0 {
        if side == SIDE_BUY && fill_price > limit_price {
            return None;
        }
        if side == SIDE_SELL && fill_price < limit_price {
            return None;
        }
    }
    let signed = if side == SIDE_BUY { qty_abs } else { -qty_abs };
    let f = fee(fill_price, signed, cost_kind, cost_p1, cost_p2);
    let prev_qty = book.qty[asset_idx];
    let prev_cost = book.avg_cost[asset_idx];
    let new_qty = prev_qty + signed;
    let is_add = prev_qty == 0.0 || (prev_qty > 0.0) == (signed > 0.0);
    if signed > 0.0 && new_qty > 0.0 {
        book.avg_cost[asset_idx] = (prev_qty * prev_cost + signed * fill_price) / new_qty;
    } else if new_qty == 0.0 {
        book.avg_cost[asset_idx] = 0.0;
    }
    book.qty[asset_idx] = new_qty;
    book.cash -= signed * fill_price + f;
    book.last_price[asset_idx] = fill_price;

    // Bracket-level bookkeeping. Mirrors `Portfolio.apply` and
    // `_sim_pyfallback._execute_one`.
    if new_qty == 0.0 {
        book.sl_level[asset_idx] = 0.0;
        book.tp_level[asset_idx] = 0.0;
        book.tsl_pct[asset_idx] = 0.0;
        book.tsl_anchor[asset_idx] = 0.0;
    } else if is_add && fill_price > 0.0 {
        if sl_stop > 0.0 {
            book.sl_level[asset_idx] = if new_qty > 0.0 {
                fill_price * (1.0 - sl_stop)
            } else {
                fill_price * (1.0 + sl_stop)
            };
        }
        if tp_stop > 0.0 {
            book.tp_level[asset_idx] = if new_qty > 0.0 {
                fill_price * (1.0 + tp_stop)
            } else {
                fill_price * (1.0 - tp_stop)
            };
        }
        if tsl_stop > 0.0 && book.tsl_pct[asset_idx] == 0.0 {
            // First entry that carries `tsl_stop` — initialise the
            // trail. Subsequent accumulating entries leave the anchor
            // in place so the high-water mark keeps ratcheting from
            // the original entry's price. Mirrors `Portfolio.apply`.
            book.tsl_pct[asset_idx] = tsl_stop;
            book.tsl_anchor[asset_idx] = fill_price;
        }
    }

    Some((signed, fill_price, f, slip_bps))
}

/// Synthesise intra-bar SL / TP / trailing-stop exits.
///
/// Mirrors [`fundcloud.kernels._sim_pyfallback._check_intrabar_exits`]
/// and [`fundcloud.sim.simulator.Simulator._check_intrabar_exits`]
/// exactly, including the gap-vs-ratchet ordering for trailing stops
/// and the arbitration rule (stops beat take-profit; between fixed SL
/// and trailing SL the *tighter fill* wins). Called once per bar in
/// every `run_*` entry point *after* pending fills drain, so a
/// position opened at the bar's open is visible to that bar's bracket
/// check — a same-bar fill-bar SL/TP/TSL fires when the bar's range
/// pierces.
///
/// Trailing-stop semantics — two-step ratchet within a single bar:
///
/// 1. **Pre-trigger ratchet** — anchor moves to `bar.open` if
///    favourable (long: `max(anchor, bar.open)`; short:
///    `min(anchor, bar.open)`). Most bars are a no-op; only gap-up
///    (long) or gap-down (short) bars move the anchor here.
/// 2. **Trigger check** — compute the level from the post-open
///    anchor, then check the bar's OHLC against it.
/// 3. **Post-trigger ratchet** — if the trail didn't fire, ratchet
///    the anchor against the favourable extreme (long:
///    `max(anchor, bar.high)`; short: `min(anchor, bar.low)`) so
///    subsequent bars see the new high-water mark.
///
/// Splitting the ratchet across the trigger check (open-side before,
/// full-extreme after) means a single wide-range bar can't tighten
/// the level mid-bar to something the open never traded against.
#[allow(clippy::too_many_arguments)]
fn check_intrabar_exits(
    bar_idx: usize,
    open: &ArrayView2<'_, f64>,
    high: &ArrayView2<'_, f64>,
    low: &ArrayView2<'_, f64>,
    book: &mut LiveBook,
    out: &mut SimOutput,
    cfg: &SimCfg,
) {
    let n = book.qty.len();
    for j in 0..n {
        let q = book.qty[j];
        if q == 0.0 {
            continue;
        }
        let sl = book.sl_level[j];
        let tp = book.tp_level[j];
        let tsl_pct = book.tsl_pct[j];
        if sl == 0.0 && tp == 0.0 && tsl_pct == 0.0 {
            continue;
        }
        let bar_open = open[[bar_idx, j]];
        let bar_high = high[[bar_idx, j]];
        let bar_low = low[[bar_idx, j]];
        if !bar_open.is_finite() || !bar_high.is_finite() || !bar_low.is_finite() {
            continue;
        }

        let is_long = q > 0.0;

        // Compute the fill price each potential exit would land at this
        // bar (or 0.0 = "no fire"). Separating "did it fire?" from
        // "where did it fill?" lets the trail's ratchet-mid-bar update
        // the anchor without affecting the gap rule for fixed SL / TP.

        // Fixed stop-loss
        let mut sl_fires_at = 0.0_f64;
        if sl > 0.0 {
            if is_long && bar_low <= sl {
                sl_fires_at = if bar_open <= sl { bar_open.min(sl) } else { sl };
            } else if !is_long && bar_high >= sl {
                sl_fires_at = if bar_open >= sl { bar_open.max(sl) } else { sl };
            }
        }

        // Trailing stop — two-step ratchet matching
        // `Simulator._check_intrabar_exits`:
        //   1. Before trigger check: ratchet anchor to bar.open.
        //   2. Trigger check uses that level.
        //   3. After trigger: ratchet to bar.high (long) / bar.low
        //      (short) for the next bar's check.
        let mut tsl_fires_at = 0.0_f64;
        if tsl_pct > 0.0 && book.tsl_anchor[j] > 0.0 {
            // Step 1 — ratchet to bar.open before trigger check.
            if (is_long && bar_open > book.tsl_anchor[j])
                || (!is_long && bar_open < book.tsl_anchor[j])
            {
                book.tsl_anchor[j] = bar_open;
            }
            let anchor = book.tsl_anchor[j];
            let level = if is_long {
                anchor * (1.0 - tsl_pct)
            } else {
                anchor * (1.0 + tsl_pct)
            };
            // Step 2 — trigger check.
            if is_long {
                if bar_open <= level {
                    tsl_fires_at = bar_open;
                } else if bar_low <= level {
                    tsl_fires_at = level;
                }
            } else if bar_open >= level {
                tsl_fires_at = bar_open;
            } else if bar_high >= level {
                tsl_fires_at = level;
            }
            // Step 3 — post-trigger ratchet for next bar
            // (skip when the trail fired; the position is closing).
            if tsl_fires_at == 0.0 {
                if is_long && bar_high > book.tsl_anchor[j] {
                    book.tsl_anchor[j] = bar_high;
                } else if !is_long && bar_low < book.tsl_anchor[j] {
                    book.tsl_anchor[j] = bar_low;
                }
            }
        }

        // Take-profit
        let mut tp_fires_at = 0.0_f64;
        if tp > 0.0 {
            if is_long && bar_high >= tp {
                tp_fires_at = if bar_open >= tp { bar_open.max(tp) } else { tp };
            } else if !is_long && bar_low <= tp {
                tp_fires_at = if bar_open <= tp { bar_open.min(tp) } else { tp };
            }
        }

        // Arbitrate. Stops beat take-profit. Between fixed SL and TSL
        // pick the tighter fill (higher price for long → less loss;
        // lower price for short → less loss).
        let (ref_price, reason) = if sl_fires_at > 0.0 && tsl_fires_at > 0.0 {
            if is_long {
                if tsl_fires_at >= sl_fires_at {
                    (tsl_fires_at, REASON_TRAILING_STOP)
                } else {
                    (sl_fires_at, REASON_STOP_LOSS)
                }
            } else if tsl_fires_at <= sl_fires_at {
                (tsl_fires_at, REASON_TRAILING_STOP)
            } else {
                (sl_fires_at, REASON_STOP_LOSS)
            }
        } else if sl_fires_at > 0.0 {
            (sl_fires_at, REASON_STOP_LOSS)
        } else if tsl_fires_at > 0.0 {
            (tsl_fires_at, REASON_TRAILING_STOP)
        } else if tp_fires_at > 0.0 {
            (tp_fires_at, REASON_TAKE_PROFIT)
        } else {
            continue;
        };

        let exit_side = if is_long { SIDE_SELL } else { SIDE_BUY };
        let qty_abs = q.abs();
        let (fill_price, slip) =
            apply_slippage(ref_price, exit_side, cfg.slip_kind, cfg.slip_param1);
        let signed = if exit_side == SIDE_BUY {
            qty_abs
        } else {
            -qty_abs
        };
        let fee_amt = fee(
            fill_price,
            signed,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
        );

        // Apply to book directly — bypasses execute_one because the fill
        // always goes through (no notional / limit gating for stop fills).
        book.cash -= signed * fill_price + fee_amt;
        book.qty[j] = 0.0;
        book.last_price[j] = fill_price;
        book.avg_cost[j] = 0.0;
        book.sl_level[j] = 0.0;
        book.tp_level[j] = 0.0;
        book.tsl_pct[j] = 0.0;
        book.tsl_anchor[j] = 0.0;
        out.trade_bar.push(bar_idx);
        out.trade_asset.push(j);
        out.trade_qty.push(signed);
        out.trade_price.push(fill_price);
        out.trade_fee.push(fee_amt);
        out.trade_slip_bps.push(slip);
        out.trade_reason.push(reason);
    }
}

fn mark_to_market(book: &mut LiveBook, close_row: ArrayView1<'_, f64>) -> (f64, Vec<(usize, f64)>) {
    let n = book.qty.len();
    // Refresh last-price cache from this bar's closes.
    for j in 0..n {
        let px = close_row[j];
        if px.is_finite() && px > 0.0 {
            book.last_price[j] = px;
        }
    }
    let mut equity = book.cash;
    let mut per_asset = Vec::with_capacity(n);
    for j in 0..n {
        if book.qty[j] == 0.0 {
            continue;
        }
        let mut px = close_row[j];
        if !px.is_finite() || px <= 0.0 {
            px = book.last_price[j];
            if (!px.is_finite() || px <= 0.0) && book.avg_cost[j] > 0.0 {
                px = book.avg_cost[j];
            }
            if !px.is_finite() || px <= 0.0 {
                continue;
            }
        }
        let v = book.qty[j] * px;
        equity += v;
        per_asset.push((j, v));
    }
    (equity, per_asset)
}

fn emit_weights(equity: f64, per_asset: &[(usize, f64)]) -> Vec<(usize, f64)> {
    if equity == 0.0 {
        return per_asset.iter().map(|(j, _)| (*j, 0.0)).collect();
    }
    per_asset.iter().map(|(j, v)| (*j, v / equity)).collect()
}

// Pending-order queue element. Carries `sl_stop` / `tp_stop` /
// `tsl_stop` so the fractions survive the bar-to-bar wait between
// submission and fill.
#[derive(Copy, Clone)]
struct Pending {
    fill_idx: usize,
    asset: usize,
    side: u8,
    qty: f64,
    notional: f64,
    kind: u8,
    limit_price: f64,
    /// Index into `out.order_filled` so the drain loop can mark the order filled.
    order_idx: usize,
    sl_stop: f64,
    tp_stop: f64,
    tsl_stop: f64,
}

fn drain_pending(
    pending: &mut Vec<Pending>,
    i: usize,
    open: &ArrayView2<'_, f64>,
    close: &ArrayView2<'_, f64>,
    book: &mut LiveBook,
    out: &mut SimOutput,
    cfg: &SimCfg,
) {
    let mut keep: Vec<Pending> = Vec::with_capacity(pending.len());
    for p in pending.drain(..) {
        if p.fill_idx != i {
            keep.push(p);
            continue;
        }
        let ref_price = exec_price_at(open, close, cfg.exec_kind, i, p.asset);
        if let Some((signed, price, f, slip)) = execute_one(
            book,
            p.asset,
            p.side,
            p.qty,
            p.notional,
            p.kind,
            p.limit_price,
            ref_price,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            p.sl_stop,
            p.tp_stop,
            p.tsl_stop,
        ) {
            out.trade_bar.push(i);
            out.trade_asset.push(p.asset);
            out.trade_qty.push(signed);
            out.trade_price.push(price);
            out.trade_fee.push(f);
            out.trade_slip_bps.push(slip);
            out.trade_reason.push(REASON_SIGNAL);
            out.order_filled[p.order_idx] = true;
        }
    }
    *pending = keep;
}

#[allow(clippy::too_many_arguments)]
fn submit_order(
    i: usize,
    asset: usize,
    side: u8,
    qty: f64,
    notional: f64,
    kind: u8,
    limit_price: f64,
    n_bars: usize,
    open: &ArrayView2<'_, f64>,
    close: &ArrayView2<'_, f64>,
    book: &mut LiveBook,
    pending: &mut Vec<Pending>,
    out: &mut SimOutput,
    cfg: &SimCfg,
    sl_stop: f64,
    tp_stop: f64,
    tsl_stop: f64,
) {
    let fill_i = fill_idx_for(i, n_bars, cfg.exec_kind);
    if fill_i < 0 {
        out.order_bar.push(i);
        out.order_asset.push(asset);
        out.order_side.push(side);
        out.order_qty.push(qty);
        out.order_notional.push(notional);
        out.order_kind.push(kind);
        out.order_limit_price.push(limit_price);
        out.order_filled.push(false);
        return;
    }
    out.order_bar.push(i);
    out.order_asset.push(asset);
    out.order_side.push(side);
    out.order_qty.push(qty);
    out.order_notional.push(notional);
    out.order_kind.push(kind);
    out.order_limit_price.push(limit_price);
    out.order_filled.push(false);
    let order_idx = out.order_filled.len() - 1;
    let fill_i = fill_i as usize;
    if fill_i == i {
        let ref_price = exec_price_at(open, close, cfg.exec_kind, i, asset);
        if let Some((signed, price, f, slip)) = execute_one(
            book,
            asset,
            side,
            qty,
            notional,
            kind,
            limit_price,
            ref_price,
            cfg.cost_kind,
            cfg.cost_param1,
            cfg.cost_param2,
            cfg.slip_kind,
            cfg.slip_param1,
            sl_stop,
            tp_stop,
            tsl_stop,
        ) {
            out.trade_bar.push(i);
            out.trade_asset.push(asset);
            out.trade_qty.push(signed);
            out.trade_price.push(price);
            out.trade_fee.push(f);
            out.trade_slip_bps.push(slip);
            out.trade_reason.push(REASON_SIGNAL);
            *out.order_filled.last_mut().expect("order just pushed") = true;
        }
    } else {
        pending.push(Pending {
            fill_idx: fill_i,
            asset,
            side,
            qty,
            notional,
            kind,
            limit_price,
            order_idx,
            sl_stop,
            tp_stop,
            tsl_stop,
        });
    }
}

// --------------------------------------------------------------------------
// run_weights

fn weights_to_orders_at_bar(
    book: &LiveBook,
    bar_close: ArrayView1<'_, f64>,
    targets: ArrayView1<'_, f64>,
    tolerance: f64,
) -> Vec<(usize, u8, f64)> {
    let n = targets.len();
    let mut equity = book.cash;
    for j in 0..n {
        let q = book.qty[j];
        if q == 0.0 {
            continue;
        }
        let mut px = bar_close[j];
        if !px.is_finite() || px <= 0.0 {
            px = book.last_price[j];
        }
        if px.is_finite() && px > 0.0 {
            equity += q * px;
        }
    }
    if equity <= 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for j in 0..n {
        let tw = targets[j];
        if !tw.is_finite() {
            continue;
        }
        let px = bar_close[j];
        if !px.is_finite() || px <= 0.0 {
            continue;
        }
        let target_qty = (equity * tw) / px;
        let delta = target_qty - book.qty[j];
        if (delta * px).abs() < tolerance * equity {
            continue;
        }
        if delta == 0.0 {
            continue;
        }
        let side = if delta > 0.0 { SIDE_BUY } else { SIDE_SELL };
        out.push((j, side, delta.abs()));
    }
    out
}

/// Deterministic weights-path backtest.
///
/// `high` / `low` are accepted for parity with the bracket pipeline but
/// are unused by the weights path itself — the API has no surface to
/// attach `sl_stop` / `tp_stop` to a rebalance order, so [`check_intrabar_exits`]
/// finds nothing to fire from this entry point.
#[allow(clippy::too_many_arguments)]
pub fn run_weights(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    target_weights: ArrayView2<'_, f64>,
    target_bar_indices: ArrayView1<'_, i64>,
    cfg: SimCfg,
    tolerance: f64,
) -> SimOutput {
    let (n_bars, n_assets) = close.dim();
    let mut book = LiveBook::new(cfg.cash, n_assets);
    let mut out = SimOutput {
        equity: Array1::zeros(n_bars),
        ..Default::default()
    };
    // Bars that trigger a rebalance — exactly those in target_bar_indices.
    // Forward-filling would re-rebalance every bar and explode the orders log.
    let n_rows = target_bar_indices.len();
    let mut rebalance_row_at: Vec<Option<usize>> = vec![None; n_bars];
    for r in 0..n_rows {
        let i = target_bar_indices[r];
        if i >= 0 && (i as usize) < n_bars {
            rebalance_row_at[i as usize] = Some(r);
        }
    }

    let mut pending: Vec<Pending> = Vec::new();

    for (i, row_opt) in rebalance_row_at.iter().enumerate() {
        // Drain pending fills BEFORE the same-bar bracket check so a
        // position opened at this bar's open is visible to that check —
        // matches `Simulator._drive` so all three paths agree.
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);
        check_intrabar_exits(i, &open, &high, &low, &mut book, &mut out, &cfg);

        if let Some(r) = row_opt {
            let wrow = target_weights.row(*r);
            let orders = weights_to_orders_at_bar(&book, close.row(i), wrow, tolerance);
            for (asset, side, qty) in orders {
                submit_order(
                    i,
                    asset,
                    side,
                    qty,
                    0.0,
                    KIND_MARKET,
                    0.0,
                    n_bars,
                    &open,
                    &close,
                    &mut book,
                    &mut pending,
                    &mut out,
                    &cfg,
                    0.0,
                    0.0,
                    0.0,
                );
            }
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history
            .push((i, emit_weights(equity, &per_asset)));
    }
    out
}

// --------------------------------------------------------------------------
// run_orders

/// Execute a pre-built order log against the price panels.
///
/// Each order is submitted at its stated bar and filled according to the
/// execution and cost settings in `cfg`. `order_sl_stop` /
/// `order_tp_stop` / `order_tsl_stop` carry per-order bracket fractions
/// (`0.0` = no stop); non-zero entries activate the intra-bar exit
/// machinery for the resulting position.
#[allow(clippy::too_many_arguments)]
pub fn run_orders(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    order_bar: ArrayView1<'_, i64>,
    order_asset: ArrayView1<'_, i64>,
    order_side: ArrayView1<'_, i64>,
    order_qty: ArrayView1<'_, f64>,
    order_notional: ArrayView1<'_, f64>,
    order_kind: ArrayView1<'_, i64>,
    order_limit_price: ArrayView1<'_, f64>,
    order_sl_stop: ArrayView1<'_, f64>,
    order_tp_stop: ArrayView1<'_, f64>,
    order_tsl_stop: ArrayView1<'_, f64>,
    cfg: SimCfg,
) -> SimOutput {
    let (n_bars, n_assets) = close.dim();
    let mut book = LiveBook::new(cfg.cash, n_assets);
    let mut out = SimOutput {
        equity: Array1::zeros(n_bars),
        ..Default::default()
    };

    // Group orders by submission bar.
    let n_orders = order_bar.len();
    let mut by_bar: Vec<Vec<usize>> = vec![Vec::new(); n_bars];
    for k in 0..n_orders {
        let b = order_bar[k];
        if b >= 0 && (b as usize) < n_bars {
            by_bar[b as usize].push(k);
        }
    }

    let mut pending: Vec<Pending> = Vec::new();

    for (i, orders_at_bar) in by_bar.iter().enumerate() {
        // Drain BEFORE the bracket check (see run_weights for rationale).
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);
        check_intrabar_exits(i, &open, &high, &low, &mut book, &mut out, &cfg);

        for &k in orders_at_bar {
            let mut qty = order_qty[k];
            if !qty.is_finite() {
                qty = 0.0;
            }
            let mut notional = order_notional[k];
            if !notional.is_finite() {
                notional = 0.0;
            }
            let kind = order_kind[k] as u8;
            let mut limit = order_limit_price[k];
            if !limit.is_finite() {
                limit = 0.0;
            }
            let side = order_side[k] as u8;
            let asset = order_asset[k] as usize;
            let mut sl = order_sl_stop[k];
            if !sl.is_finite() || sl < 0.0 {
                sl = 0.0;
            }
            let mut tp = order_tp_stop[k];
            if !tp.is_finite() || tp < 0.0 {
                tp = 0.0;
            }
            let mut tsl = order_tsl_stop[k];
            if !tsl.is_finite() || tsl < 0.0 {
                tsl = 0.0;
            }
            submit_order(
                i,
                asset,
                side,
                qty,
                notional,
                kind,
                limit,
                n_bars,
                &open,
                &close,
                &mut book,
                &mut pending,
                &mut out,
                &cfg,
                sl,
                tp,
                tsl,
            );
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history
            .push((i, emit_weights(equity, &per_asset)));
    }
    out
}

// --------------------------------------------------------------------------
// run_signals

/// Signal-based entry/exit simulator.
///
/// `entries` and `exits` are boolean panels (0 = false, 1 = true) of shape
/// `n_bars × n_assets`. A `1` in `entries[i, j]` opens a long position in
/// asset `j` sized at `size` units; a `1` in `exits[i, j]` closes any open
/// position in that asset. `high` / `low` are accepted for parity with
/// the bracket pipeline but `run_signals` doesn't attach brackets to its
/// synthesised orders, so [`check_intrabar_exits`] is a no-op when driven
/// from this entry point. Returns a [`SimOutput`] with the resulting
/// trades and equity curve.
#[allow(clippy::too_many_arguments)]
pub fn run_signals(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    entries: ArrayView2<'_, u8>, // 0 = false, 1 = true
    exits: ArrayView2<'_, u8>,
    size: f64,
    cfg: SimCfg,
) -> SimOutput {
    let (n_bars, n_assets) = close.dim();
    let mut book = LiveBook::new(cfg.cash, n_assets);
    let mut out = SimOutput {
        equity: Array1::zeros(n_bars),
        ..Default::default()
    };
    let mut pending: Vec<Pending> = Vec::new();

    for i in 0..n_bars {
        // Drain BEFORE the bracket check (signals path doesn't attach
        // brackets so the check is a no-op here, but kept consistent
        // with run_weights / run_orders for parity).
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);
        check_intrabar_exits(i, &open, &high, &low, &mut book, &mut out, &cfg);

        let close_row = close.row(i);
        // Entries.
        for j in 0..n_assets {
            if entries[[i, j]] == 0 {
                continue;
            }
            let px = close_row[j];
            if !px.is_finite() || px <= 0.0 {
                continue;
            }
            let qty = (book.cash * size) / px;
            if qty <= 0.0 {
                continue;
            }
            submit_order(
                i,
                j,
                SIDE_BUY,
                qty,
                0.0,
                KIND_MARKET,
                0.0,
                n_bars,
                &open,
                &close,
                &mut book,
                &mut pending,
                &mut out,
                &cfg,
                0.0,
                0.0,
                0.0,
            );
        }
        // Exits.
        for j in 0..n_assets {
            if exits[[i, j]] == 0 {
                continue;
            }
            let held = book.qty[j];
            if held <= 0.0 {
                continue;
            }
            submit_order(
                i,
                j,
                SIDE_SELL,
                held,
                0.0,
                KIND_MARKET,
                0.0,
                n_bars,
                &open,
                &close,
                &mut book,
                &mut pending,
                &mut out,
                &cfg,
                0.0,
                0.0,
                0.0,
            );
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history
            .push((i, emit_weights(equity, &per_asset)));
    }
    out
}

// --------------------------------------------------------------------------
// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn cfg_default(cash: f64) -> SimCfg {
        SimCfg {
            cash,
            cost_kind: COST_NONE,
            cost_param1: 0.0,
            cost_param2: 0.0,
            slip_kind: SLIP_NONE,
            slip_param1: 0.0,
            exec_kind: EXEC_NEXT_BAR_OPEN,
        }
    }

    /// Build OHLCV panels for a single asset from a list of (o, h, l, c) tuples.
    fn make_ohlc_panels(
        rows: &[(f64, f64, f64, f64)],
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = rows.len();
        let mut o = Array2::<f64>::zeros((n, 1));
        let mut h = Array2::<f64>::zeros((n, 1));
        let mut l = Array2::<f64>::zeros((n, 1));
        let mut c = Array2::<f64>::zeros((n, 1));
        for (i, (op, hp, lp, cp)) in rows.iter().enumerate() {
            o[[i, 0]] = *op;
            h[[i, 0]] = *hp;
            l[[i, 0]] = *lp;
            c[[i, 0]] = *cp;
        }
        (o, h, l, c)
    }

    fn run_one_buy_with_brackets(
        ohlc: &[(f64, f64, f64, f64)],
        sl_stop: f64,
        tp_stop: f64,
    ) -> SimOutput {
        run_one_buy_with_brackets_full(ohlc, sl_stop, tp_stop, 0.0)
    }

    fn run_one_buy_with_brackets_full(
        ohlc: &[(f64, f64, f64, f64)],
        sl_stop: f64,
        tp_stop: f64,
        tsl_stop: f64,
    ) -> SimOutput {
        let (open_a, high_a, low_a, close_a) = make_ohlc_panels(ohlc);
        let order_bar = Array1::from(vec![0i64]);
        let order_asset = Array1::from(vec![0i64]);
        let order_side = Array1::from(vec![SIDE_BUY as i64]);
        let order_qty = Array1::from(vec![10.0]);
        let order_notional = Array1::from(vec![0.0]);
        let order_kind = Array1::from(vec![KIND_MARKET as i64]);
        let order_limit = Array1::from(vec![0.0]);
        let order_sl = Array1::from(vec![sl_stop]);
        let order_tp = Array1::from(vec![tp_stop]);
        let order_tsl = Array1::from(vec![tsl_stop]);
        run_orders(
            open_a.view(),
            close_a.view(),
            high_a.view(),
            low_a.view(),
            order_bar.view(),
            order_asset.view(),
            order_side.view(),
            order_qty.view(),
            order_notional.view(),
            order_kind.view(),
            order_limit.view(),
            order_sl.view(),
            order_tp.view(),
            order_tsl.view(),
            cfg_default(100_000.0),
        )
    }

    fn run_one_short_with_brackets_full(
        ohlc: &[(f64, f64, f64, f64)],
        sl_stop: f64,
        tp_stop: f64,
        tsl_stop: f64,
    ) -> SimOutput {
        let (open_a, high_a, low_a, close_a) = make_ohlc_panels(ohlc);
        let order_bar = Array1::from(vec![0i64]);
        let order_asset = Array1::from(vec![0i64]);
        let order_side = Array1::from(vec![SIDE_SELL as i64]);
        let order_qty = Array1::from(vec![10.0]);
        let order_notional = Array1::from(vec![0.0]);
        let order_kind = Array1::from(vec![KIND_MARKET as i64]);
        let order_limit = Array1::from(vec![0.0]);
        let order_sl = Array1::from(vec![sl_stop]);
        let order_tp = Array1::from(vec![tp_stop]);
        let order_tsl = Array1::from(vec![tsl_stop]);
        run_orders(
            open_a.view(),
            close_a.view(),
            high_a.view(),
            low_a.view(),
            order_bar.view(),
            order_asset.view(),
            order_side.view(),
            order_qty.view(),
            order_notional.view(),
            order_kind.view(),
            order_limit.view(),
            order_sl.view(),
            order_tp.view(),
            order_tsl.view(),
            cfg_default(100_000.0),
        )
    }

    #[test]
    fn long_sl_fires_at_level_when_low_pierces() {
        // bar 0: signal; bar 1: fill at open=100; bar 2: low=88 ≤ 90 → SL at 90.
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (95.0, 96.0, 88.0, 92.0),
            ],
            0.10,
            0.0,
        );
        assert_eq!(out.trade_bar.len(), 2);
        assert_eq!(out.trade_reason[0], REASON_SIGNAL);
        assert_eq!(out.trade_reason[1], REASON_STOP_LOSS);
        assert!((out.trade_price[1] - 90.0).abs() < 1e-9);
        assert!((out.trade_qty[1] - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn long_sl_gap_down_fills_at_open_not_level() {
        // bar 2 opens at 85, below SL=90 → fill at 85 (worse than stop).
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (85.0, 87.0, 80.0, 82.0),
            ],
            0.10,
            0.0,
        );
        assert_eq!(out.trade_reason[1], REASON_STOP_LOSS);
        assert!((out.trade_price[1] - 85.0).abs() < 1e-9);
    }

    #[test]
    fn long_tp_fires_at_level_when_high_pierces() {
        // bar 2 high=115 ≥ TP=110 → take-profit at 110.
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (108.0, 115.0, 107.0, 112.0),
            ],
            0.0,
            0.10,
        );
        assert_eq!(out.trade_reason[1], REASON_TAKE_PROFIT);
        assert!((out.trade_price[1] - 110.0).abs() < 1e-9);
    }

    #[test]
    fn long_tp_gap_up_fills_at_open_better_than_level() {
        // bar 2 opens at 115, above TP=110 → fill at 115 (favourable gap).
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (115.0, 118.0, 114.0, 117.0),
            ],
            0.0,
            0.10,
        );
        assert_eq!(out.trade_reason[1], REASON_TAKE_PROFIT);
        assert!((out.trade_price[1] - 115.0).abs() < 1e-9);
    }

    #[test]
    fn short_sl_fires_at_level_when_high_pierces() {
        // Short at 100; SL = 110. bar 2 high=112 → fire at 110.
        let out = run_one_short_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (105.0, 112.0, 104.0, 108.0),
            ],
            0.10,
            0.0,
            0.0,
        );
        assert_eq!(out.trade_qty[0], -10.0); // short open
        assert_eq!(out.trade_qty[1], 10.0); // cover
        assert_eq!(out.trade_reason[1], REASON_STOP_LOSS);
        assert!((out.trade_price[1] - 110.0).abs() < 1e-9);
    }

    #[test]
    fn sl_wins_when_both_could_fire_same_bar() {
        // Bar 2 high=115 ≥ TP=110 AND low=88 ≤ SL=90 → SL wins.
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 115.0, 88.0, 105.0),
            ],
            0.10,
            0.10,
        );
        assert_eq!(out.trade_reason[1], REASON_STOP_LOSS);
        assert!((out.trade_price[1] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn levels_cleared_after_fire_so_subsequent_bars_dont_retrigger() {
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (95.0, 96.0, 88.0, 92.0),  // SL fires
                (95.0, 100.0, 89.0, 99.0), // would fire again if not cleared
            ],
            0.10,
            0.0,
        );
        let stop_count = out
            .trade_reason
            .iter()
            .filter(|&&r| r == REASON_STOP_LOSS)
            .count();
        assert_eq!(stop_count, 1);
    }

    #[test]
    fn check_intrabar_exits_skips_nan_bars() {
        // Bar 2 has NaN high/low — check should skip without crashing.
        let mut o = Array2::<f64>::zeros((3, 1));
        let mut h = Array2::<f64>::zeros((3, 1));
        let mut l = Array2::<f64>::zeros((3, 1));
        let mut c = Array2::<f64>::zeros((3, 1));
        for (i, (op, hp, lp, cp)) in [
            (100.0, 102.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        ]
        .iter()
        .enumerate()
        {
            o[[i, 0]] = *op;
            h[[i, 0]] = *hp;
            l[[i, 0]] = *lp;
            c[[i, 0]] = *cp;
        }
        let order_bar = Array1::from(vec![0i64]);
        let order_asset = Array1::from(vec![0i64]);
        let order_side = Array1::from(vec![SIDE_BUY as i64]);
        let order_qty = Array1::from(vec![10.0]);
        let order_notional = Array1::from(vec![0.0]);
        let order_kind = Array1::from(vec![KIND_MARKET as i64]);
        let order_limit = Array1::from(vec![0.0]);
        let order_sl = Array1::from(vec![0.10]);
        let order_tp = Array1::from(vec![0.0]);
        let order_tsl = Array1::from(vec![0.0]);
        let out = run_orders(
            o.view(),
            c.view(),
            h.view(),
            l.view(),
            order_bar.view(),
            order_asset.view(),
            order_side.view(),
            order_qty.view(),
            order_notional.view(),
            order_kind.view(),
            order_limit.view(),
            order_sl.view(),
            order_tp.view(),
            order_tsl.view(),
            cfg_default(100_000.0),
        );
        // Only the entry trade should fire; the SL check on the NaN bar is a no-op.
        assert_eq!(out.trade_bar.len(), 1);
    }

    #[test]
    fn no_stops_means_no_extra_reason_codes() {
        let out = run_one_buy_with_brackets(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (95.0, 96.0, 88.0, 92.0),
            ],
            0.0,
            0.0,
        );
        assert!(out.trade_reason.iter().all(|&r| r == REASON_SIGNAL));
    }

    // ----------------------------------------------------------- trailing-stop tests

    #[test]
    fn long_tsl_fires_at_ratcheted_level() {
        // Bar 0: signal. Bar 1: fill at 100; anchor=100, tsl=90.
        // Bar 2: post-open anchor stays at 100 (open=102 > 100 → ratchets to 102, but the
        //        bar.high=110 ratchet only happens AFTER trigger check), so trigger uses
        //        level=102*0.9=91.8. low=95 > 91.8 → no fire. Post-trigger ratchet pushes
        //        anchor to 110.
        // Bar 3: post-open anchor stays at 110 (open=105 < 110), level=110*0.9=99,
        //        low=95 ≤ 99 → fire at 99.
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (102.0, 110.0, 95.0, 96.0),
                (105.0, 108.0, 95.0, 98.0),
            ],
            0.0,
            0.0,
            0.10,
        );
        assert_eq!(out.trade_reason[0], REASON_SIGNAL);
        assert_eq!(out.trade_reason[1], REASON_TRAILING_STOP);
        assert!((out.trade_price[1] - 99.0).abs() < 1e-9);
    }

    #[test]
    fn long_tsl_gap_down_fills_at_open_using_old_anchor() {
        // Bar 2: ratchet anchor to 120 without firing (low=110 > new tsl=108).
        // Bar 3: open=90, below tsl=108 → fire at 90.
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (110.0, 120.0, 110.0, 115.0),
                (90.0, 92.0, 85.0, 88.0),
            ],
            0.0,
            0.0,
            0.10,
        );
        let tsl_idx = out
            .trade_reason
            .iter()
            .position(|&r| r == REASON_TRAILING_STOP)
            .expect("tsl should fire");
        assert!((out.trade_price[tsl_idx] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn short_tsl_fires_at_ratcheted_level() {
        // Short fills at 100; anchor=100, tsl=110. Bar 2 low=90 → anchor=90, tsl=99.
        // Bar 3 high=102 ≥ 99 → fire at 99.
        let out = run_one_short_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (95.0, 96.0, 90.0, 92.0),
                (97.0, 102.0, 96.0, 100.0),
            ],
            0.0,
            0.0,
            0.10,
        );
        let tsl_idx = out
            .trade_reason
            .iter()
            .position(|&r| r == REASON_TRAILING_STOP)
            .expect("tsl should fire");
        assert!((out.trade_price[tsl_idx] - 99.0).abs() < 1e-9);
    }

    #[test]
    fn short_tsl_gap_up_fills_at_open_using_old_anchor() {
        // Bar 2: ratchet anchor to 80 without firing (high=86 < new tsl=88).
        // Bar 3: open=95, above tsl=88 → fire at 95.
        let out = run_one_short_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (85.0, 86.0, 80.0, 85.0),
                (95.0, 98.0, 92.0, 96.0),
            ],
            0.0,
            0.0,
            0.10,
        );
        let tsl_idx = out
            .trade_reason
            .iter()
            .position(|&r| r == REASON_TRAILING_STOP)
            .expect("tsl should fire");
        assert!((out.trade_price[tsl_idx] - 95.0).abs() < 1e-9);
    }

    #[test]
    fn tsl_wins_over_fixed_sl_when_tighter() {
        // sl_stop=0.10 (level=90), tsl_stop=0.05.
        // Bar 1: fill at 100; anchor=100, tsl level=95.
        // Bar 2: open=102 → post-open anchor=102, level=96.9. low=101 > 96.9, no fire.
        //        Post-trigger ratchet → anchor=120.
        // Bar 3: open=115 (above tsl=114, no gap), low=110 ≤ 114 → TSL fires at 114
        //        (tighter than fixed SL=90 → trail wins).
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (102.0, 120.0, 101.0, 108.0),
                (115.0, 116.0, 110.0, 113.0),
            ],
            0.10,
            0.0,
            0.05,
        );
        let forced = out
            .trade_reason
            .iter()
            .filter(|&&r| r == REASON_STOP_LOSS || r == REASON_TRAILING_STOP)
            .count();
        assert_eq!(forced, 1);
        let idx = out
            .trade_reason
            .iter()
            .position(|&r| r != REASON_SIGNAL)
            .expect("forced exit");
        assert_eq!(out.trade_reason[idx], REASON_TRAILING_STOP);
        assert!((out.trade_price[idx] - 114.0).abs() < 1e-9);
    }

    #[test]
    fn fixed_sl_wins_over_tsl_when_tighter() {
        // sl_stop=0.10 (level=90), tsl_stop=0.20 (anchor=100, tsl=80).
        // Bar 2 low=88 ≤ 90 → fixed SL fires at 90 before TSL would.
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (98.0, 100.0, 88.0, 92.0),
            ],
            0.10,
            0.0,
            0.20,
        );
        let idx = out
            .trade_reason
            .iter()
            .position(|&r| r != REASON_SIGNAL)
            .expect("forced exit");
        assert_eq!(out.trade_reason[idx], REASON_STOP_LOSS);
        assert!((out.trade_price[idx] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn stops_beat_tp_with_tsl_present() {
        // tsl=0.05, tp=0.10. Bar 2 high=115 ratchets anchor and pierces TP=110;
        // simultaneously low=90 pierces tsl=109.25 — TSL wins (it's a stop).
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (100.0, 115.0, 90.0, 100.0),
            ],
            0.0,
            0.10,
            0.05,
        );
        let idx = out
            .trade_reason
            .iter()
            .position(|&r| r != REASON_SIGNAL)
            .expect("forced exit");
        assert_eq!(out.trade_reason[idx], REASON_TRAILING_STOP);
    }

    #[test]
    fn tsl_levels_cleared_after_fire() {
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0),
                (100.0, 101.0, 99.0, 100.0),
                (102.0, 110.0, 95.0, 96.0), // tsl fires at 99
                (95.0, 100.0, 89.0, 99.0),  // would re-fire if tsl wasn't cleared
            ],
            0.0,
            0.0,
            0.10,
        );
        let tsl_count = out
            .trade_reason
            .iter()
            .filter(|&&r| r == REASON_TRAILING_STOP)
            .count();
        assert_eq!(tsl_count, 1);
    }

    #[test]
    fn check_intrabar_exits_skips_nan_bars_with_tsl() {
        // Bar 2 has NaN OHLC — TSL check must not crash.
        let mut o = Array2::<f64>::zeros((3, 1));
        let mut h = Array2::<f64>::zeros((3, 1));
        let mut l = Array2::<f64>::zeros((3, 1));
        let mut c = Array2::<f64>::zeros((3, 1));
        for (i, (op, hp, lp, cp)) in [
            (100.0, 102.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        ]
        .iter()
        .enumerate()
        {
            o[[i, 0]] = *op;
            h[[i, 0]] = *hp;
            l[[i, 0]] = *lp;
            c[[i, 0]] = *cp;
        }
        let order_bar = Array1::from(vec![0i64]);
        let order_asset = Array1::from(vec![0i64]);
        let order_side = Array1::from(vec![SIDE_BUY as i64]);
        let order_qty = Array1::from(vec![10.0]);
        let order_notional = Array1::from(vec![0.0]);
        let order_kind = Array1::from(vec![KIND_MARKET as i64]);
        let order_limit = Array1::from(vec![0.0]);
        let order_sl = Array1::from(vec![0.0]);
        let order_tp = Array1::from(vec![0.0]);
        let order_tsl = Array1::from(vec![0.10]);
        let out = run_orders(
            o.view(),
            c.view(),
            h.view(),
            l.view(),
            order_bar.view(),
            order_asset.view(),
            order_side.view(),
            order_qty.view(),
            order_notional.view(),
            order_kind.view(),
            order_limit.view(),
            order_sl.view(),
            order_tp.view(),
            order_tsl.view(),
            cfg_default(100_000.0),
        );
        // Only the entry trade should fire.
        assert_eq!(out.trade_bar.len(), 1);
    }

    #[test]
    fn pending_carries_tsl_to_fill_bar() {
        // Order submitted at bar 0 fills at bar 1. Verify the tsl_stop
        // attached to the submission produces a tsl exit on bar 3
        // (the post-trigger ratchet on bar 2 sets anchor=110 → bar 3
        // level=99 → low=95 fires).
        let out = run_one_buy_with_brackets_full(
            &[
                (100.0, 102.0, 99.0, 100.0), // signal — pending until bar 1
                (100.0, 101.0, 99.0, 100.0), // fill at 100; anchor=100
                (102.0, 110.0, 95.0, 96.0),  // post-trigger ratchet → anchor=110
                (105.0, 108.0, 95.0, 98.0),  // level=99, low=95 → tsl fires at 99
            ],
            0.0,
            0.0,
            0.10,
        );
        assert_eq!(out.trade_reason.len(), 2);
        assert_eq!(out.trade_reason[1], REASON_TRAILING_STOP);
    }
}
