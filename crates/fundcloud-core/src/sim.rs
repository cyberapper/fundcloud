//! Deterministic simulator kernels — Rust parity reference for
//! [`fundcloud.kernels._sim_pyfallback`].
//!
//! Three entry points: [`run_weights`], [`run_orders`], [`run_signals`].
//! Each one consumes NumPy panels + a [`SimCfg`] and returns a [`SimOutput`]
//! struct-of-arrays. The PyO3 bindings in the sibling `fundcloud-py` crate
//! thin-wrap these.
//!
//! The inner loop holds no Python objects, so the PyO3 bindings run
//! everything under `py.allow_threads` for GIL-released execution.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// --------------------------------------------------------------------------
// Config / enum tags — must stay numerically identical to the Python
// constants in `fundcloud.kernels._sim_pyfallback`.

/// Cost-model tag matching `_sim_pyfallback.COST_NONE` / `_FIXED_BPS` / `_PER_SHARE`.
#[derive(Copy, Clone, Debug)]
pub struct SimCfg {
    pub cash: f64,
    pub cost_kind: u8,
    pub cost_param1: f64, // bps for FixedBps, rate for PerShare, else 0
    pub cost_param2: f64, // minimum
    pub slip_kind: u8,
    pub slip_param1: f64, // bps for HalfSpread
    pub exec_kind: u8,
}

pub const COST_NONE: u8 = 0;
pub const COST_FIXED_BPS: u8 = 1;
pub const COST_PER_SHARE: u8 = 2;

pub const SLIP_NONE: u8 = 0;
pub const SLIP_HALF_SPREAD: u8 = 1;

pub const EXEC_NEXT_BAR_OPEN: u8 = 0;
pub const EXEC_SAME_BAR_CLOSE: u8 = 1;

pub const SIDE_BUY: u8 = 0;
pub const SIDE_SELL: u8 = 1;

pub const KIND_MARKET: u8 = 0;
pub const KIND_LIMIT: u8 = 1;

// --------------------------------------------------------------------------
// Output struct-of-arrays. Every field is `Vec<...>` in trade / order order
// and `Array1<...>` for the per-bar equity / weights history.

#[derive(Debug, Default)]
pub struct SimOutput {
    pub equity: Array1<f64>, // (n_bars,)
    pub weights_history: Vec<(usize, Vec<(usize, f64)>)>, // (bar_idx, [(asset_idx, weight)])
    pub trade_bar: Vec<usize>,
    pub trade_asset: Vec<usize>,
    pub trade_qty: Vec<f64>,      // signed
    pub trade_price: Vec<f64>,
    pub trade_fee: Vec<f64>,
    pub trade_slip_bps: Vec<f64>,
    pub order_bar: Vec<usize>,
    pub order_asset: Vec<usize>,
    pub order_side: Vec<u8>,
    pub order_qty: Vec<f64>,
    pub order_notional: Vec<f64>,
    pub order_kind: Vec<u8>,
    pub order_limit_price: Vec<f64>,
    pub order_filled: Vec<bool>,
}

// --------------------------------------------------------------------------
// Shared primitives.

fn apply_slippage(price: f64, side: u8, slip_kind: u8, slip_p1: f64) -> (f64, f64) {
    if slip_kind == SLIP_HALF_SPREAD && price > 0.0 {
        let half = slip_p1 / 2.0;
        let adj = price * (half * 1e-4);
        let px = if side == SIDE_BUY { price + adj } else { price - adj };
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

fn fill_idx_for(signal_idx: usize, n_bars: usize, exec_kind: u8) -> i64 {
    if exec_kind == EXEC_SAME_BAR_CLOSE {
        signal_idx as i64
    } else {
        let nxt = signal_idx + 1;
        if nxt < n_bars {
            nxt as i64
        } else {
            -1
        }
    }
}

// --------------------------------------------------------------------------
// LiveBook mirrors _sim_pyfallback._LiveBook.

struct LiveBook {
    cash: f64,
    qty: Array1<f64>,
    avg_cost: Array1<f64>,
    last_price: Array1<f64>,
}

impl LiveBook {
    fn new(cash: f64, n_assets: usize) -> Self {
        Self {
            cash,
            qty: Array1::zeros(n_assets),
            avg_cost: Array1::zeros(n_assets),
            last_price: Array1::zeros(n_assets),
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
    if signed > 0.0 && new_qty > 0.0 {
        book.avg_cost[asset_idx] = (prev_qty * prev_cost + signed * fill_price) / new_qty;
    } else if new_qty == 0.0 {
        book.avg_cost[asset_idx] = 0.0;
    }
    book.qty[asset_idx] = new_qty;
    book.cash -= signed * fill_price + f;
    book.last_price[asset_idx] = fill_price;
    Some((signed, fill_price, f, slip_bps))
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

// Pending-order queue element.
#[derive(Copy, Clone)]
struct Pending {
    fill_idx: usize,
    asset: usize,
    side: u8,
    qty: f64,
    notional: f64,
    kind: u8,
    limit_price: f64,
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
        ) {
            out.trade_bar.push(i);
            out.trade_asset.push(p.asset);
            out.trade_qty.push(signed);
            out.trade_price.push(price);
            out.trade_fee.push(f);
            out.trade_slip_bps.push(slip);
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
    out.order_filled.push(true);
    let fill_i = fill_i as usize;
    if fill_i == i {
        let ref_price = exec_price_at(open, close, cfg.exec_kind, i, asset);
        if let Some((signed, price, f, slip)) = execute_one(
            book, asset, side, qty, notional, kind, limit_price, ref_price,
            cfg.cost_kind, cfg.cost_param1, cfg.cost_param2,
            cfg.slip_kind, cfg.slip_param1,
        ) {
            out.trade_bar.push(i);
            out.trade_asset.push(asset);
            out.trade_qty.push(signed);
            out.trade_price.push(price);
            out.trade_fee.push(f);
            out.trade_slip_bps.push(slip);
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
pub fn run_weights(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
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
        let i = target_bar_indices[r] as i64;
        if i >= 0 && (i as usize) < n_bars {
            rebalance_row_at[i as usize] = Some(r);
        }
    }

    let mut pending: Vec<Pending> = Vec::new();

    for i in 0..n_bars {
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);

        if let Some(r) = rebalance_row_at[i] {
            let wrow = target_weights.row(r);
            let orders = weights_to_orders_at_bar(&book, close.row(i), wrow, tolerance);
            for (asset, side, qty) in orders {
                submit_order(
                    i, asset, side, qty, 0.0, KIND_MARKET, 0.0,
                    n_bars, &open, &close, &mut book, &mut pending, &mut out, &cfg,
                );
            }
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history.push((i, emit_weights(equity, &per_asset)));
    }
    out
}

// --------------------------------------------------------------------------
// run_orders

#[allow(clippy::too_many_arguments)]
pub fn run_orders(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    order_bar: ArrayView1<'_, i64>,
    order_asset: ArrayView1<'_, i64>,
    order_side: ArrayView1<'_, i64>,
    order_qty: ArrayView1<'_, f64>,
    order_notional: ArrayView1<'_, f64>,
    order_kind: ArrayView1<'_, i64>,
    order_limit_price: ArrayView1<'_, f64>,
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

    for i in 0..n_bars {
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);

        for &k in &by_bar[i] {
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
            submit_order(
                i, asset, side, qty, notional, kind, limit,
                n_bars, &open, &close, &mut book, &mut pending, &mut out, &cfg,
            );
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history.push((i, emit_weights(equity, &per_asset)));
    }
    out
}

// --------------------------------------------------------------------------
// run_signals

pub fn run_signals(
    open: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
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
        drain_pending(&mut pending, i, &open, &close, &mut book, &mut out, &cfg);

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
                i, j, SIDE_BUY, qty, 0.0, KIND_MARKET, 0.0,
                n_bars, &open, &close, &mut book, &mut pending, &mut out, &cfg,
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
                i, j, SIDE_SELL, held, 0.0, KIND_MARKET, 0.0,
                n_bars, &open, &close, &mut book, &mut pending, &mut out, &cfg,
            );
        }

        let (equity, per_asset) = mark_to_market(&mut book, close.row(i));
        out.equity[i] = equity;
        out.weights_history.push((i, emit_weights(equity, &per_asset)));
    }
    out
}
