//! PyO3 bindings for `fundcloud-core`.
//!
//! Imports from Python as `fundcloud._core`. Surfaces the full kernel
//! suite: rolling reductions, drawdown analytics, risk-adjusted moments,
//! and tail-risk estimators. Every public function releases the GIL via

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fundcloud_core::{
    drawdown as core_drawdown, moments as core_moments, patterns as core_patterns,
    returns as core_returns, rolling as core_rolling, sim as core_sim, tail_risk as core_tail,
};

#[pyfunction]
fn kernel_version() -> &'static str {
    fundcloud_core::kernel_version()
}

// ---------------------------------------------------------------------- returns

#[pyfunction]
#[pyo3(text_signature = "(prices, /)")]
fn returns_from_prices<'py>(
    py: Python<'py>,
    prices: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = prices.as_array();
    let out = py.detach(|| core_returns::returns_from_prices(view));
    out.into_pyarray(py)
}

// ---------------------------------------------------------------------- rolling

#[pyfunction]
#[pyo3(text_signature = "(x, window, /)")]
fn rolling_mean<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    window: usize,
) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let out = py.detach(|| core_rolling::rolling_mean(view, window));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(x, window, ddof=1, /)")]
fn rolling_std<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    window: usize,
    ddof: usize,
) -> Bound<'py, PyArray1<f64>> {
    let view = x.as_array();
    let out = py.detach(|| core_rolling::rolling_std(view, window, ddof));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(x, window, /)")]
fn rolling_mean_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    window: usize,
) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.detach(|| core_rolling::rolling_mean_batch(view, window));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(x, window, ddof=1, /)")]
fn rolling_std_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    window: usize,
    ddof: usize,
) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.detach(|| core_rolling::rolling_std_batch(view, window, ddof));
    out.into_pyarray(py)
}

// -------------------------------------------------------------------- drawdown

#[pyfunction]
#[pyo3(text_signature = "(returns, /)")]
fn drawdown_series<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_drawdown::drawdown_series(view));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(returns, /)")]
fn max_drawdown_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_drawdown::max_drawdown_batch(view));
    out.into_pyarray(py)
}

// --------------------------------------------------------------------- moments

#[pyfunction]
#[pyo3(text_signature = "(returns, rf_per_period, periods_per_year, /)")]
fn sharpe_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    rf_per_period: f64,
    periods_per_year: f64,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_moments::sharpe_batch(view, rf_per_period, periods_per_year));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(returns, target, periods_per_year, /)")]
fn sortino_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    target: f64,
    periods_per_year: f64,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_moments::sortino_batch(view, target, periods_per_year));
    out.into_pyarray(py)
}

// ------------------------------------------------------------------- tail risk

#[pyfunction]
#[pyo3(text_signature = "(returns, alpha, /)")]
fn var_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_tail::var_batch(view, alpha));
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(text_signature = "(returns, alpha, /)")]
fn cvar_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.detach(|| core_tail::cvar_batch(view, alpha));
    out.into_pyarray(py)
}

// ---------------------------------------------------------------------- sim

fn sim_output_to_dict<'py>(py: Python<'py>, out: core_sim::SimOutput) -> Bound<'py, PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("equity", out.equity.into_pyarray(py))
        .expect("set equity");
    // weights_history = list of (bar_idx, dict[asset_idx->weight])
    let wh = PyList::empty(py);
    for (bar, pairs) in &out.weights_history {
        let inner = PyDict::new(py);
        for (aj, w) in pairs {
            inner.set_item(*aj, *w).expect("set weight");
        }
        wh.append((*bar, inner)).expect("append wh entry");
    }
    dict.set_item("weights_history", wh)
        .expect("set weights_history");
    dict.set_item("trade_bar", out.trade_bar)
        .expect("set trade_bar");
    dict.set_item("trade_asset", out.trade_asset)
        .expect("set trade_asset");
    dict.set_item("trade_qty", out.trade_qty)
        .expect("set trade_qty");
    dict.set_item("trade_price", out.trade_price)
        .expect("set trade_price");
    dict.set_item("trade_fee", out.trade_fee)
        .expect("set trade_fee");
    dict.set_item("trade_slip_bps", out.trade_slip_bps)
        .expect("set trade_slip_bps");
    // u8 reason codes; the Python dispatcher
    // (``fundcloud.sim.simulator._rehydrate_sim_result``) translates them
    // to ``"signal"`` / ``"stop_loss"`` / ``"take_profit"`` /
    // ``"trailing_stop"`` via the shared ``REASON_*`` mapping.
    dict.set_item("trade_reason", out.trade_reason)
        .expect("set trade_reason");
    dict.set_item("order_bar", out.order_bar)
        .expect("set order_bar");
    dict.set_item("order_asset", out.order_asset)
        .expect("set order_asset");
    dict.set_item("order_side", out.order_side)
        .expect("set order_side");
    dict.set_item("order_qty", out.order_qty)
        .expect("set order_qty");
    dict.set_item("order_notional", out.order_notional)
        .expect("set order_notional");
    dict.set_item("order_kind", out.order_kind)
        .expect("set order_kind");
    dict.set_item("order_limit_price", out.order_limit_price)
        .expect("set order_limit_price");
    dict.set_item("order_filled", out.order_filled)
        .expect("set order_filled");
    dict
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, high_panel, low_panel, target_weights, \
                      target_bar_indices, cash, cost_kind, cost_p1, cost_p2, slip_kind, \
                      slip_p1, exec_kind, tolerance, /)"
)]
fn sim_run_weights<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
    high_panel: PyReadonlyArray2<'py, f64>,
    low_panel: PyReadonlyArray2<'py, f64>,
    target_weights: PyReadonlyArray2<'py, f64>,
    target_bar_indices: PyReadonlyArray1<'py, i64>,
    cash: f64,
    cost_kind: u8,
    cost_p1: f64,
    cost_p2: f64,
    slip_kind: u8,
    slip_p1: f64,
    exec_kind: u8,
    tolerance: f64,
) -> Bound<'py, PyDict> {
    let open_v = open_panel.as_array();
    let close_v = close_panel.as_array();
    let high_v = high_panel.as_array();
    let low_v = low_panel.as_array();
    let tw_v = target_weights.as_array();
    let tbi_v = target_bar_indices.as_array();
    assert_eq!(
        open_v.dim(),
        close_v.dim(),
        "open and close panels must have the same shape"
    );
    assert_eq!(open_v.dim(), high_v.dim(), "high panel shape mismatch");
    assert_eq!(open_v.dim(), low_v.dim(), "low panel shape mismatch");
    assert!(
        tbi_v.len() <= tw_v.nrows(),
        "target_bar_indices length ({}) exceeds target_weights rows ({})",
        tbi_v.len(),
        tw_v.nrows()
    );
    let cfg = core_sim::SimCfg {
        cash,
        cost_kind,
        cost_param1: cost_p1,
        cost_param2: cost_p2,
        slip_kind,
        slip_param1: slip_p1,
        exec_kind,
    };
    let out = py.detach(|| {
        core_sim::run_weights(open_v, close_v, high_v, low_v, tw_v, tbi_v, cfg, tolerance)
    });
    sim_output_to_dict(py, out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, high_panel, low_panel, order_bar, order_asset, \
                      order_side, order_qty, order_notional, order_kind, order_limit_price, \
                      order_sl_stop, order_tp_stop, order_tsl_stop, cash, cost_kind, cost_p1, \
                      cost_p2, slip_kind, slip_p1, exec_kind, /)"
)]
fn sim_run_orders<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
    high_panel: PyReadonlyArray2<'py, f64>,
    low_panel: PyReadonlyArray2<'py, f64>,
    order_bar: PyReadonlyArray1<'py, i64>,
    order_asset: PyReadonlyArray1<'py, i64>,
    order_side: PyReadonlyArray1<'py, i64>,
    order_qty: PyReadonlyArray1<'py, f64>,
    order_notional: PyReadonlyArray1<'py, f64>,
    order_kind: PyReadonlyArray1<'py, i64>,
    order_limit_price: PyReadonlyArray1<'py, f64>,
    order_sl_stop: PyReadonlyArray1<'py, f64>,
    order_tp_stop: PyReadonlyArray1<'py, f64>,
    order_tsl_stop: PyReadonlyArray1<'py, f64>,
    cash: f64,
    cost_kind: u8,
    cost_p1: f64,
    cost_p2: f64,
    slip_kind: u8,
    slip_p1: f64,
    exec_kind: u8,
) -> Bound<'py, PyDict> {
    let cfg = core_sim::SimCfg {
        cash,
        cost_kind,
        cost_param1: cost_p1,
        cost_param2: cost_p2,
        slip_kind,
        slip_param1: slip_p1,
        exec_kind,
    };
    let op = open_panel.as_array();
    let cl = close_panel.as_array();
    let hi = high_panel.as_array();
    let lo = low_panel.as_array();
    let ob = order_bar.as_array();
    let oa = order_asset.as_array();
    let os = order_side.as_array();
    let oq = order_qty.as_array();
    let on = order_notional.as_array();
    let ok_ = order_kind.as_array();
    let olp = order_limit_price.as_array();
    let osl = order_sl_stop.as_array();
    let otp = order_tp_stop.as_array();
    let otsl = order_tsl_stop.as_array();
    let n_orders = ob.len();
    assert_eq!(op.dim(), hi.dim(), "high panel shape mismatch");
    assert_eq!(op.dim(), lo.dim(), "low panel shape mismatch");
    assert_eq!(oa.len(), n_orders, "order_asset length mismatch");
    assert_eq!(os.len(), n_orders, "order_side length mismatch");
    assert_eq!(oq.len(), n_orders, "order_qty length mismatch");
    assert_eq!(on.len(), n_orders, "order_notional length mismatch");
    assert_eq!(ok_.len(), n_orders, "order_kind length mismatch");
    assert_eq!(olp.len(), n_orders, "order_limit_price length mismatch");
    assert_eq!(osl.len(), n_orders, "order_sl_stop length mismatch");
    assert_eq!(otp.len(), n_orders, "order_tp_stop length mismatch");
    assert_eq!(otsl.len(), n_orders, "order_tsl_stop length mismatch");
    let out = py.detach(|| {
        core_sim::run_orders(
            op, cl, hi, lo, ob, oa, os, oq, on, ok_, olp, osl, otp, otsl, cfg,
        )
    });
    sim_output_to_dict(py, out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, high_panel, low_panel, entries, exits, \
                      size, cash, cost_kind, cost_p1, cost_p2, slip_kind, slip_p1, exec_kind, /)"
)]
fn sim_run_signals<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
    high_panel: PyReadonlyArray2<'py, f64>,
    low_panel: PyReadonlyArray2<'py, f64>,
    entries: PyReadonlyArray2<'py, u8>,
    exits: PyReadonlyArray2<'py, u8>,
    size: f64,
    cash: f64,
    cost_kind: u8,
    cost_p1: f64,
    cost_p2: f64,
    slip_kind: u8,
    slip_p1: f64,
    exec_kind: u8,
) -> Bound<'py, PyDict> {
    let cfg = core_sim::SimCfg {
        cash,
        cost_kind,
        cost_param1: cost_p1,
        cost_param2: cost_p2,
        slip_kind,
        slip_param1: slip_p1,
        exec_kind,
    };
    let op = open_panel.as_array();
    let cl = close_panel.as_array();
    let hi = high_panel.as_array();
    let lo = low_panel.as_array();
    let en = entries.as_array();
    let ex = exits.as_array();
    assert_eq!(en.dim(), cl.dim(), "entries shape must match close_panel");
    assert_eq!(ex.dim(), cl.dim(), "exits shape must match close_panel");
    assert_eq!(hi.dim(), cl.dim(), "high panel shape mismatch");
    assert_eq!(lo.dim(), cl.dim(), "low panel shape mismatch");
    let out = py.detach(|| core_sim::run_signals(op, cl, hi, lo, en, ex, size, cfg));
    sim_output_to_dict(py, out)
}

// ----------------------------------------------------------------- patterns

fn pivot_to_dict<'py>(py: Python<'py>, p: &core_patterns::Pivot) -> Bound<'py, PyDict> {
    let d = PyDict::new_bound(py);
    d.set_item("index", p.index).expect("set index");
    d.set_item("ts_ns", p.ts_ns).expect("set ts_ns");
    d.set_item("price", p.price).expect("set price");
    d.set_item("kind", p.kind.as_str()).expect("set kind");
    d.set_item("order", p.order).expect("set order");
    d
}

fn trendline_to_dict<'py>(py: Python<'py>, tl: &core_patterns::TrendLine) -> Bound<'py, PyDict> {
    let d = PyDict::new_bound(py);
    d.set_item("start_index", tl.start_index)
        .expect("set start_index");
    d.set_item("end_index", tl.end_index).expect("set end_index");
    d.set_item("slope", tl.slope).expect("set slope");
    d.set_item("intercept", tl.intercept).expect("set intercept");
    d.set_item("r_squared", tl.r_squared).expect("set r_squared");
    d.set_item("touch_count", tl.touch_count)
        .expect("set touch_count");
    d
}

fn detection_to_dict<'py>(py: Python<'py>, d: &core_patterns::Detection) -> Bound<'py, PyDict> {
    let out = PyDict::new_bound(py);
    out.set_item("name", d.pattern.name).expect("set name");
    out.set_item("direction", d.pattern.direction.as_str())
        .expect("set direction");
    let pivots = PyList::empty_bound(py);
    for p in &d.pattern.pivots {
        pivots.append(pivot_to_dict(py, p)).expect("append pivot");
    }
    out.set_item("pivots", pivots).expect("set pivots");
    let lines = PyList::empty_bound(py);
    for tl in &d.pattern.trend_lines {
        lines
            .append(trendline_to_dict(py, tl))
            .expect("append trendline");
    }
    out.set_item("trend_lines", lines).expect("set trend_lines");
    out.set_item("formation_start", d.pattern.formation.0)
        .expect("set formation_start");
    out.set_item("formation_end", d.pattern.formation.1)
        .expect("set formation_end");
    out.set_item("entry_price", d.pattern.entry_price)
        .expect("set entry_price");
    out.set_item("breakout_price", d.pattern.breakout_price)
        .expect("set breakout_price");
    out.set_item("variant", d.pattern.variant.as_deref())
        .expect("set variant");
    out.set_item("quality", d.score.quality).expect("set quality");
    let features = PyDict::new_bound(py);
    for (k, v) in &d.score.features {
        features.set_item(k, v).expect("set feature");
    }
    out.set_item("features", features).expect("set features");
    out
}

#[pyfunction]
#[pyo3(text_signature = "(highs, lows, ts_ns, orders, /)")]
fn multi_level_pivots<'py>(
    py: Python<'py>,
    highs: PyReadonlyArray1<'py, f64>,
    lows: PyReadonlyArray1<'py, f64>,
    ts_ns: PyReadonlyArray1<'py, i64>,
    orders: Vec<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let h = highs.as_slice()?;
    let l = lows.as_slice()?;
    let t = ts_ns.as_slice()?;
    let pivots = py.allow_threads(|| core_patterns::multi_level_pivots(h, l, t, &orders));
    let out = PyList::empty_bound(py);
    for p in &pivots {
        out.append(pivot_to_dict(py, p)).expect("append pivot");
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(name, ts_ns, open, high, low, close, volume, pivot_orders, min_quality, /)"
)]
fn scan_pattern<'py>(
    py: Python<'py>,
    name: &str,
    ts_ns: PyReadonlyArray1<'py, i64>,
    open: PyReadonlyArray1<'py, f64>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    volume: PyReadonlyArray1<'py, f64>,
    pivot_orders: Vec<usize>,
    min_quality: f64,
) -> PyResult<Bound<'py, PyList>> {
    let ts = ts_ns.as_slice()?;
    let o = open.as_slice()?;
    let h = high.as_slice()?;
    let l = low.as_slice()?;
    let c = close.as_slice()?;
    let v = volume.as_slice()?;
    let view = core_patterns::OhlcvView {
        ts_ns: ts,
        open: o,
        high: h,
        low: l,
        close: c,
        volume: v,
    };
    let detections = py
        .allow_threads(|| core_patterns::scan(name, view, &pivot_orders, min_quality))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let out = PyList::empty_bound(py);
    for d in &detections {
        out.append(detection_to_dict(py, d)).expect("append");
    }
    Ok(out)
}

#[pyfunction]
fn list_pattern_names() -> Vec<&'static str> {
    vec![
        "head_and_shoulders",
        "inverse_head_and_shoulders",
        "double_top",
        "double_bottom",
        "triple_top",
        "triple_bottom",
        "ascending_triangle",
        "descending_triangle",
        "symmetrical_triangle",
    ]
}

// ---------------------------------------------------------------------- module

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kernel_version, m)?)?;
    m.add_function(wrap_pyfunction!(returns_from_prices, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_batch, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown_series, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_batch, m)?)?;
    m.add_function(wrap_pyfunction!(var_batch, m)?)?;
    m.add_function(wrap_pyfunction!(cvar_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sim_run_weights, m)?)?;
    m.add_function(wrap_pyfunction!(sim_run_orders, m)?)?;
    m.add_function(wrap_pyfunction!(sim_run_signals, m)?)?;
    m.add_function(wrap_pyfunction!(scan_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(multi_level_pivots, m)?)?;
    m.add_function(wrap_pyfunction!(list_pattern_names, m)?)?;
    Ok(())
}
