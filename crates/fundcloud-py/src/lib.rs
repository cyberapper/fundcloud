//! PyO3 bindings for `fundcloud-core`.
//!
//! Imports from Python as `fundcloud._core`. Surfaces the full kernel
//! suite: rolling reductions, drawdown analytics, risk-adjusted moments,
//! and tail-risk estimators. Every public function releases the GIL via
//! `py.allow_threads` before handing work to the pure-Rust crate.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use fundcloud_core::{
    drawdown as core_drawdown, moments as core_moments, returns as core_returns,
    rolling as core_rolling, sim as core_sim, tail_risk as core_tail,
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
    let out = py.allow_threads(|| core_returns::returns_from_prices(view));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_rolling::rolling_mean(view, window));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_rolling::rolling_std(view, window, ddof));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(text_signature = "(x, window, /)")]
fn rolling_mean_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    window: usize,
) -> Bound<'py, PyArray2<f64>> {
    let view = x.as_array();
    let out = py.allow_threads(|| core_rolling::rolling_mean_batch(view, window));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_rolling::rolling_std_batch(view, window, ddof));
    out.into_pyarray_bound(py)
}

// -------------------------------------------------------------------- drawdown

#[pyfunction]
#[pyo3(text_signature = "(returns, /)")]
fn drawdown_series<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.allow_threads(|| core_drawdown::drawdown_series(view));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(text_signature = "(returns, /)")]
fn max_drawdown_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.allow_threads(|| core_drawdown::max_drawdown_batch(view));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_moments::sharpe_batch(view, rf_per_period, periods_per_year));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_moments::sortino_batch(view, target, periods_per_year));
    out.into_pyarray_bound(py)
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
    let out = py.allow_threads(|| core_tail::var_batch(view, alpha));
    out.into_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(text_signature = "(returns, alpha, /)")]
fn cvar_batch<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    alpha: f64,
) -> Bound<'py, PyArray1<f64>> {
    let view = returns.as_array();
    let out = py.allow_threads(|| core_tail::cvar_batch(view, alpha));
    out.into_pyarray_bound(py)
}

// ---------------------------------------------------------------------- sim

fn sim_output_to_dict<'py>(
    py: Python<'py>,
    out: core_sim::SimOutput,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("equity", out.equity.into_pyarray_bound(py))?;
    // weights_history = list of (bar_idx, dict[asset_idx->weight])
    let wh = PyList::empty_bound(py);
    for (bar, pairs) in &out.weights_history {
        let inner = PyDict::new_bound(py);
        for (aj, w) in pairs {
            inner.set_item(*aj, *w)?;
        }
        wh.append(PyTuple::new_bound(py, [bar.into_py(py), inner.into_py(py)]))?;
    }
    dict.set_item("weights_history", wh)?;
    dict.set_item("trade_bar", out.trade_bar)?;
    dict.set_item("trade_asset", out.trade_asset)?;
    dict.set_item("trade_qty", out.trade_qty)?;
    dict.set_item("trade_price", out.trade_price)?;
    dict.set_item("trade_fee", out.trade_fee)?;
    dict.set_item("trade_slip_bps", out.trade_slip_bps)?;
    dict.set_item("order_bar", out.order_bar)?;
    dict.set_item("order_asset", out.order_asset)?;
    dict.set_item("order_side", out.order_side)?;
    dict.set_item("order_qty", out.order_qty)?;
    dict.set_item("order_notional", out.order_notional)?;
    dict.set_item("order_kind", out.order_kind)?;
    dict.set_item("order_limit_price", out.order_limit_price)?;
    dict.set_item("order_filled", out.order_filled)?;
    Ok(dict)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, target_weights, target_bar_indices, cash, \
                      cost_kind, cost_p1, cost_p2, slip_kind, slip_p1, exec_kind, tolerance, /)"
)]
fn sim_run_weights<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
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
) -> PyResult<Bound<'py, PyDict>> {
    let open_v = open_panel.as_array();
    let close_v = close_panel.as_array();
    let tw_v = target_weights.as_array();
    let tbi_v = target_bar_indices.as_array();
    let cfg = core_sim::SimCfg {
        cash,
        cost_kind,
        cost_param1: cost_p1,
        cost_param2: cost_p2,
        slip_kind,
        slip_param1: slip_p1,
        exec_kind,
    };
    let out = py.allow_threads(|| {
        core_sim::run_weights(open_v, close_v, tw_v, tbi_v, cfg, tolerance)
    });
    sim_output_to_dict(py, out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, order_bar, order_asset, order_side, \
                      order_qty, order_notional, order_kind, order_limit_price, \
                      cash, cost_kind, cost_p1, cost_p2, slip_kind, slip_p1, exec_kind, /)"
)]
fn sim_run_orders<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
    order_bar: PyReadonlyArray1<'py, i64>,
    order_asset: PyReadonlyArray1<'py, i64>,
    order_side: PyReadonlyArray1<'py, i64>,
    order_qty: PyReadonlyArray1<'py, f64>,
    order_notional: PyReadonlyArray1<'py, f64>,
    order_kind: PyReadonlyArray1<'py, i64>,
    order_limit_price: PyReadonlyArray1<'py, f64>,
    cash: f64,
    cost_kind: u8,
    cost_p1: f64,
    cost_p2: f64,
    slip_kind: u8,
    slip_p1: f64,
    exec_kind: u8,
) -> PyResult<Bound<'py, PyDict>> {
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
    let ob = order_bar.as_array();
    let oa = order_asset.as_array();
    let os = order_side.as_array();
    let oq = order_qty.as_array();
    let on = order_notional.as_array();
    let ok_ = order_kind.as_array();
    let olp = order_limit_price.as_array();
    let out = py.allow_threads(|| {
        core_sim::run_orders(op, cl, ob, oa, os, oq, on, ok_, olp, cfg)
    });
    sim_output_to_dict(py, out)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(
    text_signature = "(open_panel, close_panel, entries, exits, size, \
                      cash, cost_kind, cost_p1, cost_p2, slip_kind, slip_p1, exec_kind, /)"
)]
fn sim_run_signals<'py>(
    py: Python<'py>,
    open_panel: PyReadonlyArray2<'py, f64>,
    close_panel: PyReadonlyArray2<'py, f64>,
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
) -> PyResult<Bound<'py, PyDict>> {
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
    let en = entries.as_array();
    let ex = exits.as_array();
    let out = py.allow_threads(|| core_sim::run_signals(op, cl, en, ex, size, cfg));
    sim_output_to_dict(py, out)
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
    Ok(())
}
