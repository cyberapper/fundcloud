//! Drawdown and related peak-to-trough analytics.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Drawdown at each timestamp — ``wealth / cummax - 1`` on the returns series.
///
/// The returned array has the same length as the input; the first element is
/// exactly zero (we define `wealth[0] = 1.0 + r[0]`, `peak[0] = wealth[0]`).
pub fn drawdown_series(returns: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = returns.len();
    let mut out = Array1::<f64>::zeros(n);
    if n == 0 {
        return out;
    }
    let mut wealth = 1.0;
    let mut peak = 1.0;
    for i in 0..n {
        let r = returns[i];
        if !r.is_nan() {
            wealth *= 1.0 + r;
        }
        if wealth > peak {
            peak = wealth;
        }
        out[i] = if peak == 0.0 {
            0.0
        } else {
            wealth / peak - 1.0
        };
    }
    out
}

/// Largest peak-to-trough loss per column (returns a negative number).
pub fn max_drawdown_batch(returns: ArrayView2<'_, f64>) -> Array1<f64> {
    let m = returns.ncols();
    let mut out = Array1::<f64>::zeros(m);
    let vals: Vec<(usize, f64)> = (0..m)
        .into_par_iter()
        .map(|c| {
            let dd = drawdown_series(returns.column(c));
            let mn = dd.iter().copied().fold(f64::INFINITY, f64::min);
            (c, if mn.is_finite() { mn } else { 0.0 })
        })
        .collect();
    for (c, v) in vals {
        out[c] = v;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn drawdown_is_nonpositive() {
        let r = arr1(&[0.01, -0.02, 0.005, -0.01, 0.02]);
        let dd = drawdown_series(r.view());
        assert!(dd.iter().all(|v| *v <= 1e-12));
    }

    #[test]
    fn max_drawdown_matches_manual() {
        // wealth = [1.1, 1.21, 0.605, 0.635]  (peak = 1.21, trough = 0.605)
        // max_dd = 0.605 / 1.21 - 1 = -0.5
        let r = arr1(&[0.10, 0.10, -0.50, 0.05]);
        let dd = drawdown_series(r.view());
        let min = dd.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((min + 0.5).abs() < 1e-12);
    }
}
