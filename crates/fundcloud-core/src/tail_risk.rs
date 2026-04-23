//! Tail-risk estimators — historical VaR and Conditional VaR (Expected Shortfall).

use ndarray::{Array1, ArrayView2};
use rayon::prelude::*;

/// Linear-interpolated empirical quantile (pandas default).
/// Consumes the buffer so we can strip NaNs and sort in place.
fn quantile(mut values: Vec<f64>, q: f64) -> f64 {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return values[lo];
    }
    let frac = pos - lo as f64;
    values[lo] + (values[hi] - values[lo]) * frac
}

/// Historical Value-at-Risk at confidence ``alpha`` (``0 < alpha < 1``).
/// Returns a **loss** as a negative number.
pub fn var_batch(returns: ArrayView2<'_, f64>, alpha: f64) -> Array1<f64> {
    let m = returns.ncols();
    let mut out = Array1::<f64>::zeros(m);
    let vals: Vec<(usize, f64)> = (0..m)
        .into_par_iter()
        .map(|c| {
            let col: Vec<f64> = returns.column(c).to_vec();
            (c, quantile(col, 1.0 - alpha))
        })
        .collect();
    for (c, v) in vals {
        out[c] = v;
    }
    out
}

/// Conditional VaR — mean of returns below the ``1 - alpha`` quantile.
pub fn cvar_batch(returns: ArrayView2<'_, f64>, alpha: f64) -> Array1<f64> {
    let m = returns.ncols();
    let mut out = Array1::<f64>::zeros(m);
    let vals: Vec<(usize, f64)> = (0..m)
        .into_par_iter()
        .map(|c| {
            let col: Vec<f64> = returns
                .column(c)
                .iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect();
            if col.is_empty() {
                return (c, f64::NAN);
            }
            let q = quantile(col.clone(), 1.0 - alpha);
            // Inclusive tail — keep observations ≤ the quantile.
            let tail: Vec<f64> = col.into_iter().filter(|&v| v <= q).collect();
            if tail.is_empty() {
                return (c, f64::NAN);
            }
            let sum: f64 = tail.iter().sum();
            (c, sum / tail.len() as f64)
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
    use ndarray::arr2;

    #[test]
    fn var_less_extreme_than_cvar() {
        let r = arr2(&[
            [-0.10, -0.05],
            [-0.05, -0.02],
            [0.00, 0.01],
            [0.02, 0.03],
            [0.05, 0.06],
            [0.07, 0.08],
            [0.09, 0.10],
        ]);
        let v = var_batch(r.view(), 0.95);
        let c = cvar_batch(r.view(), 0.95);
        // cvar averages the worst; it should be ≤ var.
        for (vi, ci) in v.iter().zip(c.iter()) {
            assert!(*ci <= *vi + 1e-12, "cvar {ci} should be <= var {vi}");
        }
    }

    #[test]
    fn quantile_matches_numpy_on_small_case() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        // numpy.quantile([1,2,3,4,5], 0.5, method="linear") = 3.0
        let q = quantile(x, 0.5);
        assert!((q - 3.0).abs() < 1e-12);
    }
}
