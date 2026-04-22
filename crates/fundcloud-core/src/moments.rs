//! Annualised risk-adjusted performance measures, batched by column.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Sample mean of a 1-D slice (NaN-skipping).
fn mean(col: ArrayView1<'_, f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &v in col.iter() {
        if !v.is_nan() {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Sample std (``ddof``=1 by default) of a 1-D slice.
fn sample_std(col: ArrayView1<'_, f64>, ddof: usize) -> f64 {
    let mu = mean(col);
    if mu.is_nan() {
        return f64::NAN;
    }
    let mut var = 0.0;
    let mut count = 0usize;
    for &v in col.iter() {
        if !v.is_nan() {
            var += (v - mu) * (v - mu);
            count += 1;
        }
    }
    if count <= ddof {
        return f64::NAN;
    }
    (var / (count - ddof) as f64).sqrt()
}

/// Downside deviation vs ``target``: `sqrt(mean( min(r - target, 0)^2 ))`.
/// Denominator is the sample count (not `count - ddof`), matching the
/// convention in `fundcloud.metrics.core.sortino`.
fn downside_deviation(col: ArrayView1<'_, f64>, target: f64) -> f64 {
    let mut sq = 0.0;
    let mut count = 0usize;
    for &v in col.iter() {
        if v.is_nan() {
            continue;
        }
        count += 1;
        let d = v - target;
        if d < 0.0 {
            sq += d * d;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sq / count as f64).sqrt()
}

/// Annualised Sharpe ratio per column.
pub fn sharpe_batch(
    returns: ArrayView2<'_, f64>,
    rf_per_period: f64,
    periods_per_year: f64,
) -> Array1<f64> {
    let m = returns.ncols();
    let mut out = Array1::<f64>::zeros(m);
    let sqrt_pp = periods_per_year.sqrt();
    let vals: Vec<(usize, f64)> = (0..m)
        .into_par_iter()
        .map(|c| {
            let col = returns.column(c);
            let mu = mean(col) - rf_per_period;
            let sigma = sample_std(col, 1);
            let val = if !sigma.is_finite() || sigma == 0.0 {
                f64::NAN
            } else {
                (mu / sigma) * sqrt_pp
            };
            (c, val)
        })
        .collect();
    for (c, v) in vals {
        out[c] = v;
    }
    out
}

/// Annualised Sortino ratio per column.
pub fn sortino_batch(
    returns: ArrayView2<'_, f64>,
    target: f64,
    periods_per_year: f64,
) -> Array1<f64> {
    let m = returns.ncols();
    let mut out = Array1::<f64>::zeros(m);
    let sqrt_pp = periods_per_year.sqrt();
    let vals: Vec<(usize, f64)> = (0..m)
        .into_par_iter()
        .map(|c| {
            let col = returns.column(c);
            let mu = mean(col) - target;
            let dd = downside_deviation(col, target);
            let val = if !dd.is_finite() || dd == 0.0 {
                f64::NAN
            } else {
                (mu / dd) * sqrt_pp
            };
            (c, val)
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
    fn sharpe_is_finite_for_normal_input() {
        let r = arr2(&[[0.01, 0.02], [-0.005, 0.01], [0.007, -0.01], [0.002, 0.005]]);
        let s = sharpe_batch(r.view(), 0.0, 252.0);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn constant_series_gives_nan_sharpe() {
        let r = arr2(&[[0.01, 0.0], [0.01, 0.0], [0.01, 0.0]]);
        let s = sharpe_batch(r.view(), 0.0, 252.0);
        // Zero std → NaN, by our convention.
        assert!(s[0].is_nan());
        assert!(s[1].is_nan());
    }

    #[test]
    fn sortino_only_penalises_downside() {
        let r = arr2(&[[0.01, -0.02], [0.02, -0.01], [0.015, -0.015]]);
        let s = sortino_batch(r.view(), 0.0, 252.0);
        assert!(s[0].is_nan()); // upside only — dd = 0 → NaN
        assert!(s[1].is_finite() && s[1] < 0.0);
    }
}
