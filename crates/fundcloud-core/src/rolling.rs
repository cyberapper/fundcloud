//! Rolling-window reductions.
//!
//! Every kernel takes a 1-D input (or a 2-D panel with columns = assets
//! / strategies) and returns an output of the same shape. The first
//! ``window - 1`` elements are ``NaN`` — we don't have enough history yet.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Rolling arithmetic mean over a sliding window.
pub fn rolling_mean(x: ArrayView1<'_, f64>, window: usize) -> Array1<f64> {
    let n = x.len();
    let mut out = Array1::<f64>::from_elem(n, f64::NAN);
    if window == 0 || window > n {
        return out;
    }
    let mut sum = 0.0;
    let mut nan_count = 0usize;
    for i in 0..n {
        let v = x[i];
        if v.is_nan() {
            nan_count += 1;
        } else {
            sum += v;
        }
        if i >= window {
            let dropped = x[i - window];
            if dropped.is_nan() {
                nan_count -= 1;
            } else {
                sum -= dropped;
            }
        }
        if i + 1 >= window && nan_count == 0 {
            out[i] = sum / window as f64;
        }
    }
    out
}

/// Rolling (sample) standard deviation. ``ddof = 1`` matches pandas default.
pub fn rolling_std(x: ArrayView1<'_, f64>, window: usize, ddof: usize) -> Array1<f64> {
    let n = x.len();
    let mut out = Array1::<f64>::from_elem(n, f64::NAN);
    if window <= ddof || window > n {
        return out;
    }
    let denom = (window - ddof) as f64;
    for i in (window - 1)..n {
        let start = i + 1 - window;
        let mut mean = 0.0;
        let mut any_nan = false;
        for k in start..=i {
            if x[k].is_nan() {
                any_nan = true;
                break;
            }
            mean += x[k];
        }
        if any_nan {
            continue;
        }
        mean /= window as f64;
        let mut var = 0.0;
        for k in start..=i {
            let d = x[k] - mean;
            var += d * d;
        }
        out[i] = (var / denom).sqrt();
    }
    out
}

/// Column-wise rolling mean across a 2-D panel.
pub fn rolling_mean_batch(x: ArrayView2<'_, f64>, window: usize) -> Array2<f64> {
    let (n, m) = (x.nrows(), x.ncols());
    let mut out = Array2::<f64>::from_elem((n, m), f64::NAN);
    let parts: Vec<(usize, Array1<f64>)> = (0..m)
        .into_par_iter()
        .map(|c| (c, rolling_mean(x.column(c), window)))
        .collect();
    for (c, col) in parts {
        out.column_mut(c).assign(&col);
    }
    out
}

/// Column-wise rolling std across a 2-D panel.
pub fn rolling_std_batch(x: ArrayView2<'_, f64>, window: usize, ddof: usize) -> Array2<f64> {
    let (n, m) = (x.nrows(), x.ncols());
    let mut out = Array2::<f64>::from_elem((n, m), f64::NAN);
    let parts: Vec<(usize, Array1<f64>)> = (0..m)
        .into_par_iter()
        .map(|c| (c, rolling_std(x.column(c), window, ddof)))
        .collect();
    for (c, col) in parts {
        out.column_mut(c).assign(&col);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Axis};

    #[test]
    fn rolling_mean_matches_hand_calc() {
        let x = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = rolling_mean(x.view(), 3);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!((out[2] - 2.0).abs() < 1e-12);
        assert!((out[3] - 3.0).abs() < 1e-12);
        assert!((out[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_std_matches_population_and_sample_formulas() {
        let x = arr1(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        // Population std (ddof=0) = sqrt(32/8) = 2.0 on this classic case.
        let pop = rolling_std(x.view(), 8, 0);
        assert!((pop[7] - 2.0).abs() < 1e-12);
        // Sample std (ddof=1) = sqrt(32/7).
        let samp = rolling_std(x.view(), 8, 1);
        assert!((samp[7] - (32.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn rolling_mean_batch_matches_per_column() {
        let x = arr2(&[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]);
        let out = rolling_mean_batch(x.view(), 2);
        assert!(out[[0, 0]].is_nan());
        assert!((out[[1, 0]] - 1.5).abs() < 1e-12);
        assert!((out[[3, 1]] - 35.0).abs() < 1e-12);
    }

    #[test]
    fn returns_empty_when_window_too_large() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let out = rolling_mean(x.view(), 10);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn axis_helpers_stay_readable() {
        // Guard against accidental axis drift when refactoring.
        let x = arr2(&[[1.0, 10.0], [2.0, 20.0]]);
        assert_eq!(x.axis_iter(Axis(1)).count(), 2);
    }
}
