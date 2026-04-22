//! Return calculations.
//!
//! Slice 1 implements the simplest kernel end-to-end (`returns_from_prices`)
//! so the Python/Rust bridge can be exercised in CI. Remaining kernels land in
//! slice 6.

use ndarray::{Array1, ArrayView1};

/// Simple period-over-period returns: `r_t = p_t / p_{t-1} - 1`.
///
/// The first element of the returned array is always `NaN` to preserve length.
/// An empty input yields an empty output.
pub fn returns_from_prices(prices: ArrayView1<'_, f64>) -> Array1<f64> {
    let n = prices.len();
    let mut out = Array1::<f64>::from_elem(n, f64::NAN);
    if n < 2 {
        return out;
    }
    for i in 1..n {
        let prev = prices[i - 1];
        out[i] = prices[i] / prev - 1.0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn empty_input_gives_empty_output() {
        let p = Array1::<f64>::zeros(0);
        let r = returns_from_prices(p.view());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn single_element_is_nan() {
        let p = arr1(&[100.0]);
        let r = returns_from_prices(p.view());
        assert!(r[0].is_nan());
    }

    #[test]
    fn simple_returns_are_correct() {
        let p = arr1(&[100.0, 110.0, 99.0]);
        let r = returns_from_prices(p.view());
        assert!(r[0].is_nan());
        assert!((r[1] - 0.1).abs() < 1e-12);
        assert!((r[2] - (-0.1)).abs() < 1e-12);
    }
}
