//! Least-squares trend line fitting through pivot anchors.
//!
//! Pure-Rust port of `pattern_service.detection.trendline.fit_trendline`.
//! The closed-form OLS path matches the reference Python implementation —
//! parity is locked in by the Python integration tests in
//! `tests/integration/test_pattern_pipeline.py`.

use crate::patterns::types::{Pivot, TrendLine};

/// Fit a least-squares line through `pivots`.
///
/// Returns `None` when fewer than two pivots are supplied. With exactly
/// two pivots the fit is exact (`r² = 1.0`). When all pivot indices
/// collide, slope is forced to zero and r² collapses to 1.0 only when
/// every price is identical (matches the rank-deficient lstsq fallback in
/// the reference implementation).
pub fn fit_trendline(pivots: &[Pivot]) -> Option<TrendLine> {
    let n = pivots.len();
    if n < 2 {
        return None;
    }

    let n_f = n as f64;
    let mut x_mean = 0.0;
    let mut y_mean = 0.0;
    for p in pivots {
        x_mean += p.index as f64;
        y_mean += p.price;
    }
    x_mean /= n_f;
    y_mean /= n_f;

    let mut s_xx = 0.0;
    let mut s_yy = 0.0;
    let mut s_xy = 0.0;
    for p in pivots {
        let xc = p.index as f64 - x_mean;
        let yc = p.price - y_mean;
        s_xx += xc * xc;
        s_yy += yc * yc;
        s_xy += xc * yc;
    }

    let (slope, intercept, r_squared) = if s_xx == 0.0 {
        // Rank-deficient: lstsq returns slope = 0, intercept = y_mean.
        // r² is 1.0 when y is constant, else 0.0 (model no better than mean).
        let r2 = if s_yy == 0.0 { 1.0 } else { 0.0 };
        (0.0, y_mean, r2)
    } else {
        let slope = s_xy / s_xx;
        let intercept = y_mean - slope * x_mean;
        let r2 = if s_yy == 0.0 {
            1.0
        } else {
            1.0 - (s_yy - slope * s_xy) / s_yy
        };
        (slope, intercept, r2)
    };

    Some(TrendLine {
        start_index: pivots[0].index,
        end_index: pivots[n - 1].index,
        slope,
        intercept,
        r_squared: r_squared.clamp(0.0, 1.0),
        touch_count: n.min(u8::MAX as usize) as u8,
    })
}

/// Validate that all bars between `start_idx..=end_idx` stay between an
/// `upper` and `lower` trend line, within `tolerance` of the channel
/// width. Returns `false` on any breach.
///
/// Mirrors `validate_boundaries` in the reference Python — the key
/// false-positive filter for triangle / channel / rectangle detectors.
pub fn validate_boundaries(
    highs: &[f64],
    lows: &[f64],
    upper: &TrendLine,
    lower: &TrendLine,
    start_idx: usize,
    end_idx: usize,
    tolerance: f64,
) -> bool {
    let last = end_idx.min(highs.len().saturating_sub(1));
    for i in start_idx..=last {
        let upper_price = upper.price_at(i);
        let lower_price = lower.price_at(i);
        let range = upper_price - lower_price;
        if range <= 0.0 {
            return false;
        }
        let tol_amount = range * tolerance;
        if highs[i] > upper_price + tol_amount {
            return false;
        }
        if lows[i] < lower_price - tol_amount {
            return false;
        }
    }
    true
}

/// Count how many bars in `[line.start_index, line.end_index]` come within
/// `tolerance` of the line price. Mirrors `count_touches` in the
/// reference Python.
pub fn count_touches(prices: &[f64], line: &TrendLine, tolerance: f64) -> u32 {
    let last = line.end_index.min(prices.len().saturating_sub(1));
    let mut touches = 0u32;
    for (i, price) in prices
        .iter()
        .enumerate()
        .take(last + 1)
        .skip(line.start_index)
    {
        let line_price = line.price_at(i);
        if line_price == 0.0 {
            continue;
        }
        if (price - line_price).abs() / line_price.abs() <= tolerance {
            touches += 1;
        }
    }
    touches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::types::PivotKind;

    fn pv(index: usize, price: f64) -> Pivot {
        Pivot {
            index,
            ts_ns: 0,
            price,
            kind: PivotKind::High,
            order: 5,
        }
    }

    #[test]
    fn fewer_than_two_pivots_returns_none() {
        assert!(fit_trendline(&[]).is_none());
        assert!(fit_trendline(&[pv(0, 1.0)]).is_none());
    }

    #[test]
    fn two_pivots_give_exact_fit() {
        let line = fit_trendline(&[pv(0, 1.0), pv(10, 11.0)]).unwrap();
        assert!((line.slope - 1.0).abs() < 1e-12);
        assert!((line.intercept - 1.0).abs() < 1e-12);
        assert!((line.r_squared - 1.0).abs() < 1e-12);
        assert_eq!(line.touch_count, 2);
    }

    #[test]
    fn three_collinear_points_are_perfectly_fit() {
        let line = fit_trendline(&[pv(1, 3.0), pv(4, 9.0), pv(7, 15.0)]).unwrap();
        assert!((line.slope - 2.0).abs() < 1e-12);
        assert!((line.r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_y_has_zero_slope_and_r2_of_one() {
        let line = fit_trendline(&[pv(0, 5.0), pv(5, 5.0), pv(10, 5.0)]).unwrap();
        assert_eq!(line.slope, 0.0);
        assert!((line.intercept - 5.0).abs() < 1e-12);
        assert_eq!(line.r_squared, 1.0);
    }

    #[test]
    fn collapsed_x_with_varying_y_gives_zero_r2() {
        // All points share index 5 (degenerate); price differs.
        let line = fit_trendline(&[pv(5, 1.0), pv(5, 2.0), pv(5, 3.0)]).unwrap();
        assert_eq!(line.slope, 0.0);
        assert!((line.intercept - 2.0).abs() < 1e-12);
        assert_eq!(line.r_squared, 0.0);
    }

    #[test]
    fn validate_boundaries_passes_when_bars_inside_channel() {
        // Upper line: y = 10 (flat); lower line: y = 0 (flat).
        let upper = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 10.0,
            r_squared: 1.0,
            touch_count: 2,
        };
        let lower = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 2,
        };
        let highs = [9.0, 9.5, 9.0, 8.0, 9.0];
        let lows = [1.0, 0.5, 1.0, 1.5, 1.0];
        assert!(validate_boundaries(
            &highs, &lows, &upper, &lower, 0, 4, 0.02
        ));
    }

    #[test]
    fn validate_boundaries_fails_when_high_exceeds_upper_line() {
        let upper = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 10.0,
            r_squared: 1.0,
            touch_count: 2,
        };
        let lower = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 2,
        };
        let highs = [9.0, 9.5, 12.0, 8.0, 9.0]; // bar 2 breaks out
        let lows = [1.0; 5];
        assert!(!validate_boundaries(
            &highs, &lows, &upper, &lower, 0, 4, 0.02
        ));
    }

    #[test]
    fn count_touches_counts_only_bars_within_tolerance() {
        let line = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 100.0,
            r_squared: 1.0,
            touch_count: 2,
        };
        // Tolerance 1% of 100 = 1.0
        let prices = [100.0, 99.5, 102.0, 101.0, 100.0];
        // bars 0, 1, 3, 4 within 1% (0, 0.5, 1, 0); bar 2 (2%) is out.
        assert_eq!(count_touches(&prices, &line, 0.01), 4);
    }
}
