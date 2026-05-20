//! Least-squares trend line fitting through pivot anchors.
//!
//! Pure-Rust port of `pattern_service.detection.trendline.fit_trendline`.
//! The closed-form OLS path matches the reference Python implementation —
//! parity is locked in by the Python integration tests in
//! `tests/integration/test_pattern_pipeline.py`.

use crate::patterns::types::{Pivot, Role, TrendLine};

/// Fit a least-squares line through `pivots`.
///
/// `role` is recorded on the returned [`TrendLine`] for downstream
/// boundary-respect scoring — the fit itself doesn't depend on it.
///
/// Returns `None` when fewer than two pivots are supplied. With exactly
/// two pivots the fit is exact (`r² = 1.0`). When all pivot indices
/// collide, slope is forced to zero and r² collapses to 1.0 only when
/// every price is identical (matches the rank-deficient lstsq fallback in
/// the reference implementation).
pub fn fit_trendline(pivots: &[Pivot], role: Role) -> Option<TrendLine> {
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
        role,
    })
}

/// Per-bar goodness-of-fit of a trend line against intermediate bars
/// over its span.
///
/// **Not used by the quality scorer** — it collapses to 0 for extreme-anchor
/// lines (e.g. triple_bottom support sitting at the trough level, where bars
/// rise far above by design). See `score_trendline` for the dispatch the
/// scorer uses instead. Kept as a tested primitive for callers that want
/// intermediate-bar fit specifically.
pub fn trendline_fit_r2(prices: &[f64], line: &TrendLine) -> f64 {
    let start = line.start_index;
    let last = line.end_index.min(prices.len().saturating_sub(1));
    if last <= start + 1 {
        return 0.0;
    }
    let bars = &prices[start..=last];
    let n_f = bars.len() as f64;

    // SS_tot: variance against the mean.
    let mean = bars.iter().sum::<f64>() / n_f;
    let mut ss_tot = 0.0;
    for p in bars {
        ss_tot += (p - mean).powi(2);
    }

    // SS_res: variance against the trend line.
    let mut ss_res = 0.0;
    for (offset, p) in bars.iter().enumerate() {
        let i = start + offset;
        ss_res += (p - line.price_at(i)).powi(2);
    }

    if ss_tot == 0.0 {
        // Bars are constant. The line is informative only if it predicts
        // that constant; otherwise it disagrees with every bar.
        return if ss_res < 1e-12 { 1.0 } else { 0.0 };
    }
    (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
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
    let n = highs.len().min(lows.len());
    if n == 0 || start_idx >= n {
        return false;
    }
    let last = end_idx.min(n - 1);
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

/// Default fractional tolerance used by [`boundary_respect_ratio`]. The
/// same 0.5% level adopted by the triple boundary gate: anything beyond
/// that is a meaningful breach of the line, not just noise.
pub const DEFAULT_BOUNDARY_RESPECT_TOLERANCE: f64 = 0.005;

/// Fraction of bars in the line's span whose high / low respects the
/// line within `tolerance` of the line price, evaluated per [`Role`]:
/// `Upper` checks highs against the line + tolerance; `Lower` checks lows
/// against the line − tolerance.
///
/// Substituted for anchor R² on 2-anchor lines (double_top / double_bottom,
/// H&S necklines), where anchor R² is trivially 1.0 by construction and
/// gives the scorer nothing to discriminate on. Reading `line.role`
/// directly (instead of `max(upper, lower)`) is what keeps that
/// discrimination from saturating.
///
/// Tolerance is a fraction of the line price, not the bar price — scale-free
/// across price magnitudes. Returns `0.0` if the span has fewer than 2 bars.
pub fn boundary_respect_ratio(
    highs: &[f64],
    lows: &[f64],
    line: &TrendLine,
    tolerance: f64,
) -> f64 {
    let n = highs.len().min(lows.len());
    if n == 0 {
        return 0.0;
    }
    let last = line.end_index.min(n - 1);
    let start = line.start_index;
    if last <= start {
        return 0.0;
    }
    let span = last - start + 1;

    let mut ok = 0usize;
    for i in start..=last {
        let line_price = line.price_at(i);
        let tol_amount = tolerance * line_price.abs();
        let respects = match line.role {
            Role::Upper => highs[i] <= line_price + tol_amount,
            Role::Lower => lows[i] >= line_price - tol_amount,
        };
        if respects {
            ok += 1;
        }
    }
    (ok as f64 / span as f64).clamp(0.0, 1.0)
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
        assert!(fit_trendline(&[], Role::Upper).is_none());
        assert!(fit_trendline(&[pv(0, 1.0)], Role::Upper).is_none());
    }

    #[test]
    fn two_pivots_give_exact_fit() {
        let line = fit_trendline(&[pv(0, 1.0), pv(10, 11.0)], Role::Upper).unwrap();
        assert!((line.slope - 1.0).abs() < 1e-12);
        assert!((line.intercept - 1.0).abs() < 1e-12);
        assert!((line.r_squared - 1.0).abs() < 1e-12);
        assert_eq!(line.touch_count, 2);
        assert_eq!(line.role, Role::Upper);
    }

    #[test]
    fn three_collinear_points_are_perfectly_fit() {
        let line = fit_trendline(&[pv(1, 3.0), pv(4, 9.0), pv(7, 15.0)], Role::Upper).unwrap();
        assert!((line.slope - 2.0).abs() < 1e-12);
        assert!((line.r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn constant_y_has_zero_slope_and_r2_of_one() {
        let line = fit_trendline(&[pv(0, 5.0), pv(5, 5.0), pv(10, 5.0)], Role::Lower).unwrap();
        assert_eq!(line.slope, 0.0);
        assert!((line.intercept - 5.0).abs() < 1e-12);
        assert_eq!(line.r_squared, 1.0);
        assert_eq!(line.role, Role::Lower);
    }

    #[test]
    fn collapsed_x_with_varying_y_gives_zero_r2() {
        // All points share index 5 (degenerate); price differs.
        let line = fit_trendline(&[pv(5, 1.0), pv(5, 2.0), pv(5, 3.0)], Role::Upper).unwrap();
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
            role: Role::Upper,
        };
        let lower = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 2,
            role: Role::Lower,
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
            role: Role::Upper,
        };
        let lower = TrendLine {
            start_index: 0,
            end_index: 4,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 2,
            role: Role::Lower,
        };
        let highs = [9.0, 9.5, 12.0, 8.0, 9.0]; // bar 2 breaks out
        let lows = [1.0; 5];
        assert!(!validate_boundaries(
            &highs, &lows, &upper, &lower, 0, 4, 0.02
        ));
    }

    fn flat_line(start: usize, end: usize, price: f64, role: Role) -> TrendLine {
        TrendLine {
            start_index: start,
            end_index: end,
            slope: 0.0,
            intercept: price,
            r_squared: 1.0,
            touch_count: 2,
            role,
        }
    }

    fn sloped_line(start: usize, end: usize, slope: f64, intercept: f64, role: Role) -> TrendLine {
        TrendLine {
            start_index: start,
            end_index: end,
            slope,
            intercept,
            r_squared: 1.0,
            touch_count: 2,
            role,
        }
    }

    #[test]
    fn fit_r2_is_one_when_prices_lie_exactly_on_line() {
        // Prices follow y = 2 + x exactly across 5 bars.
        let prices: Vec<f64> = (0..5).map(|i| 2.0 + i as f64).collect();
        let line = sloped_line(0, 4, 1.0, 2.0, Role::Upper);
        assert!((trendline_fit_r2(&prices, &line) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fit_r2_is_one_when_prices_constant_and_line_predicts_constant() {
        // Constant prices at 100, flat line at 100 → perfect fit.
        let prices = vec![100.0; 10];
        let line = flat_line(0, 9, 100.0, Role::Upper);
        assert!((trendline_fit_r2(&prices, &line) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fit_r2_is_zero_when_prices_constant_but_line_disagrees() {
        // Constant prices at 100, sloped line wandering away → no fit.
        let prices = vec![100.0; 10];
        let line = sloped_line(0, 9, 1.0, 100.0, Role::Upper); // y = 100 + x
        assert_eq!(trendline_fit_r2(&prices, &line), 0.0);
    }

    #[test]
    fn fit_r2_is_zero_when_line_predicts_constant_but_prices_vary() {
        // Flat line at 100, prices oscillate around 100 → SS_res ≈ SS_tot.
        let prices: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 100.0 } else { 110.0 })
            .collect();
        let line = flat_line(0, 9, 105.0, Role::Upper); // mean of oscillation
        let r2 = trendline_fit_r2(&prices, &line);
        // Flat line at the mean is the null model — R² collapses to 0.
        assert!(r2 < 1e-9, "expected ~0, got {r2}");
    }

    #[test]
    fn fit_r2_decreases_as_prices_deviate_more_from_line() {
        // Hold the line flat at 100; progressively scatter prices.
        // R² must monotonically decrease as scatter increases.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let mut prev = f64::INFINITY;
        for &amplitude in &[0.0, 1.0, 5.0, 15.0] {
            let prices: Vec<f64> = (0..10)
                .map(|i| 100.0 + amplitude * (i as f64).sin())
                .collect();
            let r2 = trendline_fit_r2(&prices, &line);
            assert!(
                r2 <= prev + 1e-9,
                "R² should not increase: amplitude {amplitude} → {r2}, prev {prev}"
            );
            prev = r2;
        }
    }

    #[test]
    fn fit_r2_returns_zero_for_too_few_bars() {
        let prices = vec![100.0, 101.0];
        let line = flat_line(0, 0, 100.0, Role::Upper);
        assert_eq!(trendline_fit_r2(&prices, &line), 0.0);
    }

    #[test]
    fn fit_r2_clamps_end_index_to_prices_length() {
        // Line claims it spans bars 0..=20, but prices only has 5 bars.
        // Should not panic; should evaluate over the available 0..=4.
        let prices: Vec<f64> = (0..5).map(|i| 2.0 + i as f64).collect();
        let line = sloped_line(0, 20, 1.0, 2.0, Role::Upper);
        assert!((trendline_fit_r2(&prices, &line) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fit_r2_distinguishes_two_lines_with_identical_anchor_r_squared() {
        // The whole point of the fix: two TrendLines with the same
        // anchor-only `r_squared = 1.0` should produce different
        // `trendline_fit_r2` values when the bars between anchors
        // differ.
        let line = flat_line(0, 9, 100.0, Role::Upper);

        // Bars hugging the line.
        let clean: Vec<f64> = (0..10).map(|_| 100.0).collect();
        // Bars wandering wildly.
        let noisy: Vec<f64> = (0..10).map(|i| 100.0 + 20.0 * (i as f64).sin()).collect();

        let r2_clean = trendline_fit_r2(&clean, &line);
        let r2_noisy = trendline_fit_r2(&noisy, &line);
        assert!(
            r2_clean > r2_noisy,
            "clean ({r2_clean}) must beat noisy ({r2_noisy})"
        );
    }

    #[test]
    fn boundary_respect_ratio_one_when_all_bars_respect_upper_line() {
        // Flat resistance line at 100; every bar's high sits below.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let highs = vec![99.0; 10];
        let lows = vec![90.0; 10];
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {r}");
    }

    #[test]
    fn boundary_respect_ratio_one_when_all_bars_respect_lower_line() {
        // Flat support line at 100; every bar's low sits above.
        let line = flat_line(0, 9, 100.0, Role::Lower);
        let highs = vec![110.0; 10];
        let lows = vec![101.0; 10];
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {r}");
    }

    #[test]
    fn boundary_respect_ratio_half_when_half_the_bars_pierce_upper_line() {
        // Upper resistance line at 100; alternating highs at 99 (respect)
        // and 110 (breach, well outside 0.5% tolerance). Role::Upper
        // evaluates only the upper-respect ratio, which is 0.5.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let highs: Vec<f64> = (0..10)
            .map(|i| if i % 2 == 0 { 99.0 } else { 110.0 })
            .collect();
        let lows = vec![10.0; 10];
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert!((r - 0.5).abs() < 1e-9, "expected 0.5, got {r}");
    }

    #[test]
    fn boundary_respect_ratio_zero_when_upper_role_breached() {
        // Upper line at 100. Every high breaches above (110 > 100.5),
        // so the upper-respect ratio is 0 — Role::Lower information is
        // ignored even though the lows respect the same level.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let highs = vec![110.0; 10];
        let lows = vec![101.0; 10];
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn boundary_respect_ratio_does_not_fall_back_to_other_role() {
        // Upper line at 100. All highs break above (upper-respect = 0).
        // The lows respect (lower-respect = 1), but Role::Upper must
        // NOT fall back to it — this is the whole point of the role
        // enum. Earlier max-of-two heuristic returned 1.0 here.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let highs = vec![110.0; 10];
        let lows = vec![101.0; 10];
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert_eq!(r, 0.0, "Role::Upper must not fall back to lower role");

        // Same panel, but with Role::Lower — now the metric reads lows.
        let line = flat_line(0, 9, 100.0, Role::Lower);
        let r = boundary_respect_ratio(&highs, &lows, &line, 0.005);
        assert!(
            (r - 1.0).abs() < 1e-9,
            "Role::Lower should score 1, got {r}"
        );
    }

    #[test]
    fn boundary_respect_ratio_respects_tolerance_band() {
        // Upper line at 100, tolerance 1%. A high of 100.5 (0.5% above) is
        // inside tolerance and respects the upper role. A high of
        // 102 (2% above) is outside tolerance.
        let line = flat_line(0, 9, 100.0, Role::Upper);
        let highs = vec![100.5; 10];
        let lows = vec![10.0; 10];
        assert!((boundary_respect_ratio(&highs, &lows, &line, 0.01) - 1.0).abs() < 1e-9);

        let highs = vec![102.0; 10];
        assert_eq!(boundary_respect_ratio(&highs, &lows, &line, 0.01), 0.0);
    }

    #[test]
    fn boundary_respect_ratio_zero_for_single_bar_span() {
        // Line spans a single bar — no usable signal.
        let line = flat_line(5, 5, 100.0, Role::Upper);
        let highs = vec![99.0; 10];
        let lows = vec![101.0; 10];
        assert_eq!(boundary_respect_ratio(&highs, &lows, &line, 0.005), 0.0);
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
            role: Role::Upper,
        };
        // Tolerance 1% of 100 = 1.0
        let prices = [100.0, 99.5, 102.0, 101.0, 100.0];
        // bars 0, 1, 3, 4 within 1% (0, 0.5, 1, 0); bar 2 (2%) is out.
        assert_eq!(count_touches(&prices, &line, 0.01), 4);
    }
}
