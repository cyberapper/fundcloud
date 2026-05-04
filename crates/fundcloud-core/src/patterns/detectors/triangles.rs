//! Triangle detectors: ascending, descending, symmetrical.
//!
//! Port of `pattern_service.detection.patterns.triangle` and
//! `symmetrical_triangle`. All three detectors share the same shape:
//!
//! 1. Pick a sliding window of high-pivots (asc/sym) or low-pivots (desc).
//! 2. Fit an upper and lower trend line through the pivots in that window.
//! 3. Reject if the slopes don't have the right signs.
//! 4. Reject if the lines diverge or collapse.
//! 5. Validate every bar inside the formation stays within the channel
//!    (asc/desc use a fraction-of-channel-width tolerance via
//!    `trendline::validate_boundaries`; symmetric uses an absolute-price
//!    tolerance because the channel collapses to zero near the apex).
//! 6. Deduplicate overlapping detections, keeping the one with more pivots.

use crate::patterns::detect::{prior_trend_slope, PatternDetector};
use crate::patterns::trendline::{fit_trendline, validate_boundaries};
use crate::patterns::types::{Direction, OhlcvView, Pattern, Pivot, PivotKind, TrendLine};

/// Default normalised-slope tolerance for the "flat" leg of asc/desc.
const DEFAULT_FLAT_THRESHOLD: f64 = 0.0005;
/// Asymmetric multiplier — in the "wrong direction" only 70% of the
/// flat_threshold is allowed (rising-by-2.3% lows on a descending
/// triangle, etc.).
const WRONG_DIR_FRACTION: f64 = 0.7;
/// Default minimum touches per trend line.
const DEFAULT_MIN_TOUCHES: usize = 2;
/// Minimum bar count for asc/desc triangles.
const ASC_DESC_MIN_BAR_COUNT: usize = 8;
/// Minimum bar count for symmetrical triangles.
const SYM_MIN_BAR_COUNT: usize = 10;
/// Channel-width tolerance used by asc/desc when calling
/// `validate_boundaries` (2% of channel width).
const CHANNEL_TOLERANCE: f64 = 0.02;
/// Absolute-price tolerance fraction for symmetrical-triangle boundary
/// validation (5% of the starting gap).
const SYM_ABS_TOLERANCE_FRACTION: f64 = 0.05;
/// Symmetrical-triangle minimum normalised slope magnitude (each leg must
/// move at least this fast to count as "converging").
const SYM_MIN_SLOPE_THRESHOLD: f64 = 0.0005;
/// Symmetrical-triangle prior-trend look-back window.
const SYM_PRIOR_WINDOW: usize = 10;

/// Slope normalised by the average price level — makes thresholds
/// independent of an asset's absolute price scale. Mirrors
/// `pattern_service.utils.normalized_slope`.
fn normalized_slope(slope: f64, prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    let avg = prices.iter().sum::<f64>() / (prices.len() as f64);
    if avg == 0.0 {
        0.0
    } else {
        slope / avg
    }
}

fn pivot_prices(pivots: &[Pivot]) -> Vec<f64> {
    pivots.iter().map(|p| p.price).collect()
}

/// Drop overlapping detections, keeping the ones with more pivots first.
/// Mirrors `_deduplicate_patterns` / `_dedup` in the reference Python.
fn deduplicate(mut patterns: Vec<Pattern>) -> Vec<Pattern> {
    if patterns.len() <= 1 {
        return patterns;
    }
    patterns.sort_by(|a, b| b.pivots.len().cmp(&a.pivots.len()));
    let mut kept: Vec<Pattern> = Vec::with_capacity(patterns.len());
    for pat in patterns {
        let (a_start, a_end) = pat.formation;
        let len = a_end.saturating_sub(a_start);
        let overlaps = kept.iter().any(|existing| {
            let (b_start, b_end) = existing.formation;
            let overlap_start = a_start.max(b_start);
            let overlap_end = a_end.min(b_end);
            if overlap_end <= overlap_start || len == 0 {
                return false;
            }
            let overlap_len = overlap_end - overlap_start;
            (overlap_len as f64) / (len as f64) > 0.5
        });
        if !overlaps {
            kept.push(pat);
        }
    }
    kept
}

/// Detect ascending triangles (flat resistance + rising support).
#[derive(Debug, Clone)]
pub struct AscendingTriangleDetector {
    /// Maximum allowed normalised slope magnitude for the "flat" leg.
    pub flat_threshold: f64,
    /// Minimum touches per trend line.
    pub min_touches: usize,
}

impl Default for AscendingTriangleDetector {
    fn default() -> Self {
        Self {
            flat_threshold: DEFAULT_FLAT_THRESHOLD,
            min_touches: DEFAULT_MIN_TOUCHES,
        }
    }
}

impl PatternDetector for AscendingTriangleDetector {
    fn name(&self) -> &'static str {
        "ascending_triangle"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let high_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::High)
            .collect();
        let low_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::Low)
            .collect();
        if high_pivots.len() < self.min_touches || low_pivots.len() < self.min_touches {
            return Vec::new();
        }

        let mut patterns = Vec::new();
        for start_h in 0..high_pivots.len().saturating_sub(1) {
            for end_h in (start_h + 1)..high_pivots.len() {
                let h_subset = &high_pivots[start_h..=end_h];
                if h_subset.len() < self.min_touches {
                    continue;
                }
                let Some(upper) = fit_trendline(h_subset) else {
                    continue;
                };

                // Asymmetric flat tolerance: full threshold for upward
                // drift (consistent with bullish bias), 70% for downward.
                let h_prices = pivot_prices(h_subset);
                let upper_norm = normalized_slope(upper.slope, &h_prices);
                let wrong_dir_bound = -self.flat_threshold * WRONG_DIR_FRACTION;
                if !(wrong_dir_bound..=self.flat_threshold).contains(&upper_norm) {
                    continue;
                }

                let range_start = h_subset[0].index;
                let range_end = h_subset[h_subset.len() - 1].index;

                let l_subset: Vec<Pivot> = low_pivots
                    .iter()
                    .copied()
                    .filter(|p| (range_start..=range_end).contains(&p.index))
                    .collect();
                if l_subset.len() < self.min_touches {
                    continue;
                }
                let Some(lower) = fit_trendline(&l_subset) else {
                    continue;
                };

                let l_prices = pivot_prices(&l_subset);
                let lower_norm = normalized_slope(lower.slope, &l_prices);
                if lower_norm <= 0.0 {
                    continue;
                }

                if !lines_converge(&upper, &lower, range_start, range_end) {
                    continue;
                }

                if range_end.saturating_sub(range_start) < ASC_DESC_MIN_BAR_COUNT {
                    continue;
                }

                if !validate_boundaries(
                    ohlcv.high,
                    ohlcv.low,
                    &upper,
                    &lower,
                    range_start,
                    range_end,
                    CHANNEL_TOLERANCE,
                ) {
                    continue;
                }

                let mut all_pivots: Vec<Pivot> =
                    h_subset.iter().chain(l_subset.iter()).copied().collect();
                all_pivots.sort_by_key(|p| p.index);

                let entry_price = upper.price_at(range_end);

                patterns.push(Pattern {
                    name: "ascending_triangle",
                    direction: Direction::Bullish,
                    pivots: all_pivots,
                    trend_lines: vec![upper, lower],
                    formation: (range_start, range_end),
                    entry_price: Some(entry_price),
                    breakout_price: Some(entry_price),
                    variant: None,
                });
            }
        }
        deduplicate(patterns)
    }
}

/// Detect descending triangles (flat support + falling resistance).
#[derive(Debug, Clone)]
pub struct DescendingTriangleDetector {
    /// Maximum allowed normalised slope magnitude for the "flat" leg.
    pub flat_threshold: f64,
    /// Minimum touches per trend line.
    pub min_touches: usize,
}

impl Default for DescendingTriangleDetector {
    fn default() -> Self {
        Self {
            flat_threshold: DEFAULT_FLAT_THRESHOLD,
            min_touches: DEFAULT_MIN_TOUCHES,
        }
    }
}

impl PatternDetector for DescendingTriangleDetector {
    fn name(&self) -> &'static str {
        "descending_triangle"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let high_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::High)
            .collect();
        let low_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::Low)
            .collect();
        if high_pivots.len() < self.min_touches || low_pivots.len() < self.min_touches {
            return Vec::new();
        }

        let mut patterns = Vec::new();
        for start_l in 0..low_pivots.len().saturating_sub(1) {
            for end_l in (start_l + 1)..low_pivots.len() {
                let l_subset = &low_pivots[start_l..=end_l];
                if l_subset.len() < self.min_touches {
                    continue;
                }
                let Some(lower) = fit_trendline(l_subset) else {
                    continue;
                };

                let l_prices = pivot_prices(l_subset);
                let lower_norm = normalized_slope(lower.slope, &l_prices);
                let wrong_dir_bound = self.flat_threshold * WRONG_DIR_FRACTION;
                if !(-self.flat_threshold..=wrong_dir_bound).contains(&lower_norm) {
                    continue;
                }

                let range_start = l_subset[0].index;
                let range_end = l_subset[l_subset.len() - 1].index;

                let h_subset: Vec<Pivot> = high_pivots
                    .iter()
                    .copied()
                    .filter(|p| (range_start..=range_end).contains(&p.index))
                    .collect();
                if h_subset.len() < self.min_touches {
                    continue;
                }
                let Some(upper) = fit_trendline(&h_subset) else {
                    continue;
                };

                let h_prices = pivot_prices(&h_subset);
                let upper_norm = normalized_slope(upper.slope, &h_prices);
                if upper_norm >= 0.0 {
                    continue;
                }

                if !lines_converge(&upper, &lower, range_start, range_end) {
                    continue;
                }

                if range_end.saturating_sub(range_start) < ASC_DESC_MIN_BAR_COUNT {
                    continue;
                }

                if !validate_boundaries(
                    ohlcv.high,
                    ohlcv.low,
                    &upper,
                    &lower,
                    range_start,
                    range_end,
                    CHANNEL_TOLERANCE,
                ) {
                    continue;
                }

                let mut all_pivots: Vec<Pivot> =
                    h_subset.iter().chain(l_subset.iter()).copied().collect();
                all_pivots.sort_by_key(|p| p.index);

                let entry_price = lower.price_at(range_end);

                patterns.push(Pattern {
                    name: "descending_triangle",
                    direction: Direction::Bearish,
                    pivots: all_pivots,
                    trend_lines: vec![upper, lower],
                    formation: (range_start, range_end),
                    entry_price: Some(entry_price),
                    breakout_price: Some(entry_price),
                    variant: None,
                });
            }
        }
        deduplicate(patterns)
    }
}

/// Detect symmetrical triangles (falling resistance + rising support).
///
/// Direction is inferred from the prior trend: `Bullish` after an
/// uptrend, `Bearish` after a downtrend (defaulting to `Bullish` when
/// the slope is exactly zero — preserves the reference Python's
/// fallback for insufficient history).
#[derive(Debug, Clone)]
pub struct SymmetricalTriangleDetector {
    /// Each leg must have a normalised slope magnitude greater than this
    /// to count as "converging" (defaults to 0.0005).
    pub min_slope_threshold: f64,
    /// Minimum touches per trend line.
    pub min_touches: usize,
    /// Minimum bar count between formation start and end.
    pub min_bar_count: usize,
    /// Look-back window for the prior-trend slope.
    pub prior_window: usize,
}

impl Default for SymmetricalTriangleDetector {
    fn default() -> Self {
        Self {
            min_slope_threshold: SYM_MIN_SLOPE_THRESHOLD,
            min_touches: DEFAULT_MIN_TOUCHES,
            min_bar_count: SYM_MIN_BAR_COUNT,
            prior_window: SYM_PRIOR_WINDOW,
        }
    }
}

impl PatternDetector for SymmetricalTriangleDetector {
    fn name(&self) -> &'static str {
        "symmetrical_triangle"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let high_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::High)
            .collect();
        let low_pivots: Vec<Pivot> = pivots
            .iter()
            .copied()
            .filter(|p| p.kind == PivotKind::Low)
            .collect();
        if high_pivots.len() < self.min_touches || low_pivots.len() < self.min_touches {
            return Vec::new();
        }

        let mut patterns = Vec::new();
        for start_h in 0..high_pivots.len().saturating_sub(1) {
            for end_h in (start_h + 1)..high_pivots.len() {
                let h_subset = &high_pivots[start_h..=end_h];
                if h_subset.len() < self.min_touches {
                    continue;
                }
                let Some(upper) = fit_trendline(h_subset) else {
                    continue;
                };

                let h_prices = pivot_prices(h_subset);
                let upper_norm = normalized_slope(upper.slope, &h_prices);
                // Upper must slope DOWN by at least the threshold.
                if upper_norm >= -self.min_slope_threshold {
                    continue;
                }

                let range_start = h_subset[0].index;
                let range_end = h_subset[h_subset.len() - 1].index;

                let l_subset: Vec<Pivot> = low_pivots
                    .iter()
                    .copied()
                    .filter(|p| (range_start..=range_end).contains(&p.index))
                    .collect();
                if l_subset.len() < self.min_touches {
                    continue;
                }
                let Some(lower) = fit_trendline(&l_subset) else {
                    continue;
                };

                let l_prices = pivot_prices(&l_subset);
                let lower_norm = normalized_slope(lower.slope, &l_prices);
                // Lower must slope UP by at least the threshold.
                if lower_norm <= self.min_slope_threshold {
                    continue;
                }

                if !lines_converge(&upper, &lower, range_start, range_end) {
                    continue;
                }

                if range_end.saturating_sub(range_start) < self.min_bar_count {
                    continue;
                }

                let start_gap = upper.price_at(range_start) - lower.price_at(range_start);
                // Symmetric triangles converge by definition, so a
                // fraction-of-gap tolerance becomes tiny near the apex.
                // Use an absolute-price tolerance based on the starting gap.
                let tol_price = start_gap.max(1e-6) * SYM_ABS_TOLERANCE_FRACTION;
                if !validate_boundaries_abs(
                    ohlcv.high,
                    ohlcv.low,
                    &upper,
                    &lower,
                    range_start,
                    range_end,
                    tol_price,
                ) {
                    continue;
                }

                let direction =
                    direction_from_prior_trend(ohlcv.close, range_start, self.prior_window);
                let upper_end = upper.price_at(range_end);
                let lower_end = lower.price_at(range_end);
                let (entry, breakout) = match direction {
                    Direction::Bearish => (lower_end, lower_end),
                    _ => (upper_end, upper_end),
                };

                let mut all_pivots: Vec<Pivot> =
                    h_subset.iter().chain(l_subset.iter()).copied().collect();
                all_pivots.sort_by_key(|p| p.index);

                patterns.push(Pattern {
                    name: "symmetrical_triangle",
                    direction,
                    pivots: all_pivots,
                    trend_lines: vec![upper, lower],
                    formation: (range_start, range_end),
                    entry_price: Some(entry),
                    breakout_price: Some(breakout),
                    variant: None,
                });
            }
        }
        deduplicate(patterns)
    }
}

/// Lines must converge (end gap strictly less than start gap) and both
/// gaps must be positive (upper above lower at both ends).
fn lines_converge(upper: &TrendLine, lower: &TrendLine, start: usize, end: usize) -> bool {
    let start_gap = upper.price_at(start) - lower.price_at(start);
    let end_gap = upper.price_at(end) - lower.price_at(end);
    start_gap > 0.0 && end_gap > 0.0 && end_gap < start_gap
}

/// Absolute-price-tolerance variant of `validate_boundaries`. Used by the
/// symmetrical-triangle detector because the channel collapses to zero
/// near the apex, so a fraction-of-channel-width tolerance vanishes.
fn validate_boundaries_abs(
    highs: &[f64],
    lows: &[f64],
    upper: &TrendLine,
    lower: &TrendLine,
    start: usize,
    end: usize,
    tol_price: f64,
) -> bool {
    let last = end.min(highs.len().saturating_sub(1));
    for (i, (h, l)) in highs
        .iter()
        .zip(lows.iter())
        .enumerate()
        .take(last + 1)
        .skip(start)
    {
        let up = upper.price_at(i);
        let lo = lower.price_at(i);
        if up <= lo {
            return false;
        }
        if *h > up + tol_price {
            return false;
        }
        if *l < lo - tol_price {
            return false;
        }
    }
    true
}

/// Map prior-trend slope to a triangle direction. Defaults to `Bullish`
/// when the slope is exactly zero (mirrors the reference Python).
fn direction_from_prior_trend(closes: &[f64], formation_start: usize, window: usize) -> Direction {
    let slope = prior_trend_slope(closes, formation_start, window);
    if slope > 0.0 {
        Direction::Bullish
    } else if slope < 0.0 {
        Direction::Bearish
    } else {
        Direction::Bullish
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pivot(index: usize, price: f64, kind: PivotKind) -> Pivot {
        Pivot {
            index,
            ts_ns: index as i64 * 60 * 1_000_000_000,
            price,
            kind,
            order: 5,
        }
    }

    /// Synthetic OHLCV panel for triangle tests.
    struct Panel {
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    }

    fn view<'a>(p: &'a Panel, ts: &'a [i64]) -> OhlcvView<'a> {
        OhlcvView {
            ts_ns: ts,
            open: &p.open,
            high: &p.high,
            low: &p.low,
            close: &p.close,
            volume: &p.volume,
        }
    }

    /// Build a panel where every bar's high sits on the upper trend line
    /// and every bar's low sits on the lower trend line, computed
    /// piecewise-linearly between the supplied (index, upper, lower)
    /// anchors. Outside the formation, prices stay at the boundary values.
    fn channel_panel(n: usize, anchors: &[(usize, f64, f64)], prior_close: f64) -> Panel {
        let mut high = vec![0.0; n];
        let mut low = vec![0.0; n];
        for i in 0..n {
            // Find surrounding anchor pair, clamping at the ends.
            let mut window = anchors.windows(2).find(|w| {
                let (a, _, _) = w[0];
                let (b, _, _) = w[1];
                i >= a && i <= b
            });
            if window.is_none() {
                if i < anchors[0].0 {
                    window = Some(&anchors[..2]);
                } else {
                    window = Some(&anchors[anchors.len() - 2..]);
                }
            }
            let w = window.unwrap();
            let (a_i, a_up, a_lo) = w[0];
            let (b_i, b_up, b_lo) = w[1];
            let span = (b_i - a_i) as f64;
            let t = if span == 0.0 {
                0.0
            } else {
                (i as f64 - a_i as f64) / span
            };
            high[i] = a_up + t * (b_up - a_up);
            low[i] = a_lo + t * (b_lo - a_lo);
        }
        // Closes / opens halfway between high and low; pre-formation close
        // forced to a single level so prior_trend_slope is meaningful.
        let mut close: Vec<f64> = high
            .iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();
        if anchors[0].0 > 0 {
            for c in close.iter_mut().take(anchors[0].0) {
                *c = prior_close;
            }
        }
        let open = close.clone();
        let volume = vec![1000.0; n];
        Panel {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn detects_canonical_ascending_triangle() {
        // Flat resistance at 110, rising support 90 → 105 over bars 5..=20.
        let anchors = [(5, 110.0, 90.0), (20, 110.0, 105.0)];
        let p = channel_panel(30, &anchors, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(5, 110.0, PivotKind::High),
            pivot(8, 93.0, PivotKind::Low),
            pivot(13, 110.0, PivotKind::High),
            pivot(17, 102.0, PivotKind::Low),
            pivot(20, 110.0, PivotKind::High),
        ];
        let raw = AscendingTriangleDetector::default().detect(&pivots, v);
        assert!(
            !raw.is_empty(),
            "expected at least one ascending-triangle detection"
        );
        let det = &raw[0];
        assert_eq!(det.name, "ascending_triangle");
        assert_eq!(det.direction, Direction::Bullish);
        assert_eq!(det.trend_lines.len(), 2);
    }

    #[test]
    fn rejects_when_resistance_drops_too_steeply() {
        // Resistance falling from 110 → 100 (norm slope ≈ -1%/bar / 105 ≈ huge).
        let anchors = [(5, 110.0, 90.0), (20, 100.0, 105.0)];
        let p = channel_panel(30, &anchors, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(5, 110.0, PivotKind::High),
            pivot(8, 93.0, PivotKind::Low),
            pivot(13, 105.0, PivotKind::High),
            pivot(17, 102.0, PivotKind::Low),
            pivot(20, 100.0, PivotKind::High),
        ];
        let raw = AscendingTriangleDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }

    #[test]
    fn detects_canonical_descending_triangle() {
        // Falling resistance 102 → 92, flat support at 90 over bars 5..=22.
        // Need ≥2 lows defining the window AND ≥2 highs inside it, so use
        // three lows (bars 8, 16, 22) sandwiching two highs (12, 20).
        let anchors = [(5, 102.0, 90.0), (22, 92.0, 90.0)];
        let p = channel_panel(30, &anchors, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        // High prices match the linear channel exactly so the fitted line
        // sits on the bar highs and boundary validation passes.
        let pivots = vec![
            pivot(8, 90.0, PivotKind::Low),
            pivot(
                12,
                102.0 + (12.0 - 5.0) / 17.0 * (92.0 - 102.0),
                PivotKind::High,
            ),
            pivot(16, 90.0, PivotKind::Low),
            pivot(
                20,
                102.0 + (20.0 - 5.0) / 17.0 * (92.0 - 102.0),
                PivotKind::High,
            ),
            pivot(22, 90.0, PivotKind::Low),
        ];
        let raw = DescendingTriangleDetector::default().detect(&pivots, v);
        assert!(
            !raw.is_empty(),
            "expected at least one descending-triangle detection"
        );
        let det = &raw[0];
        assert_eq!(det.name, "descending_triangle");
        assert_eq!(det.direction, Direction::Bearish);
    }

    #[test]
    fn detects_canonical_symmetrical_triangle_after_uptrend() {
        // Falling resistance 112 → 102, rising support 88 → 98 over bars 5..=22.
        let anchors = [(5, 112.0, 88.0), (22, 102.0, 98.0)];
        let mut p = channel_panel(35, &anchors, 90.0);
        // Prior 5 bars ascend 80 → 90 to set a positive prior_trend_slope.
        for (i, c) in p.close.iter_mut().take(5).enumerate() {
            *c = 80.0 + (i as f64) * 2.0;
        }
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(5, 112.0, PivotKind::High),
            pivot(9, 89.0, PivotKind::Low),
            pivot(14, 107.0, PivotKind::High),
            pivot(18, 95.0, PivotKind::Low),
            pivot(22, 102.0, PivotKind::High),
        ];
        let raw = SymmetricalTriangleDetector::default().detect(&pivots, v);
        assert!(
            !raw.is_empty(),
            "expected at least one symmetrical-triangle detection"
        );
        let det = &raw[0];
        assert_eq!(det.name, "symmetrical_triangle");
        assert_eq!(det.direction, Direction::Bullish);
    }

    #[test]
    fn rejects_symmetrical_when_lines_diverge() {
        // Upper rising 100 → 115, lower falling 90 → 80 — diverging, not
        // converging — so neither slope passes the symmetric check.
        let anchors = [(5, 100.0, 90.0), (22, 115.0, 80.0)];
        let p = channel_panel(35, &anchors, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(5, 100.0, PivotKind::High),
            pivot(9, 88.0, PivotKind::Low),
            pivot(14, 107.0, PivotKind::High),
            pivot(18, 84.0, PivotKind::Low),
            pivot(22, 115.0, PivotKind::High),
        ];
        let raw = SymmetricalTriangleDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }
}
