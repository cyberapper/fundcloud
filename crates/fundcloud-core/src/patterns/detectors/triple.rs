//! Triple-top and triple-bottom detectors.
//!
//! Port of `pattern_service.detection.patterns.triple`. Triple Top is
//! `H-L-H-L-H` with all three highs within tolerance of each other (the
//! distinguishing feature vs head-and-shoulders, where the middle peak
//! is prominently above the shoulders). Triple Bottom mirrors it.
//!
//! Per Bulkowski, the breakout level is the *lowest* intervening trough
//! for tops and the *highest* intervening peak for bottoms — the level
//! the pattern actually has to close through to confirm.

use crate::patterns::detect::PatternDetector;
use crate::patterns::trendline::fit_trendline;
use crate::patterns::types::{Direction, OhlcvView, Pattern, Pivot, PivotKind, TrendLine};

/// Default maximum `pct_diff` between any peak/trough and the trio's mean.
const DEFAULT_EXTREMA_TOLERANCE: f64 = 0.02;
/// Default minimum trough depth / peak height as a fraction of the mean.
const DEFAULT_MIN_PROMINENCE: f64 = 0.02;
/// Minimum bar count between the first and last pivot.
const DEFAULT_MIN_BAR_COUNT: usize = 10;
/// Default fractional tolerance for the boundary-respect gate.
///
/// Expressed as a fraction of the average pivot price level — a 0.5% breach
/// of the support / resistance line is already a meaningful violation. This
/// is intentionally tighter than `DEFAULT_EXTREMA_TOLERANCE` (2%): the
/// extrema tolerance allows the three pivots to differ from each other by
/// up to 2%, but once the line has been fit a single intermediate bar
/// piercing it by 0.5% materially undermines the support / resistance
/// claim the pattern is making.
const DEFAULT_BOUNDARY_TOLERANCE: f64 = 0.005;

/// Absolute percentage difference using the average magnitude as the
/// denominator. Returns `0.0` when both values collapse to zero.
fn pct_diff(a: f64, b: f64) -> f64 {
    let avg = (a.abs() + b.abs()) / 2.0;
    if avg == 0.0 {
        0.0
    } else {
        (a - b).abs() / avg
    }
}

/// Reject the formation if any intermediate `low[i]` in `start..=end`
/// dips below the support line by more than `tolerance * level`.
fn respects_support(
    lows: &[f64],
    line: &TrendLine,
    start: usize,
    end: usize,
    level: f64,
    tolerance: f64,
) -> bool {
    if lows.is_empty() {
        return true;
    }
    let last = end.min(lows.len() - 1);
    let tol_amount = tolerance * level.abs();
    for (i, low) in lows.iter().enumerate().take(last + 1).skip(start) {
        if *low < line.price_at(i) - tol_amount {
            return false;
        }
    }
    true
}

/// Reject the formation if any intermediate `high[i]` in `start..=end`
/// rises above the resistance line by more than `tolerance * level`.
fn respects_resistance(
    highs: &[f64],
    line: &TrendLine,
    start: usize,
    end: usize,
    level: f64,
    tolerance: f64,
) -> bool {
    if highs.is_empty() {
        return true;
    }
    let last = end.min(highs.len() - 1);
    let tol_amount = tolerance * level.abs();
    for (i, high) in highs.iter().enumerate().take(last + 1).skip(start) {
        if *high > line.price_at(i) + tol_amount {
            return false;
        }
    }
    true
}

/// Detect bearish "Triple Top" reversals.
#[derive(Debug, Clone)]
pub struct TripleTopDetector {
    /// Maximum allowed `pct_diff` between any peak and the trio's mean.
    pub peak_tolerance: f64,
    /// Minimum trough depth as a fraction of the mean peak.
    pub min_trough_depth: f64,
    /// Minimum bar count between the first and fifth pivot.
    pub min_bar_count: usize,
    /// Fractional tolerance for the boundary-respect gate. A bar's high
    /// is allowed to pierce the resistance line by up to
    /// `boundary_tolerance * avg_peak`; anything beyond that rejects the
    /// formation. Defaults to [`DEFAULT_BOUNDARY_TOLERANCE`].
    pub boundary_tolerance: f64,
}

impl Default for TripleTopDetector {
    fn default() -> Self {
        Self {
            peak_tolerance: DEFAULT_EXTREMA_TOLERANCE,
            min_trough_depth: DEFAULT_MIN_PROMINENCE,
            min_bar_count: DEFAULT_MIN_BAR_COUNT,
            boundary_tolerance: DEFAULT_BOUNDARY_TOLERANCE,
        }
    }
}

impl PatternDetector for TripleTopDetector {
    fn name(&self) -> &'static str {
        "triple_top"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 5 {
            return out;
        }

        for w in pivots.windows(5) {
            let (p1, p2, p3, p4, p5) = (w[0], w[1], w[2], w[3], w[4]);

            // Sequence: H-L-H-L-H.
            if p1.kind != PivotKind::High
                || p2.kind != PivotKind::Low
                || p3.kind != PivotKind::High
                || p4.kind != PivotKind::Low
                || p5.kind != PivotKind::High
            {
                continue;
            }

            let peaks = [p1.price, p3.price, p5.price];
            let avg_peak = (peaks[0] + peaks[1] + peaks[2]) / 3.0;
            if avg_peak == 0.0 {
                continue;
            }
            // All three peaks within tolerance of the mean.
            if peaks
                .iter()
                .any(|&p| pct_diff(p, avg_peak) > self.peak_tolerance)
            {
                continue;
            }

            // Neckline = the *lowest* intervening trough.
            let neckline = p2.price.min(p4.price);
            let pattern_height = avg_peak - neckline;
            let depth_pct = pattern_height / avg_peak;
            if depth_pct < self.min_trough_depth {
                continue;
            }

            if p5.index.saturating_sub(p1.index) < self.min_bar_count {
                continue;
            }

            let resistance = fit_trendline(&[p1, p3, p5]);
            let mut trend_lines = Vec::new();
            if let Some(tl) = resistance {
                // Boundary-respect gate: no intermediate bar may pierce the
                // resistance line by more than `boundary_tolerance * avg_peak`.
                // Without this, a "triple top" can be reported even when
                // price breaks decisively above the resistance level between
                // the three peaks — which structurally is not a triple top.
                if !respects_resistance(
                    ohlcv.high,
                    &tl,
                    p1.index,
                    p5.index,
                    avg_peak,
                    self.boundary_tolerance,
                ) {
                    continue;
                }
                trend_lines.push(tl);
            }

            out.push(Pattern {
                name: "triple_top",
                direction: Direction::Bearish,
                pivots: vec![p1, p2, p3, p4, p5],
                trend_lines,
                formation: (p1.index, p5.index),
                entry_price: Some(neckline),
                breakout_price: Some(neckline),
                variant: None,
            });
        }
        out
    }
}

/// Detect bullish "Triple Bottom" reversals.
#[derive(Debug, Clone)]
pub struct TripleBottomDetector {
    /// Maximum allowed `pct_diff` between any trough and the trio's mean.
    pub trough_tolerance: f64,
    /// Minimum peak height as a fraction of the mean trough.
    pub min_peak_height: f64,
    /// Minimum bar count between the first and fifth pivot.
    pub min_bar_count: usize,
    /// Fractional tolerance for the boundary-respect gate. A bar's low
    /// is allowed to dip below the support line by up to
    /// `boundary_tolerance * avg_trough`; anything beyond that rejects
    /// the formation. Defaults to [`DEFAULT_BOUNDARY_TOLERANCE`].
    pub boundary_tolerance: f64,
}

impl Default for TripleBottomDetector {
    fn default() -> Self {
        Self {
            trough_tolerance: DEFAULT_EXTREMA_TOLERANCE,
            min_peak_height: DEFAULT_MIN_PROMINENCE,
            min_bar_count: DEFAULT_MIN_BAR_COUNT,
            boundary_tolerance: DEFAULT_BOUNDARY_TOLERANCE,
        }
    }
}

impl PatternDetector for TripleBottomDetector {
    fn name(&self) -> &'static str {
        "triple_bottom"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 5 {
            return out;
        }

        for w in pivots.windows(5) {
            let (p1, p2, p3, p4, p5) = (w[0], w[1], w[2], w[3], w[4]);

            // Sequence: L-H-L-H-L.
            if p1.kind != PivotKind::Low
                || p2.kind != PivotKind::High
                || p3.kind != PivotKind::Low
                || p4.kind != PivotKind::High
                || p5.kind != PivotKind::Low
            {
                continue;
            }

            let troughs = [p1.price, p3.price, p5.price];
            let avg_trough = (troughs[0] + troughs[1] + troughs[2]) / 3.0;
            if avg_trough == 0.0 {
                continue;
            }
            if troughs
                .iter()
                .any(|&t| pct_diff(t, avg_trough) > self.trough_tolerance)
            {
                continue;
            }

            // Neckline = the *highest* intervening peak.
            let neckline = p2.price.max(p4.price);
            let pattern_height = neckline - avg_trough;
            let height_pct = pattern_height / avg_trough;
            if height_pct < self.min_peak_height {
                continue;
            }

            if p5.index.saturating_sub(p1.index) < self.min_bar_count {
                continue;
            }

            let support = fit_trendline(&[p1, p3, p5]);
            let mut trend_lines = Vec::new();
            if let Some(tl) = support {
                // Boundary-respect gate: no intermediate bar may dip below
                // the support line by more than `boundary_tolerance *
                // avg_trough`. Without this, a "triple bottom" can be
                // reported even when price punches decisively below the
                // support level between troughs — which structurally is
                // not a triple bottom.
                if !respects_support(
                    ohlcv.low,
                    &tl,
                    p1.index,
                    p5.index,
                    avg_trough,
                    self.boundary_tolerance,
                ) {
                    continue;
                }
                trend_lines.push(tl);
            }

            out.push(Pattern {
                name: "triple_bottom",
                direction: Direction::Bullish,
                pivots: vec![p1, p2, p3, p4, p5],
                trend_lines,
                formation: (p1.index, p5.index),
                entry_price: Some(neckline),
                breakout_price: Some(neckline),
                variant: None,
            });
        }
        out
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

    /// Synthetic OHLCV panel for the triple-top/bottom tests.
    struct Panel {
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    }

    fn flat_panel(n: usize, level: f64) -> Panel {
        let close = vec![level; n];
        Panel {
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: vec![1000.0; n],
        }
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

    #[test]
    fn detects_canonical_triple_top() {
        let p = flat_panel(40, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        // Three peaks at 100, troughs at 95 and 96 → neckline = 95.
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(8, 95.0, PivotKind::Low),
            pivot(15, 100.0, PivotKind::High),
            pivot(22, 96.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let raw = TripleTopDetector::default().detect(&pivots, v);
        assert_eq!(raw.len(), 1);
        let det = &raw[0];
        assert_eq!(det.name, "triple_top");
        assert_eq!(det.direction, Direction::Bearish);
        assert_eq!(det.formation, (2, 30));
        assert_eq!(det.entry_price, Some(95.0));
        assert_eq!(det.breakout_price, Some(95.0));
    }

    #[test]
    fn rejects_when_middle_peak_is_prominent_head() {
        // Middle peak at 115 vs shoulders at 100 → pct_diff > 0.02.
        // This is what distinguishes triple-top from H&S.
        let p = flat_panel(40, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(8, 95.0, PivotKind::Low),
            pivot(15, 115.0, PivotKind::High),
            pivot(22, 95.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let raw = TripleTopDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }

    #[test]
    fn detects_canonical_triple_bottom() {
        let p = flat_panel(40, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::Low),
            pivot(8, 105.0, PivotKind::High),
            pivot(15, 100.0, PivotKind::Low),
            pivot(22, 104.0, PivotKind::High),
            pivot(30, 100.0, PivotKind::Low),
        ];
        let raw = TripleBottomDetector::default().detect(&pivots, v);
        assert_eq!(raw.len(), 1);
        let det = &raw[0];
        assert_eq!(det.name, "triple_bottom");
        assert_eq!(det.direction, Direction::Bullish);
        assert_eq!(det.entry_price, Some(105.0));
    }

    #[test]
    fn triple_bottom_rejects_when_intermediate_low_breaks_support() {
        // Canonical triple-bottom anchors at level 100, but bar 18 (between
        // the second and third trough) prints a low of 98 — a 2% breach of
        // the support line, far beyond the 0.5% boundary tolerance.
        let mut p = flat_panel(40, 100.0);
        p.low[18] = 98.0;
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::Low),
            pivot(8, 105.0, PivotKind::High),
            pivot(15, 100.0, PivotKind::Low),
            pivot(22, 104.0, PivotKind::High),
            pivot(30, 100.0, PivotKind::Low),
        ];
        let raw = TripleBottomDetector::default().detect(&pivots, v);
        assert!(
            raw.is_empty(),
            "should reject when intermediate low pierces the support line"
        );
    }

    #[test]
    fn triple_top_rejects_when_intermediate_high_breaks_resistance() {
        // Canonical triple-top anchors at level 100, but bar 18 (between
        // the second and third peak) prints a high of 102 — a 2% breach of
        // the resistance line, far beyond the 0.5% boundary tolerance.
        let mut p = flat_panel(40, 100.0);
        p.high[18] = 102.0;
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(8, 95.0, PivotKind::Low),
            pivot(15, 100.0, PivotKind::High),
            pivot(22, 96.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let raw = TripleTopDetector::default().detect(&pivots, v);
        assert!(
            raw.is_empty(),
            "should reject when intermediate high pierces the resistance line"
        );
    }

    #[test]
    fn rejects_when_formation_too_short() {
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        // p5.index - p1.index = 9 < min_bar_count (10).
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(4, 95.0, PivotKind::Low),
            pivot(6, 100.0, PivotKind::High),
            pivot(8, 96.0, PivotKind::Low),
            pivot(11, 100.0, PivotKind::High),
        ];
        let raw = TripleTopDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }
}
