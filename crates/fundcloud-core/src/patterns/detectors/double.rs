//! Double-top and double-bottom detectors.
//!
//! Port of `pattern_service.detection.patterns.double`. Each formation is
//! tagged with a Bulkowski-style variant string (e.g. `"STRICT_ADAM_ADAM"`,
//! `"WEAK_EVE_ADAM"`):
//!
//! - The first segment is `STRICT` when the second extreme does not breach
//!   the first (resistance/support held), `WEAK` otherwise.
//! - The second and third segments are `ADAM` (narrow spike) or `EVE`
//!   (rounded reversal) per pivot.

use crate::patterns::detect::PatternDetector;
use crate::patterns::trendline::fit_trendline;
use crate::patterns::types::{OhlcvView, Pattern, Pivot, PivotKind};

/// Default maximum `pct_diff` between the two peaks (or troughs).
const DEFAULT_EXTREMA_TOLERANCE: f64 = 0.015;
/// Default minimum trough depth / peak height as a fraction of the average
/// peak (or trough).
const DEFAULT_MIN_PROMINENCE: f64 = 0.03;
/// Minimum bar count between the two peaks (or troughs).
const MIN_FORMATION_BARS: usize = 5;
/// Adam/Eve "near the extreme" tolerance.
const ADAM_EVE_NEAR_PCT: f64 = 0.015;
/// Bars on each side of the pivot that count toward the Adam/Eve tag.
const ADAM_EVE_HALF_WINDOW: usize = 5;
/// Maximum near-bar count to qualify as Adam.
const ADAM_MAX_BARS: usize = 3;

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

/// Classify a pivot as `"ADAM"` (narrow) or `"EVE"` (rounded) based on how
/// many bars in a ±5 window sit within `ADAM_EVE_NEAR_PCT` of the pivot
/// price. Mirrors `_tag_pivot` in the reference Python.
fn tag_pivot(values: &[f64], pivot: &Pivot) -> &'static str {
    if pivot.price == 0.0 || values.is_empty() {
        return "EVE";
    }
    let lo = pivot.index.saturating_sub(ADAM_EVE_HALF_WINDOW);
    let hi = (pivot.index + ADAM_EVE_HALF_WINDOW + 1).min(values.len());
    if lo >= hi {
        return "EVE";
    }
    let abs_pivot = pivot.price.abs();
    let mut near = 0usize;
    for v in values.iter().take(hi).skip(lo) {
        if (v - pivot.price).abs() / abs_pivot <= ADAM_EVE_NEAR_PCT {
            near += 1;
        }
    }
    if near <= ADAM_MAX_BARS {
        "ADAM"
    } else {
        "EVE"
    }
}

/// Build the `STRICT/WEAK_ADAM/EVE_ADAM/EVE` variant string.
fn build_variant(strict: bool, adam_eve_left: &str, adam_eve_right: &str) -> String {
    let strict_tag = if strict { "STRICT" } else { "WEAK" };
    format!("{strict_tag}_{adam_eve_left}_{adam_eve_right}")
}

/// Detect bearish "Double Top" reversals (`H-L-H` with similar peaks).
#[derive(Debug, Clone)]
pub struct DoubleTopDetector {
    /// Maximum allowed `pct_diff` between the two peaks.
    pub peak_tolerance: f64,
    /// Minimum trough depth as a fraction of the average peak.
    pub min_trough_depth: f64,
}

impl Default for DoubleTopDetector {
    fn default() -> Self {
        Self {
            peak_tolerance: DEFAULT_EXTREMA_TOLERANCE,
            min_trough_depth: DEFAULT_MIN_PROMINENCE,
        }
    }
}

impl PatternDetector for DoubleTopDetector {
    fn name(&self) -> &'static str {
        "double_top"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 3 {
            return out;
        }
        let highs = ohlcv.high;

        for w in pivots.windows(3) {
            let (p1, p2, p3) = (w[0], w[1], w[2]);

            // Sequence: H-L-H.
            if p1.kind != PivotKind::High || p2.kind != PivotKind::Low || p3.kind != PivotKind::High
            {
                continue;
            }
            // Peaks within tolerance.
            if pct_diff(p1.price, p3.price) > self.peak_tolerance {
                continue;
            }
            // Trough must be deep enough below the peaks.
            let avg_peak = (p1.price + p3.price) / 2.0;
            if avg_peak == 0.0 {
                continue;
            }
            let trough_depth = (avg_peak - p2.price) / avg_peak;
            if trough_depth < self.min_trough_depth {
                continue;
            }
            // Minimum duration.
            if p3.index.saturating_sub(p1.index) < MIN_FORMATION_BARS {
                continue;
            }

            let resistance = fit_trendline(&[p1, p3]);
            let mut trend_lines = Vec::new();
            if let Some(tl) = resistance {
                trend_lines.push(tl);
            }

            let strict = p3.price <= p1.price;
            let left_tag = tag_pivot(highs, &p1);
            let right_tag = tag_pivot(highs, &p3);
            let variant = build_variant(strict, left_tag, right_tag);

            // Unsigned measured-move: peaks above the trough.
            let formation_height = (avg_peak - p2.price).abs();

            out.push(Pattern {
                name: "double_top",
                pivots: vec![p1, p2, p3],
                trend_lines,
                formation: (p1.index, p3.index),
                breakout_level: p2.price,
                formation_height,
                variant: Some(variant),
            });
        }
        out
    }
}

/// Detect bullish "Double Bottom" reversals (`L-H-L` with similar troughs).
#[derive(Debug, Clone)]
pub struct DoubleBottomDetector {
    /// Maximum allowed `pct_diff` between the two troughs.
    pub trough_tolerance: f64,
    /// Minimum peak height as a fraction of the average trough.
    pub min_peak_height: f64,
}

impl Default for DoubleBottomDetector {
    fn default() -> Self {
        Self {
            trough_tolerance: DEFAULT_EXTREMA_TOLERANCE,
            min_peak_height: DEFAULT_MIN_PROMINENCE,
        }
    }
}

impl PatternDetector for DoubleBottomDetector {
    fn name(&self) -> &'static str {
        "double_bottom"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 3 {
            return out;
        }
        let lows = ohlcv.low;

        for w in pivots.windows(3) {
            let (p1, p2, p3) = (w[0], w[1], w[2]);

            // Sequence: L-H-L.
            if p1.kind != PivotKind::Low || p2.kind != PivotKind::High || p3.kind != PivotKind::Low
            {
                continue;
            }
            if pct_diff(p1.price, p3.price) > self.trough_tolerance {
                continue;
            }
            let avg_trough = (p1.price + p3.price) / 2.0;
            if avg_trough == 0.0 {
                continue;
            }
            let peak_height = (p2.price - avg_trough) / avg_trough;
            if peak_height < self.min_peak_height {
                continue;
            }
            if p3.index.saturating_sub(p1.index) < MIN_FORMATION_BARS {
                continue;
            }

            let support = fit_trendline(&[p1, p3]);
            let mut trend_lines = Vec::new();
            if let Some(tl) = support {
                trend_lines.push(tl);
            }

            let strict = p3.price >= p1.price;
            let left_tag = tag_pivot(lows, &p1);
            let right_tag = tag_pivot(lows, &p3);
            let variant = build_variant(strict, left_tag, right_tag);

            // Unsigned measured-move: peak above the troughs.
            let formation_height = (p2.price - avg_trough).abs();

            out.push(Pattern {
                name: "double_bottom",
                pivots: vec![p1, p2, p3],
                trend_lines,
                formation: (p1.index, p3.index),
                breakout_level: p2.price,
                formation_height,
                variant: Some(variant),
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

    /// Synthetic OHLCV panel for the double-top/bottom tests. Holds the
    /// five columns we need without tripping clippy's type-complexity lint.
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
    fn detects_canonical_double_top() {
        // Peaks at idx 2 and 12 (price 100), trough at idx 7 (price 95).
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(7, 95.0, PivotKind::Low),
            pivot(12, 100.0, PivotKind::High),
        ];
        let raw = DoubleTopDetector::default().detect(&pivots, v);
        assert_eq!(raw.len(), 1);
        let det = &raw[0];
        assert_eq!(det.name, "double_top");
        assert_eq!(det.formation, (2, 12));
        assert!(det.variant.as_deref().unwrap().starts_with("STRICT_"));
        assert_eq!(det.breakout_level, 95.0);
        assert!(det.formation_height > 0.0);
    }

    #[test]
    fn rejects_when_peaks_too_asymmetric() {
        // Peaks 100 vs 110 → pct_diff ≈ 0.095 > 0.015 tolerance.
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(7, 95.0, PivotKind::Low),
            pivot(12, 110.0, PivotKind::High),
        ];
        let raw = DoubleTopDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }

    #[test]
    fn weak_variant_when_second_peak_higher() {
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        // Within tolerance (1% < 1.5%) but p3 > p1.
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(7, 95.0, PivotKind::Low),
            pivot(12, 101.0, PivotKind::High),
        ];
        let raw = DoubleTopDetector::default().detect(&pivots, v);
        assert_eq!(raw.len(), 1);
        assert!(raw[0].variant.as_deref().unwrap().starts_with("WEAK_"));
    }

    #[test]
    fn detects_canonical_double_bottom() {
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        let pivots = vec![
            pivot(2, 100.0, PivotKind::Low),
            pivot(7, 105.0, PivotKind::High),
            pivot(12, 100.0, PivotKind::Low),
        ];
        let raw = DoubleBottomDetector::default().detect(&pivots, v);
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].name, "double_bottom");
        assert!(raw[0].formation_height > 0.0);
        assert_eq!(raw[0].formation, (2, 12));
        assert!(raw[0].variant.as_deref().unwrap().starts_with("STRICT_"));
    }

    #[test]
    fn rejects_when_formation_too_short() {
        let p = flat_panel(20, 100.0);
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let v = view(&p, &ts);
        // Only 4 bars between peaks (< MIN_FORMATION_BARS = 5).
        let pivots = vec![
            pivot(2, 100.0, PivotKind::High),
            pivot(4, 95.0, PivotKind::Low),
            pivot(6, 100.0, PivotKind::High),
        ];
        let raw = DoubleTopDetector::default().detect(&pivots, v);
        assert!(raw.is_empty());
    }
}
