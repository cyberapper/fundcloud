//! Head-and-shoulders detector pair (regular + inverse).
//!
//! Port of `pattern_service.detection.patterns.headshoulders`. The two
//! detectors are mirrors: regular H&S looks for `H-L-H-L-H` after an
//! uptrend (bearish reversal); inverse looks for `L-H-L-H-L` after a
//! downtrend (bullish reversal).

use crate::patterns::detect::{prior_trend_slope, PatternDetector};
use crate::patterns::trendline::fit_trendline;
use crate::patterns::types::{Direction, OhlcvView, Pattern, Pivot, PivotKind};

/// Window (in bars) used to infer the pre-formation trend. Matches
/// `PRIOR_TREND_WINDOW` in the reference Python.
const PRIOR_TREND_WINDOW: usize = 10;
/// Minimum bar count between the first and last pivot of the formation.
const MIN_FORMATION_BARS: usize = 8;
/// Default shoulder-symmetry tolerance (10%).
const DEFAULT_SHOULDER_TOLERANCE: f64 = 0.10;
/// Default minimum head prominence above the average shoulder (3%).
const DEFAULT_MIN_HEAD_PROMINENCE: f64 = 0.03;

/// Absolute percentage difference using the average magnitude as
/// denominator (same definition used by the geometric scorer).
fn pct_diff(a: f64, b: f64) -> f64 {
    let avg = (a.abs() + b.abs()) / 2.0;
    if avg == 0.0 {
        0.0
    } else {
        (a - b).abs() / avg
    }
}

/// Detect bearish "Head and Shoulders" reversals.
#[derive(Debug, Clone)]
pub struct HeadShouldersDetector {
    /// Maximum allowed `pct_diff` between the two shoulders.
    pub shoulder_tolerance: f64,
    /// Minimum prominence of the head above the average shoulder.
    pub min_head_prominence: f64,
    /// Bars before the left shoulder used to verify the prior uptrend.
    pub prior_trend_window: usize,
}

impl Default for HeadShouldersDetector {
    fn default() -> Self {
        Self {
            shoulder_tolerance: DEFAULT_SHOULDER_TOLERANCE,
            min_head_prominence: DEFAULT_MIN_HEAD_PROMINENCE,
            prior_trend_window: PRIOR_TREND_WINDOW,
        }
    }
}

impl PatternDetector for HeadShouldersDetector {
    fn name(&self) -> &'static str {
        "head_and_shoulders"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 5 {
            return out;
        }
        let closes = ohlcv.close;

        for w in pivots.windows(5) {
            let (h1, l1, h2, l2, h3) = (w[0], w[1], w[2], w[3], w[4]);

            // Sequence: H-L-H-L-H.
            if h1.kind != PivotKind::High
                || l1.kind != PivotKind::Low
                || h2.kind != PivotKind::High
                || l2.kind != PivotKind::Low
                || h3.kind != PivotKind::High
            {
                continue;
            }

            // Head must be the highest of the three highs.
            if h2.price <= h1.price || h2.price <= h3.price {
                continue;
            }
            // Shoulders within tolerance.
            if pct_diff(h1.price, h3.price) > self.shoulder_tolerance {
                continue;
            }
            // Head prominence relative to the shoulders.
            let avg_shoulder = (h1.price + h3.price) / 2.0;
            if avg_shoulder == 0.0 {
                continue;
            }
            let prominence = (h2.price - avg_shoulder) / avg_shoulder;
            if prominence < self.min_head_prominence {
                continue;
            }
            // Minimum duration.
            if h3.index.saturating_sub(h1.index) < MIN_FORMATION_BARS {
                continue;
            }
            // Reversal gating: require a prior uptrend.
            if prior_trend_slope(closes, h1.index, self.prior_trend_window) <= 0.0 {
                continue;
            }

            // Neckline through the two lows; resistance through the two
            // shoulder highs.
            let neckline = fit_trendline(&[l1, l2]);
            let resistance = fit_trendline(&[h1, h3]);

            let neckline_price = match &neckline {
                Some(tl) => tl.price_at(h3.index),
                None => (l1.price + l2.price) / 2.0,
            };

            let mut trend_lines = Vec::new();
            if let Some(tl) = neckline {
                trend_lines.push(tl);
            }
            if let Some(tl) = resistance {
                trend_lines.push(tl);
            }

            out.push(Pattern {
                name: "head_and_shoulders",
                direction: Direction::Bearish,
                pivots: vec![h1, l1, h2, l2, h3],
                trend_lines,
                formation: (h1.index, h3.index),
                entry_price: Some(neckline_price),
                breakout_price: Some(neckline_price),
                variant: None,
            });
        }
        out
    }
}

/// Detect bullish "Inverse Head and Shoulders" reversals.
#[derive(Debug, Clone)]
pub struct InverseHeadShouldersDetector {
    /// Maximum allowed `pct_diff` between the two shoulders.
    pub shoulder_tolerance: f64,
    /// Minimum prominence of the head below the average shoulder.
    pub min_head_prominence: f64,
    /// Bars before the left shoulder used to verify the prior downtrend.
    pub prior_trend_window: usize,
}

impl Default for InverseHeadShouldersDetector {
    fn default() -> Self {
        Self {
            shoulder_tolerance: DEFAULT_SHOULDER_TOLERANCE,
            min_head_prominence: DEFAULT_MIN_HEAD_PROMINENCE,
            prior_trend_window: PRIOR_TREND_WINDOW,
        }
    }
}

impl PatternDetector for InverseHeadShouldersDetector {
    fn name(&self) -> &'static str {
        "inverse_head_and_shoulders"
    }

    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern> {
        let mut out = Vec::new();
        if pivots.len() < 5 {
            return out;
        }
        let closes = ohlcv.close;

        for w in pivots.windows(5) {
            let (l1, h1, l2, h2, l3) = (w[0], w[1], w[2], w[3], w[4]);

            // Sequence: L-H-L-H-L.
            if l1.kind != PivotKind::Low
                || h1.kind != PivotKind::High
                || l2.kind != PivotKind::Low
                || h2.kind != PivotKind::High
                || l3.kind != PivotKind::Low
            {
                continue;
            }

            // Head must be the lowest of the three lows.
            if l2.price >= l1.price || l2.price >= l3.price {
                continue;
            }
            if pct_diff(l1.price, l3.price) > self.shoulder_tolerance {
                continue;
            }
            let avg_shoulder = (l1.price + l3.price) / 2.0;
            if avg_shoulder == 0.0 {
                continue;
            }
            let prominence = (avg_shoulder - l2.price) / avg_shoulder;
            if prominence < self.min_head_prominence {
                continue;
            }
            if l3.index.saturating_sub(l1.index) < MIN_FORMATION_BARS {
                continue;
            }
            // Reversal gating: require a prior downtrend.
            if prior_trend_slope(closes, l1.index, self.prior_trend_window) >= 0.0 {
                continue;
            }

            // Neckline through the two highs; support through the two
            // shoulder lows.
            let neckline = fit_trendline(&[h1, h2]);
            let support = fit_trendline(&[l1, l3]);

            let neckline_price = match &neckline {
                Some(tl) => tl.price_at(l3.index),
                None => (h1.price + h2.price) / 2.0,
            };

            let mut trend_lines = Vec::new();
            if let Some(tl) = neckline {
                trend_lines.push(tl);
            }
            if let Some(tl) = support {
                trend_lines.push(tl);
            }

            out.push(Pattern {
                name: "inverse_head_and_shoulders",
                direction: Direction::Bullish,
                pivots: vec![l1, h1, l2, h2, l3],
                trend_lines,
                formation: (l1.index, l3.index),
                entry_price: Some(neckline_price),
                breakout_price: Some(neckline_price),
                variant: None,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::detect::run_detector;

    fn pivot(index: usize, price: f64, kind: PivotKind) -> Pivot {
        Pivot {
            index,
            ts_ns: index as i64 * 60 * 1_000_000_000,
            price,
            kind,
            order: 5,
        }
    }

    /// Synthetic OHLCV panel used by the H&S tests. Holds the five
    /// columns we need without tripping clippy's type-complexity lint.
    struct Panel {
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    }

    /// Build an OHLCV panel that satisfies the prior-uptrend gate and the
    /// 5-pivot H&S geometry. Formation runs from bar 10 to bar 30.
    fn synthetic_h_and_s_panel() -> Panel {
        // Closes: 10 bars rising 90 → 100, then 21 bars of formation.
        let mut close = Vec::with_capacity(31);
        for i in 0..10 {
            close.push(90.0 + i as f64);
        }
        // Formation closes: trace the H-L-H-L-H shape (10..=30, 21 bars).
        // Pivots will be at 10, 15, 20, 25, 30.
        let formation = [
            // bar -> close
            (10, 100.0), // H1
            (15, 92.0),  // L1
            (20, 110.0), // H2 (head)
            (25, 92.0),  // L2
            (30, 100.0), // H3
        ];
        for bar in 10..=30 {
            // Linear-interp between formation anchors.
            let mut price = 100.0;
            for w in formation.windows(2) {
                let (a_bar, a_p) = w[0];
                let (b_bar, b_p) = w[1];
                if bar >= a_bar && bar <= b_bar {
                    let t = (bar - a_bar) as f64 / (b_bar - a_bar) as f64;
                    price = a_p + t * (b_p - a_p);
                    break;
                }
            }
            close.push(price);
        }
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let open = close.clone();
        let volume = vec![1000.0; close.len()];
        Panel {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn detects_canonical_h_and_s() {
        let p = synthetic_h_and_s_panel();
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let view = OhlcvView {
            ts_ns: &ts,
            open: &p.open,
            high: &p.high,
            low: &p.low,
            close: &p.close,
            volume: &p.volume,
        };
        let pivots = vec![
            pivot(10, 100.0, PivotKind::High),
            pivot(15, 92.0, PivotKind::Low),
            pivot(20, 110.0, PivotKind::High),
            pivot(25, 92.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let detector = HeadShouldersDetector::default();
        let raw = detector.detect(&pivots, view);
        assert_eq!(raw.len(), 1, "expected exactly one H&S detection");
        let p = &raw[0];
        assert_eq!(p.name, "head_and_shoulders");
        assert_eq!(p.direction, Direction::Bearish);
        assert_eq!(p.formation, (10, 30));
        assert!(p.entry_price.is_some());
    }

    #[test]
    fn rejects_when_prior_trend_is_flat_or_down() {
        let panel = synthetic_h_and_s_panel();
        let mut close = panel.close.clone();
        for (i, c) in close.iter_mut().take(10).enumerate() {
            *c = 100.0 - i as f64;
        }
        let ts: Vec<i64> = (0..close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let view = OhlcvView {
            ts_ns: &ts,
            open: &panel.open,
            high: &panel.high,
            low: &panel.low,
            close: &close,
            volume: &panel.volume,
        };
        let pivots = vec![
            pivot(10, 100.0, PivotKind::High),
            pivot(15, 92.0, PivotKind::Low),
            pivot(20, 110.0, PivotKind::High),
            pivot(25, 92.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let detector = HeadShouldersDetector::default();
        let raw = detector.detect(&pivots, view);
        assert!(raw.is_empty(), "downtrend should fail the prior-trend gate");
    }

    #[test]
    fn rejects_when_shoulders_too_asymmetric() {
        let p = synthetic_h_and_s_panel();
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let view = OhlcvView {
            ts_ns: &ts,
            open: &p.open,
            high: &p.high,
            low: &p.low,
            close: &p.close,
            volume: &p.volume,
        };
        // Right shoulder 30% above left, while the head stays highest
        // → pct_diff ≈ 0.26 > 0.10 tolerance.
        let pivots = vec![
            pivot(10, 100.0, PivotKind::High),
            pivot(15, 92.0, PivotKind::Low),
            pivot(20, 140.0, PivotKind::High),
            pivot(25, 92.0, PivotKind::Low),
            pivot(30, 130.0, PivotKind::High),
        ];
        let detector = HeadShouldersDetector::default();
        let raw = detector.detect(&pivots, view);
        assert!(raw.is_empty());
    }

    #[test]
    fn run_detector_filters_below_min_quality() {
        // A formation that detects but scores low (no trend lines, no volume
        // declining); set min_quality high to ensure it gets filtered.
        let p = synthetic_h_and_s_panel();
        let ts: Vec<i64> = (0..p.close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let view = OhlcvView {
            ts_ns: &ts,
            open: &p.open,
            high: &p.high,
            low: &p.low,
            close: &p.close,
            volume: &p.volume,
        };
        let pivots = vec![
            pivot(10, 100.0, PivotKind::High),
            pivot(15, 92.0, PivotKind::Low),
            pivot(20, 110.0, PivotKind::High),
            pivot(25, 92.0, PivotKind::Low),
            pivot(30, 100.0, PivotKind::High),
        ];
        let detector = HeadShouldersDetector::default();
        let high_bar = run_detector(&detector, &pivots, view, 99.0);
        assert!(
            high_bar.is_empty(),
            "score should not exceed 99 for synthetic case"
        );
        let any = run_detector(&detector, &pivots, view, 0.0);
        assert!(!any.is_empty());
    }

    #[test]
    fn inverse_detects_canonical_pattern_after_downtrend() {
        // Prior 10 bars: downtrend 110 → 100. Formation: L-H-L-H-L.
        let mut close = Vec::with_capacity(31);
        for i in 0..10 {
            close.push(110.0 - i as f64);
        }
        let formation = [
            (10, 100.0), // L1
            (15, 108.0), // H1
            (20, 90.0),  // L2 (head)
            (25, 108.0), // H2
            (30, 100.0), // L3
        ];
        for bar in 10..=30 {
            let mut price = 100.0;
            for w in formation.windows(2) {
                let (a_bar, a_p) = w[0];
                let (b_bar, b_p) = w[1];
                if bar >= a_bar && bar <= b_bar {
                    let t = (bar - a_bar) as f64 / (b_bar - a_bar) as f64;
                    price = a_p + t * (b_p - a_p);
                    break;
                }
            }
            close.push(price);
        }
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        let open = close.clone();
        let volume = vec![1000.0; close.len()];
        let ts: Vec<i64> = (0..close.len() as i64)
            .map(|i| i * 60 * 1_000_000_000)
            .collect();
        let view = OhlcvView {
            ts_ns: &ts,
            open: &open,
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        };
        let pivots = vec![
            pivot(10, 100.0, PivotKind::Low),
            pivot(15, 108.0, PivotKind::High),
            pivot(20, 90.0, PivotKind::Low),
            pivot(25, 108.0, PivotKind::High),
            pivot(30, 100.0, PivotKind::Low),
        ];
        let raw = InverseHeadShouldersDetector::default().detect(&pivots, view);
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].name, "inverse_head_and_shoulders");
        assert_eq!(raw[0].direction, Direction::Bullish);
    }
}
