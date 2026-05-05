//! Geometric quality scorer (0–100) for detected patterns.
//!
//! Pure-Rust port of `pattern_service.detection.quality.compute_quality_score`.
//! Composite score:
//!
//! - **30%** symmetry — pattern-specific (e.g. shoulder/neckline symmetry
//!   for H&S, peak height match for double tops, pivot-spacing regularity
//!   for triangles).
//! - **25%** volume confirmation — declining volume during formation is a
//!   bullish/bearish confirmation signal.
//! - **25%** trend-line R² — average `trendline_fit_r2` across attached
//!   trend lines (i.e., how well intermediate bars hug each line, NOT
//!   the anchor-only R² stored on the line).
//! - **20%** completeness — duration in bars + total trend-line touches.
//!
//! All four sub-scorers return `0.0..=100.0`; the top-level scorer rounds
//! the weighted blend and clamps to `0..=100`.

use std::collections::HashMap;

use crate::patterns::trendline::trendline_fit_r2;
use crate::patterns::types::{OhlcvView, Pattern, PatternScore};

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

/// Geometric scorer — stateless; one instance can be reused across many
/// scans. Held as a struct so future calibration knobs (per-pattern
/// weight tweaks, regime overlays) have a place to land without breaking
/// the public API.
#[derive(Debug, Clone, Default)]
pub struct GeometricScorer;

impl GeometricScorer {
    /// Compute the composite score for a single detection.
    pub fn score(&self, pattern: &Pattern, ohlcv: OhlcvView<'_>) -> PatternScore {
        let symmetry = score_symmetry(pattern);
        let volume = score_volume(pattern, ohlcv);
        let trendline = score_trendline(pattern, ohlcv);
        let completeness = score_completeness(pattern);

        let raw = symmetry * 0.30 + volume * 0.25 + trendline * 0.25 + completeness * 0.20;
        let clamped = raw.round().clamp(0.0, 100.0);

        let mut features: HashMap<String, f64> = HashMap::new();
        features.insert("symmetry".to_string(), symmetry / 100.0);
        features.insert("volume".to_string(), volume / 100.0);
        features.insert("trendline_r2".to_string(), trendline / 100.0);
        features.insert("completeness".to_string(), completeness / 100.0);

        PatternScore {
            quality: clamped,
            features,
        }
    }
}

/// Symmetry score, dispatched by `pattern.name`.
///
/// Patterns the reference scorer knows about explicitly (head & shoulders,
/// double top/bottom, ascending/descending triangle) get tailored
/// formulas; everything else gets a neutral `50.0`.
fn score_symmetry(pattern: &Pattern) -> f64 {
    let pivots = &pattern.pivots;

    match pattern.name {
        "double_top" | "double_bottom" => {
            if pivots.len() < 3 {
                return 0.0;
            }
            let diff = pct_diff(pivots[0].price, pivots[2].price);
            // Perfect match = 100; 1.5% difference = 0.
            (100.0 * (1.0 - diff / 0.015)).max(0.0)
        }
        "triple_top" | "triple_bottom" => {
            // Three peaks (or troughs) at indices 0, 2, 4. Score on the
            // worst pct_diff against the trio's mean — perfect when all
            // three match exactly, zero at the detector's 2% tolerance.
            if pivots.len() < 5 {
                return 0.0;
            }
            let trio = [pivots[0].price, pivots[2].price, pivots[4].price];
            let mean = (trio[0] + trio[1] + trio[2]) / 3.0;
            let max_diff = trio
                .iter()
                .map(|p| pct_diff(*p, mean))
                .fold(0.0_f64, f64::max);
            (100.0 * (1.0 - max_diff / 0.02)).max(0.0)
        }
        "head_and_shoulders" | "inverse_head_and_shoulders" => {
            if pivots.len() < 5 {
                return 0.0;
            }
            let shoulder_diff = pct_diff(pivots[0].price, pivots[4].price);
            let neckline_diff = pct_diff(pivots[1].price, pivots[3].price);
            let shoulder_score = (100.0 * (1.0 - shoulder_diff / 0.10)).max(0.0);
            let neckline_score = (100.0 * (1.0 - neckline_diff / 0.10)).max(0.0);
            (shoulder_score + neckline_score) / 2.0
        }
        "ascending_triangle" | "descending_triangle" | "symmetrical_triangle" => {
            if pivots.len() < 4 {
                return 50.0;
            }
            let mut spacings: Vec<f64> = Vec::with_capacity(pivots.len() - 1);
            for w in pivots.windows(2) {
                spacings.push((w[1].index as f64) - (w[0].index as f64));
            }
            if spacings.len() < 2 {
                return 50.0;
            }
            let mean: f64 = spacings.iter().sum::<f64>() / (spacings.len() as f64);
            if mean == 0.0 {
                return 50.0;
            }
            let var: f64 =
                spacings.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (spacings.len() as f64);
            let cv = var.sqrt() / mean;
            (100.0 * (1.0 - cv)).max(0.0)
        }
        _ => 50.0,
    }
}

/// Volume confirmation: declining second-half volume → up to 100.
fn score_volume(pattern: &Pattern, ohlcv: OhlcvView<'_>) -> f64 {
    let (start, end) = pattern.formation;
    if end <= start || end >= ohlcv.len() {
        return 50.0;
    }
    let volumes = &ohlcv.volume[start..=end];
    if volumes.len() < 4 {
        return 50.0;
    }
    let mid = volumes.len() / 2;
    let first_half: f64 = volumes[..mid].iter().sum::<f64>() / (mid as f64);
    let second_half: f64 = volumes[mid..].iter().sum::<f64>() / ((volumes.len() - mid) as f64);
    if first_half == 0.0 {
        return 50.0;
    }
    let ratio = second_half / first_half;
    if ratio <= 0.5 {
        100.0
    } else if ratio >= 1.5 {
        0.0
    } else {
        100.0 * (1.5 - ratio)
    }
}

/// Trend-line fit quality.
///
/// Average `trendline_fit_r2` across attached trend lines, where each
/// line's contribution is the **maximum** of its fit against `close`,
/// `high`, and `low` over the formation. **Not** the average of
/// `TrendLine::r_squared` — that field is the anchor-only R² and is
/// essentially constant by construction. See
/// `crate::patterns::trendline::trendline_fit_r2` for the rationale.
///
/// Why max-of-three: a structural trend line auto-selects its native
/// price series — a neckline anchored on lows fits the lows; a triangle's
/// upper boundary anchored on highs fits the highs; a flat consolidation
/// line fits closes. Taking the max picks the right series without
/// requiring `TrendLine` to carry an explicit "anchored on" tag.
/// A spurious line that fits *none* of the three series remains correctly
/// scored low. Storing the anchor kind on `TrendLine` directly would
/// let us drop the max-of-three; not done yet to keep `TrendLine`
/// minimal.
fn score_trendline(pattern: &Pattern, ohlcv: OhlcvView<'_>) -> f64 {
    if pattern.trend_lines.is_empty() {
        return 50.0;
    }
    let avg_r2: f64 = pattern
        .trend_lines
        .iter()
        .map(|tl| {
            let against_close = trendline_fit_r2(ohlcv.close, tl);
            let against_high = trendline_fit_r2(ohlcv.high, tl);
            let against_low = trendline_fit_r2(ohlcv.low, tl);
            against_close.max(against_high).max(against_low)
        })
        .sum::<f64>()
        / (pattern.trend_lines.len() as f64);
    avg_r2 * 100.0
}

/// Completeness: duration in bars + cumulative trend-line touch count.
fn score_completeness(pattern: &Pattern) -> f64 {
    let (start, end) = pattern.formation;
    let bar_count = end.saturating_sub(start);
    // Duration is a quality floor (need enough bars for the formation to be
    // visually identifiable), not a quality ceiling. A textbook 6-month double
    // top is no less geometrically clean than a 30-bar one — they're just
    // different timeframes. Anything past the 10-bar minimum scores 100.
    let duration_score = if bar_count < 5 {
        0.0
    } else if bar_count < 10 {
        ((bar_count - 5) as f64) / 5.0 * 50.0
    } else {
        100.0
    };

    let total_touches: u32 = pattern
        .trend_lines
        .iter()
        .map(|tl| tl.touch_count as u32)
        .sum();
    let touch_score = if total_touches <= 2 {
        30.0
    } else if total_touches <= 4 {
        60.0
    } else {
        (60.0 + ((total_touches - 4) as f64) * 10.0).min(100.0)
    };

    (duration_score + touch_score) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::types::{Direction, Pivot, PivotKind, TrendLine};

    fn pv(index: usize, price: f64, kind: PivotKind) -> Pivot {
        Pivot {
            index,
            ts_ns: index as i64 * 60 * 1_000_000_000,
            price,
            kind,
            order: 5,
        }
    }

    fn ohlcv_with_flat_volume(n: usize, vol: f64) -> Vec<f64> {
        vec![vol; n]
    }

    fn view<'a>(
        close: &'a [f64],
        volume: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
    ) -> OhlcvView<'a> {
        OhlcvView {
            ts_ns: &[],
            open: close,
            high,
            low,
            close,
            volume,
        }
    }

    #[test]
    fn pct_diff_handles_zero_pair() {
        assert_eq!(pct_diff(0.0, 0.0), 0.0);
    }

    #[test]
    fn pct_diff_uses_average_magnitude() {
        // a=10, b=12 → avg=11 → diff = 2/11
        assert!((pct_diff(10.0, 12.0) - (2.0 / 11.0)).abs() < 1e-12);
    }

    #[test]
    fn double_top_symmetry_perfect_when_peaks_match() {
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, 100.0, PivotKind::High),
                pv(5, 90.0, PivotKind::Low),
                pv(10, 100.0, PivotKind::High),
            ],
            trend_lines: vec![],
            formation: (0, 10),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        assert!((score_symmetry(&p) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn double_top_symmetry_zero_when_peaks_far_off() {
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, 100.0, PivotKind::High),
                pv(5, 90.0, PivotKind::Low),
                pv(10, 200.0, PivotKind::High), // huge mismatch
            ],
            trend_lines: vec![],
            formation: (0, 10),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        assert_eq!(score_symmetry(&p), 0.0);
    }

    #[test]
    fn head_shoulders_symmetry_uses_shoulders_and_neckline() {
        let p = Pattern {
            name: "head_and_shoulders",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, 100.0, PivotKind::High),  // left shoulder
                pv(5, 90.0, PivotKind::Low),    // neckline left
                pv(10, 110.0, PivotKind::High), // head
                pv(15, 90.0, PivotKind::Low),   // neckline right
                pv(20, 100.0, PivotKind::High), // right shoulder
            ],
            trend_lines: vec![],
            formation: (0, 20),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        // Both shoulder and neckline pairs match exactly → 100.
        assert!((score_symmetry(&p) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn volume_score_high_when_second_half_decreases() {
        // Declining volume: first half avg 100, second half avg 30 → ratio 0.3 → 100.
        let volumes = vec![100.0, 100.0, 30.0, 30.0];
        let close = vec![1.0; 4];
        let high = vec![1.0; 4];
        let low = vec![1.0; 4];
        let v = view(&close, &volumes, &high, &low);
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![],
            formation: (0, 3),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        assert!((score_volume(&p, v) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn volume_score_zero_when_second_half_explodes() {
        let volumes = vec![10.0, 10.0, 100.0, 100.0]; // ratio 10
        let close = vec![1.0; 4];
        let high = vec![1.0; 4];
        let low = vec![1.0; 4];
        let v = view(&close, &volumes, &high, &low);
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![],
            formation: (0, 3),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        assert_eq!(score_volume(&p, v), 0.0);
    }

    #[test]
    fn completeness_zero_for_under_5_bar_formation() {
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![],
            formation: (0, 3),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        // duration_score 0, touch_score 30 → avg 15.
        assert!((score_completeness(&p) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn completeness_max_for_30_bar_with_5_touches() {
        let line = TrendLine {
            start_index: 0,
            end_index: 30,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 5,
        };
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![line.clone(), line],
            formation: (0, 30),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        // duration 100, touches 10 → score = (100 + (60+(10-4)*10)) / 2 = (100 + 100) / 2 = 100.
        assert!((score_completeness(&p) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn composite_score_clamps_to_unit_interval() {
        let line = TrendLine {
            start_index: 0,
            end_index: 30,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 5,
        };
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, 100.0, PivotKind::High),
                pv(15, 90.0, PivotKind::Low),
                pv(30, 100.0, PivotKind::High),
            ],
            trend_lines: vec![line],
            formation: (0, 30),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        let close = vec![100.0; 31];
        let high = vec![100.0; 31];
        let low = vec![90.0; 31];
        let volumes = ohlcv_with_flat_volume(31, 100.0);
        let v = view(&close, &volumes, &high, &low);

        let score = GeometricScorer.score(&p, v);
        assert!((0.0..=100.0).contains(&score.quality));
        assert!(score.features.contains_key("symmetry"));
        assert!(score.features.contains_key("trendline_r2"));
    }

    // ---------------------------------------------------------------- monotonicity
    //
    // Each test perturbs one geometric attribute of an otherwise-fixed
    // formation along an axis where "more textbook" is unambiguous, and
    // asserts the relevant score is monotonic in that perturbation.
    //
    // These tests do not pin specific values — they pin the *shape* of the
    // scorer's response. Rebalancing weights or tightening tolerances must
    // never reverse these orderings; if any future change does, the
    // calibration philosophy in `docs/scoring/quality.md` is being violated.

    fn assert_non_increasing(label: &str, scores: &[f64]) {
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-9,
                "{label}: expected non-increasing, got {scores:?}"
            );
        }
    }

    fn assert_non_decreasing(label: &str, scores: &[f64]) {
        for w in scores.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "{label}: expected non-decreasing, got {scores:?}"
            );
        }
    }

    fn double_top(p0: f64, p2: f64) -> Pattern {
        Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, p0, PivotKind::High),
                pv(5, 90.0, PivotKind::Low),
                pv(10, p2, PivotKind::High),
            ],
            trend_lines: vec![],
            formation: (0, 10),
            entry_price: None,
            breakout_price: None,
            variant: None,
        }
    }

    fn head_shoulders(left: f64, head: f64, right: f64, neck_l: f64, neck_r: f64) -> Pattern {
        Pattern {
            name: "head_and_shoulders",
            direction: Direction::Bearish,
            pivots: vec![
                pv(0, left, PivotKind::High),
                pv(5, neck_l, PivotKind::Low),
                pv(10, head, PivotKind::High),
                pv(15, neck_r, PivotKind::Low),
                pv(20, right, PivotKind::High),
            ],
            trend_lines: vec![],
            formation: (0, 20),
            entry_price: None,
            breakout_price: None,
            variant: None,
        }
    }

    fn triangle(spacings: &[usize]) -> Pattern {
        // Build a symmetrical triangle whose pivot spacings are exactly
        // the provided sequence. Prices alternate high/low around 100.
        let mut pivots = Vec::with_capacity(spacings.len() + 1);
        pivots.push(pv(0, 100.0, PivotKind::High));
        let mut idx = 0usize;
        for (i, &gap) in spacings.iter().enumerate() {
            idx += gap;
            let kind = if i % 2 == 0 {
                PivotKind::Low
            } else {
                PivotKind::High
            };
            let price = if i % 2 == 0 { 95.0 } else { 100.0 };
            pivots.push(pv(idx, price, kind));
        }
        Pattern {
            name: "symmetrical_triangle",
            direction: Direction::Neutral,
            pivots,
            trend_lines: vec![],
            formation: (0, idx),
            entry_price: None,
            breakout_price: None,
            variant: None,
        }
    }

    fn pattern_with_trendline(r_squared: f64, touch_count: u8, formation_len: usize) -> Pattern {
        Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![TrendLine {
                start_index: 0,
                end_index: formation_len,
                slope: 0.0,
                intercept: 0.0,
                r_squared,
                touch_count,
            }],
            formation: (0, formation_len),
            entry_price: None,
            breakout_price: None,
            variant: None,
        }
    }

    #[test]
    fn symmetry_monotonic_in_double_top_peak_skew() {
        // Hold the first peak at 100; widen the second peak's deviation
        // upward. Symmetry must monotonically decrease.
        let scores: Vec<f64> = [100.0, 100.5, 101.0, 101.5, 102.0]
            .iter()
            .map(|p2| score_symmetry(&double_top(100.0, *p2)))
            .collect();
        assert_non_increasing("double-top peak skew", &scores);
        assert!((scores[0] - 100.0).abs() < 1e-9, "perfect match should be 100");
    }

    #[test]
    fn symmetry_monotonic_in_h_and_s_shoulder_skew() {
        // Hold neckline level; widen right shoulder vs left. Symmetry must
        // monotonically decrease.
        let scores: Vec<f64> = [100.0, 102.0, 105.0, 108.0, 112.0]
            .iter()
            .map(|right| score_symmetry(&head_shoulders(100.0, 110.0, *right, 90.0, 90.0)))
            .collect();
        assert_non_increasing("h&s shoulder skew", &scores);
    }

    #[test]
    fn symmetry_monotonic_in_h_and_s_neckline_tilt() {
        // Hold shoulders symmetric; tilt the right neckline pivot. Symmetry
        // must monotonically decrease.
        let scores: Vec<f64> = [90.0, 91.0, 92.0, 93.0, 94.0]
            .iter()
            .map(|nr| score_symmetry(&head_shoulders(100.0, 110.0, 100.0, 90.0, *nr)))
            .collect();
        assert_non_increasing("h&s neckline tilt", &scores);
    }

    #[test]
    fn symmetry_monotonic_in_triangle_spacing_variance() {
        // Build triangles with progressively higher variance in pivot
        // spacing. Perfectly regular spacing should score highest;
        // increasingly irregular sequences must score no higher.
        let cases: Vec<Vec<usize>> = vec![
            vec![10, 10, 10, 10],     // uniform
            vec![9, 10, 11, 10],      // mild variance
            vec![5, 12, 8, 15],       // moderate variance
            vec![3, 18, 6, 13],       // high variance
        ];
        let scores: Vec<f64> = cases
            .iter()
            .map(|c| score_symmetry(&triangle(c)))
            .collect();
        assert_non_increasing("triangle spacing variance", &scores);
    }

    #[test]
    fn volume_monotonic_in_back_half_inflation() {
        // Front half fixed at 100; back half ramps from 30 → 200.
        // Volume confirmation score must monotonically decrease.
        let p = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![],
            formation: (0, 7),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };
        let close = vec![1.0; 8];
        let high = vec![1.0; 8];
        let low = vec![1.0; 8];

        let scores: Vec<f64> = [30.0, 50.0, 70.0, 100.0, 130.0, 160.0, 200.0]
            .iter()
            .map(|back| {
                let volumes = vec![100.0, 100.0, 100.0, 100.0, *back, *back, *back, *back];
                score_volume(&p, view(&close, &volumes, &high, &low))
            })
            .collect();
        assert_non_increasing("volume back-half inflation", &scores);
    }

    #[test]
    fn trendline_score_monotonic_in_bar_deviation_from_line() {
        // Hold a flat trend line at 100 over 30 bars. Vary how far the
        // closes wander from the line. Score must monotonically decrease
        // as bars deviate further.
        //
        // (Replaces an earlier test that varied `TrendLine::r_squared`
        // directly — that field is no longer read by the scorer; see
        // `score_trendline` doc.)
        let line = TrendLine {
            start_index: 0,
            end_index: 29,
            slope: 0.0,
            intercept: 100.0,
            r_squared: 1.0,
            touch_count: 4,
        };
        let pattern = Pattern {
            name: "double_top",
            direction: Direction::Bearish,
            pivots: vec![],
            trend_lines: vec![line],
            formation: (0, 29),
            entry_price: None,
            breakout_price: None,
            variant: None,
        };

        let high = vec![101.0; 30];
        let low = vec![99.0; 30];
        let volumes = ohlcv_with_flat_volume(30, 100.0);

        // Each amplitude scatters closes around 100 by ± amplitude.
        let amplitudes = [0.0_f64, 1.0, 5.0, 15.0, 30.0];
        let mut scores: Vec<f64> = Vec::new();
        for amplitude in amplitudes {
            let close: Vec<f64> = (0..30)
                .map(|i| 100.0 + amplitude * (i as f64 * 0.5).sin())
                .collect();
            let v = view(&close, &volumes, &high, &low);
            scores.push(score_trendline(&pattern, v));
        }
        assert_non_increasing("trendline bar-deviation", &scores);
        assert!(
            (scores[0] - 100.0).abs() < 1e-9,
            "perfect hug should score 100, got {}",
            scores[0]
        );
    }

    #[test]
    fn completeness_monotonic_through_duration_sweet_spot_ramp() {
        // Going from 5 to 10 bars, duration_score ramps from 0 → 50, so
        // completeness must monotonically increase.
        let scores: Vec<f64> = [5usize, 6, 7, 8, 9, 10]
            .iter()
            .map(|n| score_completeness(&pattern_with_trendline(1.0, 4, *n)))
            .collect();
        assert_non_decreasing("completeness ramp 5→10", &scores);
    }

    #[test]
    fn completeness_no_long_duration_penalty() {
        // Long formations are not penalised. Holding touch count constant,
        // completeness must be flat across all durations past the 10-bar
        // minimum (duration_score saturates at 100).
        let scores: Vec<f64> = [10usize, 30, 60, 90, 120, 200, 500]
            .iter()
            .map(|n| score_completeness(&pattern_with_trendline(1.0, 4, *n)))
            .collect();
        let first = scores[0];
        for (i, s) in scores.iter().enumerate() {
            assert!(
                (s - first).abs() < 1e-9,
                "duration {} should match {} at bar 10, got {} vs {}",
                i,
                first,
                s,
                first,
            );
        }
    }

    #[test]
    fn completeness_monotonic_in_touch_count() {
        // Hold duration in the sweet spot; vary touch count. More touches
        // must score no lower than fewer touches, up to the saturation
        // point.
        let scores: Vec<f64> = [1u8, 2, 3, 4, 5, 6, 8, 10]
            .iter()
            .map(|t| score_completeness(&pattern_with_trendline(1.0, *t, 30)))
            .collect();
        assert_non_decreasing("completeness touch count", &scores);
    }

    #[test]
    fn composite_quality_monotonic_in_symmetry_holding_others_fixed() {
        // Vary the symmetry component via the second peak; pin volume,
        // trendline, completeness via fixed inputs. Composite quality must
        // monotonically follow symmetry.
        let close = vec![100.0; 31];
        let high = vec![100.0; 31];
        let low = vec![90.0; 31];
        let volumes = ohlcv_with_flat_volume(31, 100.0);
        let v = view(&close, &volumes, &high, &low);
        let line = TrendLine {
            start_index: 0,
            end_index: 30,
            slope: 0.0,
            intercept: 0.0,
            r_squared: 1.0,
            touch_count: 5,
        };
        let scores: Vec<f64> = [100.0, 100.5, 101.0, 101.5, 102.0]
            .iter()
            .map(|p2| {
                let mut p = double_top(100.0, *p2);
                p.formation = (0, 30);
                p.pivots = vec![
                    pv(0, 100.0, PivotKind::High),
                    pv(15, 90.0, PivotKind::Low),
                    pv(30, *p2, PivotKind::High),
                ];
                p.trend_lines = vec![line.clone()];
                GeometricScorer.score(&p, v).quality
            })
            .collect();
        assert_non_increasing("composite quality vs symmetry", &scores);
    }
}
