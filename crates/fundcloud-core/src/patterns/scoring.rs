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
//! - **25%** trend-line R²  — average across all fitted trend lines.
//! - **20%** completeness — duration in bars + total trend-line touches.
//!
//! All four sub-scorers return `0.0..=100.0`; the top-level scorer rounds
//! the weighted blend and clamps to `0..=100`.

use std::collections::HashMap;

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
        let trendline = score_trendline(pattern);
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

/// Trend-line fit quality: average R² across all attached trend lines.
fn score_trendline(pattern: &Pattern) -> f64 {
    if pattern.trend_lines.is_empty() {
        return 50.0;
    }
    let avg_r2: f64 = pattern
        .trend_lines
        .iter()
        .map(|tl| tl.r_squared)
        .sum::<f64>()
        / (pattern.trend_lines.len() as f64);
    avg_r2 * 100.0
}

/// Completeness: duration in bars + cumulative trend-line touch count.
fn score_completeness(pattern: &Pattern) -> f64 {
    let (start, end) = pattern.formation;
    let bar_count = end.saturating_sub(start);
    let duration_score = if bar_count < 5 {
        0.0
    } else if bar_count < 10 {
        ((bar_count - 5) as f64) / 5.0 * 50.0
    } else if bar_count <= 60 {
        100.0
    } else if bar_count <= 120 {
        100.0 - ((bar_count - 60) as f64) / 60.0 * 50.0
    } else {
        50.0
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
}
