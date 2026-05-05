//! Detector trait + the public `scan` entry-point.
//!
//! Each detector exposes a stable lowercase `name` (matching the Python
//! `Pattern` enum value) and a `detect` function that walks a pivot list
//! and emits raw `Pattern` instances. The `scan` function wires pivots →
//! detector → `GeometricScorer` and applies the `min_quality` cutoff.

use std::collections::HashMap;

use crate::patterns::detectors::{
    AscendingTriangleDetector, DescendingTriangleDetector, DoubleBottomDetector, DoubleTopDetector,
    HeadShouldersDetector, InverseHeadShouldersDetector, SymmetricalTriangleDetector,
    TripleBottomDetector, TripleTopDetector,
};
use crate::patterns::pivots::multi_level_pivots;
use crate::patterns::scoring::GeometricScorer;
use crate::patterns::types::{Detection, OhlcvView, Pattern, Pivot};

/// Per-detector tunable parameters keyed by their Python-facing name.
///
/// Unknown keys are ignored so callers can pass a single dict to
/// [`scan`] without per-detector branching. Each detector implementation
/// reads only the keys it understands; missing keys fall back to the
/// detector's `Default` impl.
pub type DetectorParams = HashMap<String, f64>;

fn get_f64(params: &DetectorParams, key: &str) -> Option<f64> {
    params.get(key).copied()
}

fn get_usize(params: &DetectorParams, key: &str) -> Option<usize> {
    params
        .get(key)
        .and_then(|v| if *v >= 0.0 { Some(*v as usize) } else { None })
}

/// A pattern detector that walks pivots looking for one specific formation.
pub trait PatternDetector: Send + Sync {
    /// Stable lowercase identifier (e.g. `"head_and_shoulders"`).
    fn name(&self) -> &'static str;

    /// Scan `pivots` for instances of this pattern.
    ///
    /// `ohlcv` is the full panel — detectors look at it for prior-trend
    /// gating, neckline confirmation, etc. Pivots are guaranteed
    /// alternating High/Low in chronological order.
    fn detect(&self, pivots: &[Pivot], ohlcv: OhlcvView<'_>) -> Vec<Pattern>;
}

/// Linear-fit slope over the `window` bars immediately before
/// `formation_start`. Positive ⇒ uptrend, negative ⇒ downtrend.
///
/// Returns `0.0` when there isn't enough history (< 3 bars) or the data
/// is degenerate. Detectors that gate on trend direction should treat
/// `0.0` as "no signal" (reject), not as "flat — either is fine".
///
/// Mirrors `prior_trend_slope` in
/// `pattern_service.detection.patterns.base`.
pub fn prior_trend_slope(closes: &[f64], formation_start: usize, window: usize) -> f64 {
    let formation_start = formation_start.min(closes.len());
    let lo = formation_start.saturating_sub(window);
    let n = formation_start - lo;
    if n < 3 {
        return 0.0;
    }
    let xs: Vec<f64> = (lo..formation_start).map(|i| i as f64).collect();
    let ys = &closes[lo..formation_start];
    let n_f = n as f64;
    let x_mean: f64 = xs.iter().sum::<f64>() / n_f;
    let y_mean: f64 = ys.iter().sum::<f64>() / n_f;
    if y_mean == 0.0 {
        return 0.0;
    }
    let mut s_xx = 0.0;
    let mut s_xy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let xc = x - x_mean;
        let yc = y - y_mean;
        s_xx += xc * xc;
        s_xy += xc * yc;
    }
    if s_xx == 0.0 {
        0.0
    } else {
        s_xy / s_xx
    }
}

/// Run a single detector against a pre-computed pivot list and apply the
/// geometric scorer. Detections below `min_quality` are dropped.
pub fn run_detector<D: PatternDetector + ?Sized>(
    detector: &D,
    pivots: &[Pivot],
    ohlcv: OhlcvView<'_>,
    min_quality: f64,
) -> Vec<Detection> {
    let scorer = GeometricScorer;
    detector
        .detect(pivots, ohlcv)
        .into_iter()
        .filter_map(|p| {
            let score = scorer.score(&p, ohlcv);
            if score.quality >= min_quality {
                Some(Detection { pattern: p, score })
            } else {
                None
            }
        })
        .collect()
}

/// Errors surfaced by the public scan entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanError {
    /// Pattern name not registered in this build.
    UnknownPattern(String),
    /// OHLCV column lengths disagree.
    LengthMismatch,
}

impl std::fmt::Display for ScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScanError::UnknownPattern(name) => write!(f, "unknown pattern: {name:?}"),
            ScanError::LengthMismatch => f.write_str("OHLCV column lengths disagree"),
        }
    }
}

impl std::error::Error for ScanError {}

/// Resolve a stable lowercase pattern name to a boxed detector instance.
///
/// Returns `Err(ScanError::UnknownPattern)` for names not in the v1
/// catalogue. Names match the Python `Pattern(str, Enum)` values.
///
/// `params` overrides the detector's `Default` for any keys it
/// recognises. See [`DetectorParams`] and the `knobs.md` reference for
/// the per-pattern key list.
pub fn detector_for(
    name: &str,
    params: &DetectorParams,
) -> Result<Box<dyn PatternDetector>, ScanError> {
    match name {
        "head_and_shoulders" => {
            let mut d = HeadShouldersDetector::default();
            if let Some(v) = get_f64(params, "shoulder_tolerance") {
                d.shoulder_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_head_prominence") {
                d.min_head_prominence = v;
            }
            if let Some(v) = get_usize(params, "prior_trend_window") {
                d.prior_trend_window = v;
            }
            Ok(Box::new(d))
        }
        "inverse_head_and_shoulders" => {
            let mut d = InverseHeadShouldersDetector::default();
            if let Some(v) = get_f64(params, "shoulder_tolerance") {
                d.shoulder_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_head_prominence") {
                d.min_head_prominence = v;
            }
            if let Some(v) = get_usize(params, "prior_trend_window") {
                d.prior_trend_window = v;
            }
            Ok(Box::new(d))
        }
        "double_top" => {
            let mut d = DoubleTopDetector::default();
            if let Some(v) = get_f64(params, "peak_tolerance") {
                d.peak_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_trough_depth") {
                d.min_trough_depth = v;
            }
            Ok(Box::new(d))
        }
        "double_bottom" => {
            let mut d = DoubleBottomDetector::default();
            if let Some(v) = get_f64(params, "trough_tolerance") {
                d.trough_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_peak_height") {
                d.min_peak_height = v;
            }
            Ok(Box::new(d))
        }
        "triple_top" => {
            let mut d = TripleTopDetector::default();
            if let Some(v) = get_f64(params, "peak_tolerance") {
                d.peak_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_trough_depth") {
                d.min_trough_depth = v;
            }
            if let Some(v) = get_usize(params, "min_bar_count") {
                d.min_bar_count = v;
            }
            Ok(Box::new(d))
        }
        "triple_bottom" => {
            let mut d = TripleBottomDetector::default();
            if let Some(v) = get_f64(params, "trough_tolerance") {
                d.trough_tolerance = v;
            }
            if let Some(v) = get_f64(params, "min_peak_height") {
                d.min_peak_height = v;
            }
            if let Some(v) = get_usize(params, "min_bar_count") {
                d.min_bar_count = v;
            }
            Ok(Box::new(d))
        }
        "ascending_triangle" => {
            let mut d = AscendingTriangleDetector::default();
            if let Some(v) = get_f64(params, "flat_threshold") {
                d.flat_threshold = v;
            }
            if let Some(v) = get_usize(params, "min_touches") {
                d.min_touches = v;
            }
            Ok(Box::new(d))
        }
        "descending_triangle" => {
            let mut d = DescendingTriangleDetector::default();
            if let Some(v) = get_f64(params, "flat_threshold") {
                d.flat_threshold = v;
            }
            if let Some(v) = get_usize(params, "min_touches") {
                d.min_touches = v;
            }
            Ok(Box::new(d))
        }
        "symmetrical_triangle" => {
            let mut d = SymmetricalTriangleDetector::default();
            if let Some(v) = get_f64(params, "min_slope_threshold") {
                d.min_slope_threshold = v;
            }
            if let Some(v) = get_usize(params, "min_touches") {
                d.min_touches = v;
            }
            if let Some(v) = get_usize(params, "min_bar_count") {
                d.min_bar_count = v;
            }
            if let Some(v) = get_usize(params, "prior_trend_window") {
                d.prior_window = v;
            }
            Ok(Box::new(d))
        }
        other => Err(ScanError::UnknownPattern(other.to_string())),
    }
}

/// Top-level scan: build pivots, dispatch to the named detector, score,
/// filter by `min_quality`. This is the function PyO3 binds to.
///
/// `pivot_orders` is the multi-level lookback set (typically `[3, 5, 8]`).
/// `params` carries per-detector overrides; pass an empty map for defaults.
pub fn scan(
    name: &str,
    ohlcv: OhlcvView<'_>,
    pivot_orders: &[usize],
    min_quality: f64,
    params: &DetectorParams,
) -> Result<Vec<Detection>, ScanError> {
    let n = ohlcv.close.len();
    if ohlcv.high.len() != n
        || ohlcv.low.len() != n
        || ohlcv.open.len() != n
        || ohlcv.ts_ns.len() != n
        || (!ohlcv.volume.is_empty() && ohlcv.volume.len() != n)
    {
        return Err(ScanError::LengthMismatch);
    }
    let detector = detector_for(name, params)?;
    let pivots = multi_level_pivots(ohlcv.high, ohlcv.low, ohlcv.ts_ns, pivot_orders);
    Ok(run_detector(detector.as_ref(), &pivots, ohlcv, min_quality))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_trend_slope_zero_with_too_little_history() {
        let closes = [1.0, 2.0];
        assert_eq!(prior_trend_slope(&closes, 1, 10), 0.0);
    }

    #[test]
    fn prior_trend_slope_positive_for_uptrend() {
        // closes: 1, 2, 3, 4, 5 — formation starts at index 5 (out of bounds
        // is fine for the slope helper since it only reads [0..5]).
        let closes = [1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = prior_trend_slope(&closes, 5, 5);
        assert!(slope > 0.0);
    }

    #[test]
    fn prior_trend_slope_negative_for_downtrend() {
        let closes = [5.0, 4.0, 3.0, 2.0, 1.0];
        let slope = prior_trend_slope(&closes, 5, 5);
        assert!(slope < 0.0);
    }

    #[test]
    fn prior_trend_slope_zero_for_flat_zero_mean() {
        let closes = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(prior_trend_slope(&closes, 5, 5), 0.0);
    }
}
