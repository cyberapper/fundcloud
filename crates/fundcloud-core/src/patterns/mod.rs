//! Chart-pattern detection.
//!
//! Pure-Rust port of `pattern_service.detection` — pivots, trend lines,
//! detectors, and a geometric scorer. The Python bindings in
//! `fundcloud-py` expose this module as `fundcloud._core.scan_pattern` and
//! its companions; nothing in this module touches Python.
//!
//! Shape of the typical call chain:
//!
//! ```text
//! OhlcvView ─► multi_level_pivots ─► PatternDetector::detect
//!                                  ─► GeometricScorer::score
//!                                  ─► Vec<Detection>
//! ```

pub mod detect;
pub mod detectors;
pub mod pivots;
pub mod scoring;
pub mod trendline;
pub mod types;

pub use detect::{detector_for, run_detector, scan, PatternDetector, ScanError};
pub use detectors::{HeadShouldersDetector, InverseHeadShouldersDetector};
pub use pivots::multi_level_pivots;
pub use scoring::GeometricScorer;
pub use trendline::{count_touches, fit_trendline, validate_boundaries};
pub use types::{
    Detection, Direction, OhlcvView, Pattern, PatternScore, Pivot, PivotKind, TrendLine,
};
