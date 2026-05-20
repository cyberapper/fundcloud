//! Shared types for chart-pattern detection.
//!
//! These types are deliberately small, plain, and serializable so the PyO3
//! layer can hand them across the FFI boundary as Python dicts without any
//! intermediate translation.

use std::collections::HashMap;

/// Direction a pattern resolves to once confirmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    /// Bullish breakout / target above breakout level.
    Bullish,
    /// Bearish breakout / target below breakout level.
    Bearish,
    /// Continuation / undecided. Reserved for triangles before breakout.
    Neutral,
}

impl Direction {
    /// Lowercase string form used by the Python `Direction(str, Enum)`.
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Bullish => "bullish",
            Direction::Bearish => "bearish",
            Direction::Neutral => "neutral",
        }
    }
}

/// Whether a pivot is a swing high or swing low.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PivotKind {
    /// Local maximum on the highs series.
    High,
    /// Local minimum on the lows series.
    Low,
}

impl PivotKind {
    /// Uppercase string form. Matches the pattern-service `"HIGH"` / `"LOW"`
    /// literals so JSON snapshots stay byte-identical.
    pub fn as_str(self) -> &'static str {
        match self {
            PivotKind::High => "HIGH",
            PivotKind::Low => "LOW",
        }
    }
}

/// Which side of price a trend line is defined against.
///
/// Recorded at construction so [`crate::patterns::boundary_respect_ratio`]
/// scores the intended side directly. The old max-of-upper-and-lower
/// fallback saturated near 1.0 for 2-anchor patterns and killed discrimination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Resistance line (upper boundary).
    Upper,
    /// Support line (lower boundary).
    Lower,
}

impl Role {
    /// Uppercase string form; mirrors `PivotKind::as_str`.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Upper => "UPPER",
            Role::Lower => "LOWER",
        }
    }
}

/// A swing high or swing low.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pivot {
    /// Bar index into the underlying OHLCV array.
    pub index: usize,
    /// UTC timestamp in nanoseconds since the Unix epoch.
    pub ts_ns: i64,
    /// The high price for `High` pivots, the low price for `Low` pivots.
    pub price: f64,
    /// High vs. low.
    pub kind: PivotKind,
    /// `argrelextrema` order (lookback window) that detected this pivot.
    /// Carried so multi-level deduplication can prefer the larger scale.
    pub order: u8,
}

/// A least-squares trend line fitted through two or more pivots.
#[derive(Debug, Clone)]
pub struct TrendLine {
    /// Bar index of the leftmost anchor pivot.
    pub start_index: usize,
    /// Bar index of the rightmost anchor pivot.
    pub end_index: usize,
    /// Slope `dPrice / dBar`.
    pub slope: f64,
    /// y-intercept at bar `0`.
    pub intercept: f64,
    /// Coefficient of determination — capped at `[0.0, 1.0]`.
    pub r_squared: f64,
    /// Number of pivots used in the fit.
    pub touch_count: u8,
    /// See [`Role`]. Set by the detector; drives boundary-respect scoring side.
    pub role: Role,
}

impl TrendLine {
    /// Price predicted at bar `i`.
    pub fn price_at(&self, i: usize) -> f64 {
        self.slope * (i as f64) + self.intercept
    }
}

/// A detected pattern *before* quality scoring.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Stable identifier (e.g. `"head_and_shoulders"`, `"double_top"`).
    /// Lowercase `snake_case`; matches the Python `Pattern` enum value.
    pub name: &'static str,
    /// Resolved direction.
    pub direction: Direction,
    /// Pivots that anchor the formation, in chronological order.
    pub pivots: Vec<Pivot>,
    /// Trend lines (necklines, channels, triangle sides) — empty for
    /// pivot-only patterns.
    pub trend_lines: Vec<TrendLine>,
    /// First and last bar index of the formation, inclusive.
    pub formation: (usize, usize),
    /// Reference price the detector calls "entry" — usually the breakout
    /// pivot or the neckline level.
    pub entry_price: Option<f64>,
    /// Price at which the breakout occurred (None if not yet confirmed).
    pub breakout_price: Option<f64>,
    /// Optional sub-classification (e.g. `"STRICT_ADAM_ADAM"` for double
    /// tops). `None` when the pattern family has no variants.
    pub variant: Option<String>,
}

/// Geometric quality grade for a detected pattern.
#[derive(Debug, Clone, Default)]
pub struct PatternScore {
    /// `0.0..=100.0`; higher is better.
    pub quality: f64,
    /// Optional named features the scorer used (slope, neckline tilt, etc.).
    pub features: HashMap<String, f64>,
}

/// What the public `scan_pattern` returns: the formation plus its score.
#[derive(Debug, Clone)]
pub struct Detection {
    /// The detected formation.
    pub pattern: Pattern,
    /// Geometric quality grade.
    pub score: PatternScore,
}

/// Borrowed view of an OHLCV panel handed to detectors and scorers.
///
/// All slices must be the same length. Timestamps are nanoseconds since the
/// Unix epoch in monotonic ascending order.
#[derive(Debug, Clone, Copy)]
pub struct OhlcvView<'a> {
    /// UTC timestamps in nanoseconds.
    pub ts_ns: &'a [i64],
    /// Open prices.
    pub open: &'a [f64],
    /// High prices.
    pub high: &'a [f64],
    /// Low prices.
    pub low: &'a [f64],
    /// Close prices.
    pub close: &'a [f64],
    /// Volume — may be all-zeros when not provided by the caller.
    pub volume: &'a [f64],
}

impl<'a> OhlcvView<'a> {
    /// Number of bars.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Whether the panel is empty.
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}
