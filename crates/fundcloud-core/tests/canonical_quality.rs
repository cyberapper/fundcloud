//! Canonical fixture set for the geometric quality scorer.
//!
//! Each fixture is a hand-constructed formation paired with an expected
//! score band. The fixture set is the **contract** for what `quality`
//! should mean across the documented quality regions:
//!
//! | Band | Range | What it represents |
//! |---|---|---|
//! | excellent | 95–100 | Textbook-perfect formation |
//! | good | 70–94 | Recognisably the pattern, minor flaws |
//! | marginal | 40–69 | Pattern is detectable but several flaws |
//! | poor | 1–39 | Just barely passes the detector |
//! | adversarial | 0–25 | Should not score high — synthetic / degenerate |
//!
//! Adding a fixture is a 5-line change to `FIXTURES`. If a refactor
//! moves a fixture out of its band, either the refactor changed scorer
//! behaviour (intentionally — update the band) or it broke a property
//! (revert).

use fundcloud_core::patterns::{
    fit_trendline, Direction, GeometricScorer, OhlcvView, Pattern, Pivot, PivotKind, Role,
    TrendLine,
};

#[derive(Clone, Copy)]
struct Band {
    /// Minimum acceptable quality, inclusive.
    lo: f64,
    /// Maximum acceptable quality, inclusive.
    hi: f64,
}

impl Band {
    const fn excellent() -> Self {
        Self {
            lo: 95.0,
            hi: 100.0,
        }
    }
    const fn good() -> Self {
        Self { lo: 70.0, hi: 94.0 }
    }
    const fn marginal() -> Self {
        Self { lo: 40.0, hi: 69.0 }
    }
    #[allow(dead_code)] // Reserved for future fixtures hitting the poor band.
    const fn poor() -> Self {
        Self { lo: 1.0, hi: 39.0 }
    }
    const fn adversarial() -> Self {
        Self { lo: 0.0, hi: 25.0 }
    }
    fn contains(self, q: f64) -> bool {
        q >= self.lo && q <= self.hi
    }
}

struct Fixture {
    label: &'static str,
    rationale: &'static str,
    band: Band,
    build: fn() -> (Pattern, OwnedOhlcv),
}

/// Owned OHLCV buffer so the test can hand a borrow to the scorer.
struct OwnedOhlcv {
    ts_ns: Vec<i64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    volume: Vec<f64>,
}

impl OwnedOhlcv {
    fn flat(n: usize, vol: f64) -> Self {
        Self {
            ts_ns: (0..n as i64).map(|i| i * 60_000_000_000).collect(),
            open: vec![100.0; n],
            high: vec![100.5; n],
            low: vec![99.5; n],
            close: vec![100.0; n],
            volume: vec![vol; n],
        }
    }

    fn declining_volume(n: usize, front: f64, back: f64) -> Self {
        let mid = n / 2;
        let mut volume = vec![0.0; n];
        for (i, v) in volume.iter_mut().enumerate() {
            *v = if i < mid { front } else { back };
        }
        Self {
            ts_ns: (0..n as i64).map(|i| i * 60_000_000_000).collect(),
            open: vec![100.0; n],
            high: vec![100.5; n],
            low: vec![99.5; n],
            close: vec![100.0; n],
            volume,
        }
    }

    /// Closes scattered around `mean` with sinusoidal amplitude `amp`.
    /// Use for fixtures whose rationale wants "messy real-world bars"
    /// rather than perfectly flat synthetic data — but note that under
    /// the anchor-R² scorer the trendline sub-score no longer reads
    /// bar deviation, so this helper now matters mainly for visual /
    /// volume-component realism rather than for trendline scoring.
    fn noisy_around(n: usize, mean: f64, amp: f64) -> Self {
        let close: Vec<f64> = (0..n)
            .map(|i| mean + amp * (i as f64 * 0.7).sin())
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        Self {
            ts_ns: (0..n as i64).map(|i| i * 60_000_000_000).collect(),
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![100.0; n],
        }
    }

    fn view(&self) -> OhlcvView<'_> {
        OhlcvView {
            ts_ns: &self.ts_ns,
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
            volume: &self.volume,
        }
    }
}

fn pv(index: usize, price: f64, kind: PivotKind) -> Pivot {
    Pivot {
        index,
        ts_ns: index as i64 * 60_000_000_000,
        price,
        kind,
        order: 5,
    }
}

fn solid_trendline(end: usize, touches: u8, role: Role) -> TrendLine {
    TrendLine {
        start_index: 0,
        end_index: end,
        slope: 0.0,
        intercept: 100.0,
        r_squared: 0.95,
        touch_count: touches,
        role,
    }
}

// Hand-rolled "weak" line. Its `r_squared = 0.30` only bites the scorer
// when `touches >= 3` (the anchor-R² branch in `score_trendline`); for
// `touches == 2` the line goes through `boundary_respect_ratio` instead
// and "weakness" must come from bars actually breaching the line — see
// `h_and_s_marginal` for that pattern. Callers passing `touches == 2`
// here get a saturated trendline component (~1.0) whenever bars sit on
// the line's permitted side, which is fine for fixtures whose marginal
// character comes from other sub-scorers (symmetry, completeness).
fn weak_trendline(end: usize, touches: u8, role: Role) -> TrendLine {
    TrendLine {
        start_index: 0,
        end_index: end,
        slope: 0.0,
        intercept: 100.0,
        r_squared: 0.30,
        touch_count: touches,
        role,
    }
}

// ----------------------------------------------------------- pattern factories

// Double Top — excellent
fn double_top_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(15, 92.0, PivotKind::Low),
            pv(30, 100.0, PivotKind::High),
        ],
        trend_lines: vec![solid_trendline(30, 5, Role::Upper)],
        formation: (0, 30),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(31, 100.0, 30.0))
}

// Double Top — good (slight peak asymmetry, mild volume)
fn double_top_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(15, 92.0, PivotKind::Low),
            // 0.5% peak asymmetry — within tolerance, but not perfect.
            pv(30, 100.5, PivotKind::High),
        ],
        trend_lines: vec![solid_trendline(30, 4, Role::Upper)],
        formation: (0, 30),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(31, 100.0, 80.0))
}

// Double Top — marginal (asymmetry near tolerance, weak everything)
fn double_top_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(7, 92.0, PivotKind::Low),
            // 1% peak asymmetry — chips into symmetry score.
            pv(14, 101.0, PivotKind::High),
        ],
        trend_lines: vec![weak_trendline(14, 2, Role::Upper)],
        formation: (0, 14),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(15, 100.0))
}

// Head & Shoulders — excellent
fn h_and_s_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "head_and_shoulders",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),  // left shoulder
            pv(10, 92.0, PivotKind::Low),   // neckline left
            pv(20, 110.0, PivotKind::High), // head
            pv(30, 92.0, PivotKind::Low),   // neckline right
            pv(40, 100.0, PivotKind::High), // right shoulder
        ],
        // Resistance line through the shoulder level (intercept 100).
        trend_lines: vec![solid_trendline(40, 5, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 30.0))
}

// Head & Shoulders — good (mild shoulder asymmetry, mild neckline tilt)
fn h_and_s_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "head_and_shoulders",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 92.0, PivotKind::Low),
            pv(20, 110.0, PivotKind::High),
            pv(30, 93.0, PivotKind::Low),   // 1% neckline tilt
            pv(40, 102.0, PivotKind::High), // 2% shoulder asymmetry
        ],
        trend_lines: vec![solid_trendline(40, 4, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 60.0))
}

// Head & Shoulders — marginal (shoulders within but not great, weak supporting)
fn h_and_s_marginal() -> (Pattern, OwnedOhlcv) {
    // Neckline fit through the two intervening lows — this is the
    // *real* neckline geometry, not a hand-tweaked weak_trendline at
    // intercept=100 (which would be a resistance line, not a neckline).
    // Role::Lower because bars are expected to stay at or above the
    // neckline during the formation; some of the noisy lows breach it,
    // which is exactly the "weak supporting structure" the rationale
    // claims.
    let neckline = fit_trendline(
        &[pv(5, 93.0, PivotKind::Low), pv(15, 91.0, PivotKind::Low)],
        Role::Lower,
    )
    .expect("two pivots always produce a line");
    let p = Pattern {
        name: "head_and_shoulders",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(5, 93.0, PivotKind::Low),
            pv(10, 110.0, PivotKind::High),
            pv(15, 91.0, PivotKind::Low),   // 2% neckline tilt
            pv(20, 105.0, PivotKind::High), // 5% shoulder asymmetry
        ],
        trend_lines: vec![neckline],
        formation: (0, 20),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    // Noisy closes around 92 so lows wander above and below the neckline,
    // producing a sub-1.0 boundary-respect ratio for the 2-anchor line.
    (p, OwnedOhlcv::noisy_around(21, 92.0, 4.0))
}

// Triple Top — excellent
fn triple_top_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 92.0, PivotKind::Low),
            pv(20, 100.0, PivotKind::High),
            pv(30, 92.0, PivotKind::Low),
            pv(40, 100.0, PivotKind::High),
        ],
        // Triple-top resistance through the three peaks → Upper.
        trend_lines: vec![solid_trendline(40, 6, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 30.0))
}

// Symmetrical triangle — excellent (uniform pivot spacing)
fn triangle_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "symmetrical_triangle",
        direction: Direction::Neutral,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 95.0, PivotKind::Low),
            pv(20, 99.0, PivotKind::High),
            pv(30, 96.0, PivotKind::Low),
            pv(40, 98.0, PivotKind::High),
        ],
        // Symmetrical triangle: upper boundary (Upper) + lower boundary (Lower).
        trend_lines: vec![
            solid_trendline(40, 5, Role::Upper),
            solid_trendline(40, 5, Role::Lower),
        ],
        formation: (0, 40),
        entry_price: Some(98.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 50.0))
}

// Triangle — poor (irregular spacing, weak trendline, no volume signal)
fn triangle_poor() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "symmetrical_triangle",
        direction: Direction::Neutral,
        // Spacings 3, 18, 5, 14 → high coefficient of variation.
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(3, 95.0, PivotKind::Low),
            pv(21, 99.0, PivotKind::High),
            pv(26, 96.0, PivotKind::Low),
            pv(40, 98.0, PivotKind::High),
        ],
        // Single weak upper boundary — a poorly-fit symmetrical triangle.
        trend_lines: vec![weak_trendline(40, 2, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(98.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(41, 100.0))
}

// Anchor-collinearity demonstration: a triple-bottom whose three
// troughs are extremely collinear (`r_squared = 0.95`) — the
// trendline sub-score reads anchor-only R² and so picks this up as a
// strong supporting structure even though the bars between troughs
// rise to peaks by design.
fn triple_bottom_collinear_anchors() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(10, 100.0, PivotKind::High),
            pv(20, 92.0, PivotKind::Low),
            pv(30, 100.0, PivotKind::High),
            pv(40, 92.0, PivotKind::Low),
        ],
        // Support line through the three troughs at 92, anchor R² high.
        trend_lines: vec![TrendLine {
            start_index: 0,
            end_index: 40,
            slope: 0.0,
            intercept: 92.0,
            r_squared: 0.95,
            touch_count: 5,
            role: Role::Lower,
        }],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 30.0))
}

// Double Bottom — excellent (mirror of double_top_textbook)
fn double_bottom_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(15, 100.0, PivotKind::High),
            pv(30, 92.0, PivotKind::Low),
        ],
        trend_lines: vec![solid_trendline(30, 5, Role::Lower)],
        formation: (0, 30),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(31, 100.0, 30.0))
}

// Double Bottom — good (0.5% trough asymmetry)
fn double_bottom_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(15, 100.0, PivotKind::High),
            pv(30, 92.5, PivotKind::Low),
        ],
        trend_lines: vec![solid_trendline(30, 4, Role::Lower)],
        formation: (0, 30),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(31, 100.0, 80.0))
}

// Double Bottom — marginal (1% trough asymmetry, weak supporting)
fn double_bottom_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(7, 100.0, PivotKind::High),
            pv(14, 93.0, PivotKind::Low),
        ],
        trend_lines: vec![weak_trendline(14, 2, Role::Lower)],
        formation: (0, 14),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(15, 100.0))
}

// Inverse H&S — excellent
fn inverse_h_and_s_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "inverse_head_and_shoulders",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 90.0, PivotKind::Low),    // left shoulder
            pv(10, 100.0, PivotKind::High), // neckline left
            pv(20, 80.0, PivotKind::Low),   // head (lowest low)
            pv(30, 100.0, PivotKind::High), // neckline right
            pv(40, 90.0, PivotKind::Low),   // right shoulder
        ],
        // Neckline through the two intervening highs — Upper role.
        trend_lines: vec![solid_trendline(40, 5, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 30.0))
}

// Inverse H&S — good (mild shoulder asymmetry, mild neckline tilt)
fn inverse_h_and_s_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "inverse_head_and_shoulders",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 90.0, PivotKind::Low),
            pv(10, 100.0, PivotKind::High),
            pv(20, 80.0, PivotKind::Low),
            pv(30, 99.0, PivotKind::High), // 1% neckline tilt
            pv(40, 88.0, PivotKind::Low),  // ~2% shoulder asymmetry
        ],
        trend_lines: vec![solid_trendline(40, 4, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 60.0))
}

// Inverse H&S — marginal (5% shoulder asymmetry, 2% neckline tilt)
fn inverse_h_and_s_marginal() -> (Pattern, OwnedOhlcv) {
    // Neckline fit through the two intervening highs; bars wander above
    // and below it, producing a sub-1.0 boundary-respect ratio.
    let neckline = fit_trendline(
        &[pv(5, 99.0, PivotKind::High), pv(15, 101.0, PivotKind::High)],
        Role::Upper,
    )
    .expect("two pivots always produce a line");
    let p = Pattern {
        name: "inverse_head_and_shoulders",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 90.0, PivotKind::Low),
            pv(5, 99.0, PivotKind::High),
            pv(10, 80.0, PivotKind::Low),
            pv(15, 101.0, PivotKind::High), // 2% neckline tilt
            pv(20, 85.0, PivotKind::Low),   // ~5% shoulder asymmetry
        ],
        trend_lines: vec![neckline],
        formation: (0, 20),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(21, 100.0))
}

// Triple Top — good (0.5% peak asymmetry on each side)
fn triple_top_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 92.0, PivotKind::Low),
            pv(20, 100.5, PivotKind::High),
            pv(30, 92.0, PivotKind::Low),
            pv(40, 99.5, PivotKind::High),
        ],
        trend_lines: vec![solid_trendline(40, 4, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 70.0))
}

// Triple Top — marginal (peaks within tolerance but skewed; weak supporting)
fn triple_top_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(5, 95.0, PivotKind::Low),
            pv(10, 101.5, PivotKind::High), // 1.5% peak asymmetry
            pv(15, 95.0, PivotKind::Low),
            pv(20, 98.5, PivotKind::High), // 1.5% on the other side
        ],
        // Three anchors; lower R² for a "noisy" trio (not perfectly flat).
        trend_lines: vec![TrendLine {
            start_index: 0,
            end_index: 20,
            slope: 0.0,
            intercept: 100.0,
            r_squared: 0.50,
            touch_count: 3,
            role: Role::Upper,
        }],
        formation: (0, 20),
        entry_price: Some(95.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(21, 100.0))
}

// Triple Bottom — good (0.5% trough asymmetry on each side)
fn triple_bottom_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(10, 100.0, PivotKind::High),
            pv(20, 92.5, PivotKind::Low),
            pv(30, 100.0, PivotKind::High),
            pv(40, 91.5, PivotKind::Low),
        ],
        trend_lines: vec![TrendLine {
            start_index: 0,
            end_index: 40,
            slope: 0.0,
            intercept: 92.0,
            r_squared: 0.85,
            touch_count: 3,
            role: Role::Lower,
        }],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 70.0))
}

// Triple Bottom — marginal (1.5% trough asymmetry, weak supporting)
fn triple_bottom_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "triple_bottom",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 92.0, PivotKind::Low),
            pv(5, 100.0, PivotKind::High),
            pv(10, 93.5, PivotKind::Low),
            pv(15, 100.0, PivotKind::High),
            pv(20, 90.5, PivotKind::Low),
        ],
        trend_lines: vec![TrendLine {
            start_index: 0,
            end_index: 20,
            slope: 0.0,
            intercept: 92.0,
            r_squared: 0.50,
            touch_count: 3,
            role: Role::Lower,
        }],
        formation: (0, 20),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(21, 100.0))
}

// Ascending Triangle — excellent (uniform spacing, flat resistance, rising support)
fn ascending_triangle_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "ascending_triangle",
        direction: Direction::Bullish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 92.0, PivotKind::Low),
            pv(20, 100.0, PivotKind::High),
            pv(30, 94.0, PivotKind::Low),
            pv(40, 100.0, PivotKind::High),
        ],
        trend_lines: vec![
            // Flat resistance line.
            solid_trendline(40, 5, Role::Upper),
            // Rising support — hand-built with a positive slope.
            TrendLine {
                start_index: 0,
                end_index: 40,
                slope: 0.05,
                intercept: 92.0,
                r_squared: 0.95,
                touch_count: 5,
                role: Role::Lower,
            },
        ],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 50.0))
}

// Ascending Triangle — marginal (irregular spacing, weak trendline, flat volume)
fn ascending_triangle_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "ascending_triangle",
        direction: Direction::Bullish,
        // Spacings 3, 18, 5, 14 — high coefficient of variation (mirrors triangle_poor).
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(3, 92.0, PivotKind::Low),
            pv(21, 100.0, PivotKind::High),
            pv(26, 95.0, PivotKind::Low),
            pv(40, 100.0, PivotKind::High),
        ],
        trend_lines: vec![weak_trendline(40, 2, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(100.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(41, 100.0))
}

// Descending Triangle — excellent (uniform spacing, falling resistance, flat support)
fn descending_triangle_textbook() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "descending_triangle",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(10, 92.0, PivotKind::Low),
            pv(20, 98.0, PivotKind::High),
            pv(30, 92.0, PivotKind::Low),
            pv(40, 96.0, PivotKind::High),
        ],
        trend_lines: vec![
            // Falling resistance — hand-built with a negative slope.
            TrendLine {
                start_index: 0,
                end_index: 40,
                slope: -0.1,
                intercept: 100.0,
                r_squared: 0.95,
                touch_count: 5,
                role: Role::Upper,
            },
            // Flat support line.
            solid_trendline(40, 5, Role::Lower),
        ],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 50.0))
}

// Descending Triangle — marginal (irregular spacing, weak trendline)
fn descending_triangle_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "descending_triangle",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(3, 92.0, PivotKind::Low),
            pv(21, 98.0, PivotKind::High),
            pv(26, 92.0, PivotKind::Low),
            pv(40, 96.0, PivotKind::High),
        ],
        trend_lines: vec![weak_trendline(40, 2, Role::Upper)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(41, 100.0))
}

// Symmetrical Triangle — good (mild spacing irregularity)
fn symmetrical_triangle_good() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "symmetrical_triangle",
        direction: Direction::Neutral,
        // Spacings 9, 11, 11, 9 — small coefficient of variation.
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(9, 95.0, PivotKind::Low),
            pv(20, 99.0, PivotKind::High),
            pv(31, 96.0, PivotKind::Low),
            pv(40, 98.0, PivotKind::High),
        ],
        trend_lines: vec![
            solid_trendline(40, 4, Role::Upper),
            solid_trendline(40, 4, Role::Lower),
        ],
        formation: (0, 40),
        entry_price: Some(98.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 70.0))
}

// ---------------------------------------------------------------- adversarial

// Adversarial: peaks miles apart — should land in the bottom band even
// though the formation has good supporting volume / trendline.
fn double_top_broken_symmetry() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(15, 80.0, PivotKind::Low),
            // 50% peak asymmetry — symmetry score should be 0.
            pv(30, 150.0, PivotKind::High),
        ],
        trend_lines: vec![solid_trendline(30, 5, Role::Upper)],
        formation: (0, 30),
        entry_price: Some(80.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(31, 100.0, 30.0))
}

// Adversarial: under-bar formation — completeness should drag composite
// down even with otherwise-clean geometry.
fn double_top_too_short() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "double_top",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(2, 92.0, PivotKind::Low),
            pv(4, 100.0, PivotKind::High),
        ],
        trend_lines: vec![solid_trendline(4, 3, Role::Upper)],
        formation: (0, 4),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(5, 100.0, 50.0))
}

// Adversarial: H&S with no real pattern — head 1% above shoulders, weak
// everything. Detector might accept it; quality should not endorse.
fn h_and_s_degenerate_head() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "head_and_shoulders",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(2, 95.0, PivotKind::Low),
            pv(4, 101.0, PivotKind::High), // head barely above shoulders
            pv(6, 95.0, PivotKind::Low),
            pv(8, 100.0, PivotKind::High),
        ],
        trend_lines: vec![weak_trendline(8, 2, Role::Upper)],
        formation: (0, 8),
        entry_price: Some(95.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(9, 100.0))
}

// ---------------------------------------------------------------------- table

const FIXTURES: &[Fixture] = &[
    Fixture {
        label: "double_top_textbook",
        rationale: "Symmetric peaks, declining volume, clean trendline, sweet-spot duration.",
        band: Band::excellent(),
        build: double_top_textbook,
    },
    Fixture {
        label: "double_top_good",
        rationale: "0.5% peak asymmetry, moderate volume drop, otherwise clean.",
        band: Band::good(),
        build: double_top_good,
    },
    Fixture {
        label: "double_top_marginal",
        rationale: "1% peak asymmetry near tolerance, weak trendline, flat volume.",
        band: Band::marginal(),
        build: double_top_marginal,
    },
    Fixture {
        label: "h_and_s_textbook",
        rationale: "Symmetric shoulders, level neckline, declining volume, strong trendline.",
        band: Band::excellent(),
        build: h_and_s_textbook,
    },
    Fixture {
        label: "h_and_s_good",
        rationale: "2% shoulder asymmetry, 1% neckline tilt, moderate volume.",
        band: Band::good(),
        build: h_and_s_good,
    },
    Fixture {
        label: "h_and_s_marginal",
        rationale: "5% shoulder asymmetry, 2% neckline tilt, weak supporting structure.",
        band: Band::marginal(),
        build: h_and_s_marginal,
    },
    Fixture {
        label: "triple_top_textbook",
        rationale: "Three matching peaks, declining volume, strong trendline, full duration.",
        band: Band::excellent(),
        build: triple_top_textbook,
    },
    Fixture {
        label: "triangle_textbook",
        rationale: "Uniform pivot spacing, clean trendlines, declining volume.",
        band: Band::excellent(),
        build: triangle_textbook,
    },
    Fixture {
        label: "triangle_poor",
        rationale: "Highly irregular pivot spacing, weak trendline, flat volume — \
                    weak symmetry but supporting structure not zero.",
        band: Band::marginal(),
        build: triangle_poor,
    },
    Fixture {
        label: "triple_bottom_collinear_anchors",
        rationale: "Triple bottom with highly collinear troughs (anchor R²=0.95). \
                    Even though the bars between troughs rise to peaks (an extreme-\
                    anchor support line, per design), the anchor-only R² preserves \
                    the trendline sub-score — regression against the bug where \
                    per-bar fit collapsed extreme-anchor lines to 0.",
        band: Band::excellent(),
        build: triple_bottom_collinear_anchors,
    },
    Fixture {
        label: "double_bottom_textbook",
        rationale: "Symmetric troughs, declining volume, clean trendline, sweet-spot duration.",
        band: Band::excellent(),
        build: double_bottom_textbook,
    },
    Fixture {
        label: "double_bottom_good",
        rationale: "0.5% trough asymmetry, moderate volume drop, otherwise clean.",
        band: Band::good(),
        build: double_bottom_good,
    },
    Fixture {
        label: "double_bottom_marginal",
        rationale: "1% trough asymmetry near tolerance, weak trendline, flat volume.",
        band: Band::marginal(),
        build: double_bottom_marginal,
    },
    Fixture {
        label: "inverse_h_and_s_textbook",
        rationale: "Symmetric shoulders, level neckline, declining volume, strong trendline.",
        band: Band::excellent(),
        build: inverse_h_and_s_textbook,
    },
    Fixture {
        label: "inverse_h_and_s_good",
        rationale: "2% shoulder asymmetry, 1% neckline tilt, moderate volume.",
        band: Band::good(),
        build: inverse_h_and_s_good,
    },
    Fixture {
        label: "inverse_h_and_s_marginal",
        rationale: "5% shoulder asymmetry, 2% neckline tilt, weak supporting structure.",
        band: Band::marginal(),
        build: inverse_h_and_s_marginal,
    },
    Fixture {
        label: "triple_top_good",
        rationale: "0.5% peak asymmetry on each side, declining volume, clean trendline.",
        band: Band::good(),
        build: triple_top_good,
    },
    Fixture {
        label: "triple_top_marginal",
        rationale: "1.5% peak asymmetry on each side, weak trendline, flat volume.",
        band: Band::marginal(),
        build: triple_top_marginal,
    },
    Fixture {
        label: "triple_bottom_good",
        rationale: "0.5% trough asymmetry on each side, declining volume, strong anchors.",
        band: Band::good(),
        build: triple_bottom_good,
    },
    Fixture {
        label: "triple_bottom_marginal",
        rationale: "1.5% trough asymmetry, weak trendline, flat volume.",
        band: Band::marginal(),
        build: triple_bottom_marginal,
    },
    Fixture {
        label: "ascending_triangle_textbook",
        rationale: "Uniform pivot spacing, flat resistance, rising support, declining volume.",
        band: Band::excellent(),
        build: ascending_triangle_textbook,
    },
    Fixture {
        label: "ascending_triangle_marginal",
        rationale: "Highly irregular pivot spacing, weak trendline, flat volume.",
        band: Band::marginal(),
        build: ascending_triangle_marginal,
    },
    Fixture {
        label: "descending_triangle_textbook",
        rationale: "Uniform pivot spacing, falling resistance, flat support, declining volume.",
        band: Band::excellent(),
        build: descending_triangle_textbook,
    },
    Fixture {
        label: "descending_triangle_marginal",
        rationale: "Highly irregular pivot spacing, weak trendline, flat volume.",
        band: Band::marginal(),
        build: descending_triangle_marginal,
    },
    Fixture {
        label: "symmetrical_triangle_good",
        rationale: "Mild spacing irregularity, declining volume, otherwise clean.",
        band: Band::good(),
        build: symmetrical_triangle_good,
    },
    Fixture {
        label: "double_top_broken_symmetry",
        rationale: "50% peak asymmetry. Volume / trendline / completeness all perfect, \
                    but the symmetry-gate at the composite level crushes the score so \
                    the formation cannot be rescued by supporting structure alone.",
        band: Band::adversarial(),
        build: double_top_broken_symmetry,
    },
    Fixture {
        label: "double_top_too_short",
        rationale: "5-bar formation, otherwise perfect geometry. Duration-gate at \
                    the composite level crushes the score: patterns shorter than the \
                    detector minimum should not be reported high quality even when \
                    hand-built.",
        band: Band::adversarial(),
        build: double_top_too_short,
    },
    Fixture {
        label: "h_and_s_degenerate_head",
        rationale: "Head only 1% above shoulders. The H&S symmetry formula's \
                    head-prominence factor zeroes the sub-score for prominence < 2%; \
                    the composite symmetry-gate then crushes the score.",
        band: Band::adversarial(),
        build: h_and_s_degenerate_head,
    },
];

/// Calibration targets — fixtures encoding desired properties of
/// `quality` that the scorer does **not** yet satisfy. Each one motivates
/// a future change. These run as `#[ignore]`d tests so they don't break
/// CI but remain visible as durable reminders.
///
/// When a future scorer change brings one of these into its expected
/// band, move the fixture into `FIXTURES`.
///
/// Currently empty — the three original entries (broken-symmetry,
/// too-short duration, degenerate H&S head) were promoted into
/// `FIXTURES` when the composite gates + head-prominence factor were
/// added to the scorer. New gaps land here when found.
const CALIBRATION_TARGETS: &[Fixture] = &[];

#[test]
fn every_canonical_fixture_lands_in_its_band() {
    let scorer = GeometricScorer;
    let mut failures: Vec<String> = Vec::new();

    for fx in FIXTURES {
        let (pattern, ohlcv) = (fx.build)();
        let score = scorer.score(&pattern, ohlcv.view());
        if !fx.band.contains(score.quality) {
            failures.push(format!(
                "  {label:<32} expected [{lo}..={hi}], got {q:.1}\n      ↳ {rationale}",
                label = fx.label,
                lo = fx.band.lo,
                hi = fx.band.hi,
                q = score.quality,
                rationale = fx.rationale,
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "\n{} canonical fixture(s) outside their expected band:\n{}\n\n\
        If this is intentional, update the band(s); otherwise the scorer \
        regressed on a property the fixture is supposed to protect.\n",
        failures.len(),
        failures.join("\n"),
    );
}

#[test]
fn fixture_set_covers_required_bands() {
    use std::collections::HashSet;
    let mut bands_seen: HashSet<&'static str> = HashSet::new();
    for fx in FIXTURES.iter().chain(CALIBRATION_TARGETS) {
        let label = if fx.band.lo >= 95.0 {
            "excellent"
        } else if fx.band.lo >= 70.0 {
            "good"
        } else if fx.band.lo >= 40.0 {
            "marginal"
        } else if fx.band.lo >= 1.0 {
            "poor"
        } else {
            "adversarial"
        };
        bands_seen.insert(label);
    }
    // `poor` is not yet exercised by a passing fixture; that's tracked
    // in CALIBRATION_TARGETS. Required coverage today is the bands we
    // actually contract on plus adversarial as the calibration target.
    let required = ["excellent", "good", "marginal", "adversarial"];
    for band in required {
        assert!(
            bands_seen.contains(band),
            "fixture set missing coverage for the {band} band"
        );
    }
}

/// Calibration-target fixtures: ignored by default. Run with
/// `cargo test -p fundcloud-core --test canonical_quality -- --ignored`
/// to see how far each is from its desired band against the current scorer.
#[test]
#[ignore = "calibration TODOs — see CALIBRATION_TARGETS rationale"]
fn calibration_targets_describe_known_gaps() {
    let scorer = GeometricScorer;
    let mut report: Vec<String> = Vec::new();
    for fx in CALIBRATION_TARGETS {
        let (pattern, ohlcv) = (fx.build)();
        let score = scorer.score(&pattern, ohlcv.view());
        let status = if fx.band.contains(score.quality) {
            "MET"
        } else {
            "GAP"
        };
        report.push(format!(
            "  [{status}] {label:<32} target [{lo}..={hi}], current {q:.1}\n      ↳ {rationale}",
            label = fx.label,
            lo = fx.band.lo,
            hi = fx.band.hi,
            q = score.quality,
            rationale = fx.rationale,
        ));
    }
    // Always print so the human running the test can see the state of
    // each calibration target. Failure here means a target was MET and
    // should be promoted out of CALIBRATION_TARGETS into FIXTURES.
    eprintln!(
        "\nCalibration targets vs current scorer:\n{}\n",
        report.join("\n")
    );
    let met_count = CALIBRATION_TARGETS
        .iter()
        .filter(|fx| {
            let (p, o) = (fx.build)();
            fx.band.contains(scorer.score(&p, o.view()).quality)
        })
        .count();
    assert_eq!(
        met_count, 0,
        "{met_count} calibration target(s) are now within band — promote them to FIXTURES."
    );
}
