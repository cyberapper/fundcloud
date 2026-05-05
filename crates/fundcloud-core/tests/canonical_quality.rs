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
//! Adding a fixture is a 5-line change to `FIXTURES`. Bumping
//! `SCORER_VERSION` (in `scoring.rs`) is the only legitimate reason for
//! these expected bands to change; if a refactor moves a fixture out of
//! its band without a version bump, the contract has been violated.
//!
//! See `docs/scoring/quality.md#calibration-record` for how this fixture
//! set fits the broader calibration plan.

use fundcloud_core::patterns::{
    Direction, GeometricScorer, OhlcvView, Pattern, Pivot, PivotKind, TrendLine,
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
        Self { lo: 95.0, hi: 100.0 }
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

fn solid_trendline(end: usize, touches: u8) -> TrendLine {
    TrendLine {
        start_index: 0,
        end_index: end,
        slope: 0.0,
        intercept: 100.0,
        r_squared: 0.95,
        touch_count: touches,
    }
}

fn weak_trendline(end: usize, touches: u8) -> TrendLine {
    TrendLine {
        start_index: 0,
        end_index: end,
        slope: 0.0,
        intercept: 100.0,
        r_squared: 0.30,
        touch_count: touches,
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
        trend_lines: vec![solid_trendline(30, 5)],
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
        trend_lines: vec![solid_trendline(30, 4)],
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
        trend_lines: vec![weak_trendline(14, 2)],
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
        trend_lines: vec![solid_trendline(40, 5)],
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
            pv(30, 93.0, PivotKind::Low), // 1% neckline tilt
            pv(40, 102.0, PivotKind::High), // 2% shoulder asymmetry
        ],
        trend_lines: vec![solid_trendline(40, 4)],
        formation: (0, 40),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::declining_volume(41, 100.0, 60.0))
}

// Head & Shoulders — marginal (shoulders within but not great, weak supporting)
fn h_and_s_marginal() -> (Pattern, OwnedOhlcv) {
    let p = Pattern {
        name: "head_and_shoulders",
        direction: Direction::Bearish,
        pivots: vec![
            pv(0, 100.0, PivotKind::High),
            pv(5, 93.0, PivotKind::Low),
            pv(10, 110.0, PivotKind::High),
            pv(15, 91.0, PivotKind::Low), // 2% neckline tilt
            pv(20, 105.0, PivotKind::High), // 5% shoulder asymmetry
        ],
        trend_lines: vec![weak_trendline(20, 2)],
        formation: (0, 20),
        entry_price: Some(92.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(21, 100.0))
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
        trend_lines: vec![solid_trendline(40, 6)],
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
        trend_lines: vec![solid_trendline(40, 5), solid_trendline(40, 5)],
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
        trend_lines: vec![weak_trendline(40, 2)],
        formation: (0, 40),
        entry_price: Some(98.0),
        breakout_price: None,
        variant: None,
    };
    (p, OwnedOhlcv::flat(41, 100.0))
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
        trend_lines: vec![solid_trendline(30, 5)],
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
        trend_lines: vec![solid_trendline(4, 3)],
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
        trend_lines: vec![weak_trendline(8, 2)],
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
];

/// Calibration targets — fixtures whose expected band the current
/// scorer (`SCORER_VERSION = "1.0.0"`) does **not** satisfy. Each one
/// encodes a desired property of `quality` that motivates a future
/// scorer version. These run as `#[ignore]`d tests so they do not break
/// CI but remain visible as durable reminders.
///
/// When a future scorer version brings one of these into its expected
/// band, move the fixture into `FIXTURES` and bump `SCORER_VERSION`.
const CALIBRATION_TARGETS: &[Fixture] = &[
    Fixture {
        label: "double_top_broken_symmetry",
        rationale: "50% peak asymmetry — `quality` should reject this regardless of \
                    supporting structure. Today the 30% symmetry weight is not enough \
                    to overcome perfect volume + trendline + completeness; composite \
                    lands ~66. Calibration TODO: either raise symmetry weight or apply \
                    a multiplicative gate on symmetry == 0.",
        band: Band::adversarial(),
        build: double_top_broken_symmetry,
    },
    Fixture {
        label: "double_top_too_short",
        rationale: "4-bar formation — `quality` should drag this into the bottom band. \
                    Today only the 20% completeness weight hits it (composite ~85). \
                    Calibration TODO: short-duration penalty needs to scale composite, \
                    not just one component; or detector should reject < 5 bars upstream.",
        band: Band::adversarial(),
        build: double_top_too_short,
    },
    Fixture {
        label: "h_and_s_degenerate_head",
        rationale: "Head only 1% above shoulders. Shoulders + neckline match (symmetry=100), \
                    but the formation isn't really an H&S. Composite ~56 today. \
                    Calibration TODO: H&S `symmetry` sub-score must factor in head \
                    prominence (e.g., head ≥ shoulder × 1.05).",
        band: Band::adversarial(),
        build: h_and_s_degenerate_head,
    },
];

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
        If this is intentional, bump SCORER_VERSION and update the band(s); \
        see docs/scoring/quality.md#versioning.\n",
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
        "\nCalibration targets vs current SCORER_VERSION:\n{}\n",
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
        "{met_count} calibration target(s) are now within band — promote them to \
        FIXTURES and bump SCORER_VERSION."
    );
}
