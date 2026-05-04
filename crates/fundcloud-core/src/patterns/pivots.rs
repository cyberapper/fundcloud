//! Multi-level ZigZag pivot detection.
//!
//! Ports the `pattern_service.detection.pivot.multi_level_pivots` algorithm
//! to Rust. Parity with scipy's `argrelextrema(comparator, order=N,
//! mode='clip')` is the contract — the comparator is non-strict (`>=` for
//! highs, `<=` for lows) and out-of-bounds neighbours are clipped to the
//! boundary value.
//!
//! Algorithm:
//! 1. For each `order` in the supplied set (small-to-large), detect swing
//!    highs and swing lows independently.
//! 2. Concatenate; deduplicate by (kind, |Δindex| ≤ 2) keeping the more
//!    extreme price (and the larger scale on ties).
//! 3. Sort by index.
//! 4. Walk left-to-right and merge consecutive same-kind pivots into the
//!    most-extreme one — guarantees a strictly alternating sequence.
//!
//! `argrelextrema` caveat: in the reference Python the `len < 2*order + 1`
//! gate returns `[]`. We mirror that gate per-order so a tiny window stops
//! contributing rather than spitting out boundary clips.

use crate::patterns::types::{Pivot, PivotKind};

/// Indices `i` where `cmp(x[i], x[i±k]) == true` for every `k ∈ [1, order]`.
///
/// `cmp` is the non-strict comparator (`>=` for highs, `<=` for lows). Out
/// of bounds neighbours clip to the array boundary value, matching scipy's
/// `mode='clip'` default.
fn argrel_indices<F>(x: &[f64], order: usize, cmp: F) -> Vec<usize>
where
    F: Fn(f64, f64) -> bool,
{
    let n = x.len();
    if n < 2 * order + 1 || order == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..n {
        let xi = x[i];
        let mut is_extremum = true;
        for shift in 1..=order {
            let plus = if i + shift >= n { n - 1 } else { i + shift };
            let minus = i.saturating_sub(shift);
            if !cmp(xi, x[plus]) || !cmp(xi, x[minus]) {
                is_extremum = false;
                break;
            }
        }
        if is_extremum {
            out.push(i);
        }
    }
    out
}

/// Detect raw swing highs and lows at a single `order` scale.
fn detect_swing_points_one_order(
    highs: &[f64],
    lows: &[f64],
    ts_ns: &[i64],
    order: usize,
) -> Vec<Pivot> {
    let high_idx = argrel_indices(highs, order, |a, b| a >= b);
    let low_idx = argrel_indices(lows, order, |a, b| a <= b);
    let mut pivots = Vec::with_capacity(high_idx.len() + low_idx.len());
    let order_u8 = order.min(u8::MAX as usize) as u8;
    for &idx in &high_idx {
        pivots.push(Pivot {
            index: idx,
            ts_ns: ts_ns[idx],
            price: highs[idx],
            kind: PivotKind::High,
            order: order_u8,
        });
    }
    for &idx in &low_idx {
        pivots.push(Pivot {
            index: idx,
            ts_ns: ts_ns[idx],
            price: lows[idx],
            kind: PivotKind::Low,
            order: order_u8,
        });
    }
    pivots.sort_by_key(|p| p.index);
    pivots
}

/// Walk a sorted pivot list and collapse consecutive same-kind pivots into
/// the more extreme one. Output is strictly alternating High/Low.
fn merge_to_alternating(mut pivots: Vec<Pivot>) -> Vec<Pivot> {
    if pivots.len() <= 1 {
        return pivots;
    }
    pivots.sort_by_key(|p| p.index);
    let mut result: Vec<Pivot> = Vec::with_capacity(pivots.len());
    result.push(pivots[0]);
    for p in pivots.into_iter().skip(1) {
        let last = *result.last().expect("non-empty by construction");
        if p.kind == last.kind {
            let replace = match p.kind {
                PivotKind::High => p.price >= last.price,
                PivotKind::Low => p.price <= last.price,
            };
            if replace {
                let n = result.len();
                result[n - 1] = p;
            }
        } else {
            result.push(p);
        }
    }
    result
}

/// Run pivot detection at multiple scales and merge.
///
/// Detection happens from smallest to largest `order` so that, during
/// deduplication, larger-scale pivots overwrite proximate small-scale
/// duplicates. After the dedup pass we re-merge to alternating to guard
/// against same-kind pivots that survived from different scales.
pub fn multi_level_pivots(
    highs: &[f64],
    lows: &[f64],
    ts_ns: &[i64],
    orders: &[usize],
) -> Vec<Pivot> {
    debug_assert_eq!(highs.len(), lows.len(), "highs/lows length mismatch");
    debug_assert_eq!(highs.len(), ts_ns.len(), "highs/ts length mismatch");
    if highs.is_empty() {
        return Vec::new();
    }
    let mut sorted_orders: Vec<usize> = orders.to_vec();
    sorted_orders.sort_unstable();
    sorted_orders.dedup();

    let mut all: Vec<Pivot> = Vec::new();
    for &order in &sorted_orders {
        let mut detected = detect_swing_points_one_order(highs, lows, ts_ns, order);
        all.append(&mut detected);
    }

    // Deduplicate: same kind within ±2 bars → keep more extreme price.
    // `all` was filled small-order first; on a tie we still prefer the
    // current candidate when its price is strictly more extreme.
    all.sort_by(|a, b| a.index.cmp(&b.index).then_with(|| a.kind.as_str().cmp(b.kind.as_str())));

    let mut deduped: Vec<Pivot> = Vec::new();
    for p in all {
        let mut replaced = false;
        for existing in deduped.iter_mut() {
            if existing.kind == p.kind && (existing.index as isize - p.index as isize).abs() <= 2 {
                let replace = match p.kind {
                    PivotKind::High => p.price > existing.price,
                    PivotKind::Low => p.price < existing.price,
                };
                if replace {
                    *existing = p;
                }
                replaced = true;
                break;
            }
        }
        if !replaced {
            deduped.push(p);
        }
    }

    merge_to_alternating(deduped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(seconds: i64) -> i64 {
        seconds * 1_000_000_000
    }

    #[test]
    fn argrel_max_skips_when_window_too_small() {
        let x = [1.0, 2.0, 1.0];
        // 2*order+1 = 5 > 3 → empty
        assert!(argrel_indices(&x, 2, |a, b| a >= b).is_empty());
    }

    #[test]
    fn argrel_max_finds_interior_peak() {
        let x = [1.0, 2.0, 5.0, 2.0, 1.0];
        let idx = argrel_indices(&x, 2, |a, b| a >= b);
        assert_eq!(idx, vec![2]);
    }

    #[test]
    fn argrel_min_finds_interior_trough() {
        let x = [5.0, 4.0, 1.0, 4.0, 5.0];
        let idx = argrel_indices(&x, 2, |a, b| a <= b);
        assert_eq!(idx, vec![2]);
    }

    #[test]
    fn argrel_handles_flat_plateau_via_non_strict_comparator() {
        // `>=` is non-strict so flat plateaus produce consecutive indices.
        // We do not collapse plateaus inside argrel — multi_level_pivots'
        // dedup step handles that.
        let x = [1.0, 3.0, 3.0, 3.0, 1.0];
        let idx = argrel_indices(&x, 1, |a, b| a >= b);
        assert_eq!(idx, vec![1, 2, 3]);
    }

    #[test]
    fn merge_to_alternating_collapses_consecutive_highs_to_max() {
        let pivots = vec![
            Pivot {
                index: 0,
                ts_ns: ns(0),
                price: 10.0,
                kind: PivotKind::High,
                order: 3,
            },
            Pivot {
                index: 1,
                ts_ns: ns(60),
                price: 12.0,
                kind: PivotKind::High,
                order: 3,
            },
            Pivot {
                index: 2,
                ts_ns: ns(120),
                price: 8.0,
                kind: PivotKind::Low,
                order: 3,
            },
        ];
        let merged = merge_to_alternating(pivots);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].price, 12.0);
        assert_eq!(merged[1].kind, PivotKind::Low);
    }

    #[test]
    fn multi_level_pivots_returns_alternating_sequence_on_zigzag() {
        // Zigzag: 1 → 5 → 1 → 6 → 1 → 7 → 1 → 8 → 1 → 9 → 1
        // Highs and lows alternate by construction.
        let highs: Vec<f64> = (0..11)
            .map(|i| if i % 2 == 0 { 1.0 } else { (i / 2 + 5) as f64 })
            .collect();
        let lows: Vec<f64> = (0..11)
            .map(|i| if i % 2 == 0 { 1.0 } else { (i / 2 + 5) as f64 })
            .collect();
        let ts_ns: Vec<i64> = (0..11).map(|i| ns(i as i64 * 60)).collect();
        let pivots = multi_level_pivots(&highs, &lows, &ts_ns, &[1]);
        // Strictly alternating
        for w in pivots.windows(2) {
            assert_ne!(w[0].kind, w[1].kind, "expected alternating pivots");
        }
    }

    #[test]
    fn multi_level_pivots_dedupes_across_scales() {
        // Sharp spike at index 5; both order 3 and order 5 will detect it.
        let mut highs = vec![1.0_f64; 21];
        let mut lows = vec![0.5_f64; 21];
        highs[5] = 10.0;
        lows[15] = -1.0;
        let ts_ns: Vec<i64> = (0..21).map(|i| ns(i as i64 * 60)).collect();
        let pivots = multi_level_pivots(&highs, &lows, &ts_ns, &[3, 5]);
        // We should have exactly one HIGH at idx 5 and one LOW at idx 15.
        let highs_count = pivots
            .iter()
            .filter(|p| p.kind == PivotKind::High && p.index == 5)
            .count();
        let lows_count = pivots
            .iter()
            .filter(|p| p.kind == PivotKind::Low && p.index == 15)
            .count();
        assert_eq!(highs_count, 1);
        assert_eq!(lows_count, 1);
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(multi_level_pivots(&[], &[], &[], &[3, 5]).is_empty());
    }
}
