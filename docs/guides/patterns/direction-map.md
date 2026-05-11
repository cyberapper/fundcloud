# Empirical direction map

Detection in this library is deliberately **direction-agnostic**. A
detector emits the geometry — neckline level, formation height, pivot
prices — and that's it. Whether a "double top" actually resolves
*down* (textbook: bearish) or *up* (the pattern fails, you fade it) is
an empirical question to be answered from data, not a textbook prior
to be hard-coded into the detector.

This page covers the small machinery that does the answering:
[`fundcloud.metrics.pattern_direction.direction_map_from_outcomes`][df].

## Why this exists

Textbook chart-pattern direction is folk wisdom dressed up as
geometry. "Symmetrical triangle continues the prior trend" is a
statement about what *most often* happened in some sample some
analyst looked at — not a property of the geometry. If you trade a
hundred symmetrical triangles on your assets, the empirical answer
might disagree with the textbook one. The map lets your data speak.

The dropped detectors and their textbook direction:

| Pattern | Textbook | Empirical (typical, data-dependent) |
|---|---|---|
| Head and Shoulders | bearish | depends |
| Inverse Head and Shoulders | bullish | depends |
| Double Top / Triple Top | bearish | depends |
| Double Bottom / Triple Bottom | bullish | depends |
| Ascending Triangle | bullish | depends |
| Descending Triangle | bearish | depends |
| Symmetrical Triangle | continuation of prior trend | depends |

Replacing "depends" with an actual direction per pattern, computed on
your bars, is what this module does.

## The classification rule

For each pattern, the function takes every event of that pattern in
the supplied events frame and computes the **mean forward close-to-close
return** at `horizon` bars after the breakout. The sign of the mean
picks the direction:

- positive mean → `Direction.BULLISH` (go long)
- negative mean → `Direction.BEARISH` (go short)
- absolute mean below `null_threshold` → `default` (undecided)
- fewer than `min_samples` events → `default` (insufficient data)

The simplicity is intentional. Mean forward return is the most
defensible single statistic — it's signed, it's interpretable, it
doesn't require choosing a stop distance, and a pattern with a small
but consistent edge gets the right sign even with high volatility. If
you need something fancier (an MFE/MAE skew test, an ML-scored
direction, a regime-conditional map), that's a follow-up — this is
the lane to start in.

## Workflow

```python
from fundcloud.features.patterns import scan_all_patterns
from fundcloud.metrics import pattern_direction as pd_
from fundcloud.strategies import PatternStrategy
from fundcloud.features.patterns import HeadAndShoulders

events = scan_all_patterns(bars)
direction_map = pd_.direction_map_from_outcomes(events, bars, horizon=20)

# direction_map = {"head_and_shoulders": <Direction.BEARISH>, ...}

strat = PatternStrategy(
    HeadAndShoulders(min_quality=70),
    direction_map=direction_map,
)
result = bars.fc.run_strategy(strat)
```

`PatternStrategy` looks up each event's pattern in the map and applies
the matching direction; missing patterns fall back to the strategy's
`direction` kwarg (default `Direction.BULLISH`). The same map is
accepted by `apply_condition(...)` directly if you want to fill
target / stop without going through the strategy:

```python
from fundcloud.features.patterns import apply_condition

events_with_levels = apply_condition(
    events,
    PatternCondition(),
    bars,
    direction_map=direction_map,
)
```

## Tuning

| Knob | Default | When to raise / lower |
|---|---|---|
| `horizon` | 20 | Match the holding period of your strategy. Long horizons average over more noise but require more lookahead-available events. |
| `min_samples` | 30 | Raise if you trust your data and want to gate against noisy small-sample classifications. Lower if your dataset is thin and you're willing to take more risk on the empirical estimate. |
| `null_threshold` | 0.0 | Raise to add a "the mean must be at least this big to commit" margin. `0.005` ≡ require a 50 bps mean forward return, useful when you'd rather default than chase a 5 bps edge. |
| `default` | `Direction.BULLISH` | The fallback for both small-sample and undecided patterns. Set per the rest of your library's "assume long" convention or invert if you're writing a short-only system. |

## What this isn't

A regime-conditional map (long in trends, short in chop). That's a
followup — see `docs/ROADMAP.md`. Likewise per-symbol direction maps,
ML-scored direction, and walk-forward refit. The current function
gives you the simplest defensible answer; the more elaborate variants
should land only when you have data showing the simple one is missing
something that matters.

[df]: ../../reference/patterns.md
