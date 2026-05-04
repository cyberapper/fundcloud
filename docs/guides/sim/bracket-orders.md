# Bracket orders (stop-loss, take-profit, trailing stop)

A **stop-loss** is a forced exit when the price moves against you; a
**take-profit** is a forced exit when it moves in your favour; a
**trailing stop** is a stop-loss whose level *ratchets* in the
favourable direction as the position runs (long: tightens up, never
down; short: tightens down, never up). *"Bracket order"* means
attaching one or more at entry. The `Simulator` checks all of them
intra-bar — using the bar's `high` and `low` rather than just the close
— so an exit fires the moment a wick pierces the level, not at
end-of-day.

## Attaching brackets at entry

Put the fractions on the entry `Order` — any combination of the three
is valid:

```python
from fundcloud.sim import Order
import pandas as pd

Order(
    ts=pd.Timestamp("2024-01-02"),
    asset="SPY",
    side="buy",
    qty=10.0,
    sl_stop=0.05,    # 5% below entry → forced sell
    tp_stop=0.10,    # 10% above entry → forced sell
    tsl_stop=0.03,   # 3% trailing stop — anchor ratchets up with bar.high
)
```

When the order fills the simulator records the absolute levels (or, for
trailing, the running anchor + fraction) on the position; subsequent
bars test the bar's range against them. A fill synthesised by the
bracket check appears in the `trades` DataFrame with its `reason`
column set to `"stop_loss"`, `"take_profit"`, or `"trailing_stop"`
(regular fills are tagged `"signal"`).

## Long vs short formulas

| Side | `sl_level` | `tp_level` | `tsl_level` | SL/TSL trips on | TP trips on |
|---|---|---|---|---|---|
| long | `entry × (1 − sl_stop)` | `entry × (1 + tp_stop)` | `tsl_anchor × (1 − tsl_stop)` | `bar.low ≤ level` | `bar.high ≥ tp_level` |
| short | `entry × (1 + sl_stop)` | `entry × (1 − tp_stop)` | `tsl_anchor × (1 + tsl_stop)` | `bar.high ≥ level` | `bar.low ≤ tp_level` |

`sl_stop` and `tsl_stop` must be in `(0, 1)` (a 100%+ stop on a long
would mean "never fires until price goes negative"). `tp_stop` must be
`> 0` and has no upper bound — values `≥ 1` are valid but never fire
on a short (price can't drop more than 100%).

For trailing stops, the **anchor** is the high-water-mark price for
longs (or low-water-mark for shorts): it starts at the entry fill
price and ratchets in the favourable direction each bar (long:
`max(anchor, bar.high)`; short: `min(anchor, bar.low)`). The level
follows the anchor; it can only tighten, never loosen.

## Where the check happens

Per bar:

1. Drain pending fills from prior bars.
2. **Intra-bar bracket check** — fires SL / TP / trailing-stop on
   positions whose level the current bar's range pierces.
3. Strategy `decide()` runs (or, for the `run_orders` / `run_signals` /
   `run_weights` paths, the equivalent built-in submission step).
4. Mark to market at bar close.

So an order placed on bar *t* (which fills at *t+1* under the default
`NextBarOpen` execution) can have its bracket fire as early as bar
*t+2*.

## Gap behaviour

Markets can open past a level overnight. The simulator handles both
sides realistically:

* **SL gap** (open is *worse* than the stop): fill at `bar.open`.
  Selling a long that gapped down to $85 with a stop at $90 fills at
  $85, not $90 — you lost more than the stop's nominal protection.
* **TP gap** (open is *better* than the take-profit): fill at
  `bar.open`. A long with TP at $110 that gapped up to $115 fills at
  $115, not $110 — you got the favourable gap.

```python
# Long entry at 100, sl_stop=0.10 → SL level = 90.
# Bar opens at 85 (gap-down through SL): trade fills at 85.

# Long entry at 100, tp_stop=0.10 → TP level = 110.
# Bar opens at 115 (gap-up through TP): trade fills at 115.
```

## Trailing stop — two-step ratchet around the trigger

The trail's anchor can move *within* a single bar (it ratchets against
the bar's favourable extreme as the price runs). To stay realistic,
the simulator splits the ratchet across the trigger check:

1. **Pre-trigger ratchet** — bump the anchor to `bar.open` if
   favourable (gap-up for long, gap-down for short). On most bars
   this is a no-op; only true gap bars move the anchor here.
2. **Trigger check** — compute the trail level from the post-open
   anchor. If `bar.open` is already past the level (real gap-through),
   fill at the open. Otherwise, if the bar's unfavourable extreme
   reaches the level (`bar.low` for long, `bar.high` for short),
   fill at the level.
3. **Post-trigger ratchet** — only if the trail didn't fire, ratchet
   the anchor against the favourable extreme (`bar.high` for long,
   `bar.low` for short) so the *next* bar sees the new high-water
   mark.

The split matters: ratcheting to the bar's full extreme *before* the
trigger would let a wide-range bar tighten the trail level mid-bar to
a value the bar's open never actually traded against, then fire on
the bar's own low — a phantom exit. Splitting the ratchet keeps the
trigger honest and matches the convention used by mature engines
(vbt's `from_signals(tsl_stop=…)`).

```python
# Long entry at $100, tsl_stop=0.10 → anchor=100, tsl_level=90.
# Bar 2: open=110, high=120, low=110.
#   step 1 — open=110 > anchor=100, ratchet anchor to 110.
#   step 2 — level=110*0.9=99. open(110)>99 and low(110)>99 → no fire.
#   step 3 — ratchet anchor to bar.high=120.
# Bar 3: open=90 — below the level (120*0.9 = 108) in force at start of
#   bar 3 → real gap-down → fill at 90.
```

Trailing stop coexists naturally with the fixed `sl_stop` and
`tp_stop`. When both a fixed SL and the trail could fire on the same
bar, the **tighter fill** wins (long: `max(sl_fill, tsl_fill)`; short:
`min`). The trade `reason` reflects which bound the trigger:
`"stop_loss"` or `"trailing_stop"`.

## Stops beat take-profit when both could fire

A wide-range bar can pierce both a stop (fixed or trailing) and the
take-profit between open and close. The simulator picks the **stop**
— the conservative choice for the trader (assume the worst-case
sequence of intra-bar moves). Between the fixed `sl_stop` and the
trail, the *tighter* fill wins (above). There's no flag to override;
if you need the opposite default, exit by signal instead of bracket.

## Inspecting forced exits

The trades DataFrame's `reason` column lets you split discretionary
fills from bracket-driven ones:

```python
result = Simulator(bars, cash=100_000).run_strategy(strategy)

stops = result.trades[result.trades["reason"] == "stop_loss"]
profits = result.trades[result.trades["reason"] == "take_profit"]
trails = result.trades[result.trades["reason"] == "trailing_stop"]
signal = result.trades[result.trades["reason"] == "signal"]

print(
    f"{len(stops)} forced stops, {len(trails)} trail exits, "
    f"{len(profits)} take-profits, {len(signal)} signal fills"
)
```

## Accumulation

The two bracket families behave differently when a position
accumulates (multiple entries to the same asset before the first
exit closes the position):

* **Fixed `sl_stop` / `tp_stop`** — re-anchor to the **latest fill
  price** on every accumulating entry. Tightens both brackets
  relative to current price as the position grows, which is the
  conservative choice for risk management.
* **Trailing `tsl_stop`** — initialised on the *first* entry that
  carries it, then **retained** across accumulating entries. The
  trailing anchor only moves via the bar-by-bar ratchet (step 3
  above), not by subsequent fills. If you want a fresh trail per
  add, close and re-open instead of accumulating.

Worked example. Suppose three buys at progressively higher prices,
each carrying `sl_stop=0.10, tp_stop=0.10, tsl_stop=0.05`. The
bar-by-bar ratchet has been running between the buys, so the
trailing anchor has moved up alongside the price:

```text
                            sl_level   tp_level   tsl_anchor   tsl_level
Buy 1: 100 @ $50            $45.00     $55.00     $50          $47.50
Buy 2: 100 @ $60 (later)    $54.00     $66.00     $60          $57.00
Buy 3: 100 @ $80 (later)    $72.00     $88.00     $80          $76.00
```

The fixed levels jump on each fill (latest-fill anchor); the trail
anchor moves *only* through the bar-by-bar high-water-mark ratchet,
not because the new buy reset it. If price had pulled back to $55
between Buy 2 and Buy 3, Buy 3 still wouldn't have happened (no
signal in this scenario), but if it had, the trail anchor would
remain at the highest favourable price seen — say $65 — not jump to
$55.

A trade without `sl_stop` / `tp_stop` / `tsl_stop` set (e.g. an
exit-only fill) leaves the existing bracket state alone.

When the position fully closes (`qty == 0`), `sl_level`, `tp_level`,
`tsl_pct` and `tsl_anchor` are all cleared.

## All four entry points support brackets

Brackets work uniformly across the simulator's surface:

* **`run_strategy(strategy)`** — `BaseStrategy` subclasses emit
  `Order(... sl_stop=, tp_stop=, tsl_stop=)`. Pure-Python execution.
* **`run_orders(orders_df)`** — long-format DataFrame with optional
  `sl_stop` / `tp_stop` / `tsl_stop` columns. Dispatches to the Rust
  kernel for speed.
* **`run_signals(entries, exits)`** — boolean panels. The signal API
  doesn't have a per-order bracket surface; if you need brackets here,
  drive the same logic through `run_strategy` with a custom
  `BaseStrategy`.
* **`run_weights(weights_df)`** — same caveat as `run_signals`.

The Rust kernel and the pure-Python fallback are kept in sync by the
`tests/unit/test_sim_parity.py` suite (≈75 cases at `atol=1e-10`).

## Caveats

* **No configurable arbitration.** Stops always beat take-profit; the
  tighter-fill rule between the fixed SL and the trail is hardcoded.
  No flag.
* **No intra-bar timing.** We have only the OHLC summary of each bar,
  not the order in which the high / low were touched. Two bars with
  identical OHLC values produce identical fills regardless of the
  underlying tick sequence. The trailing stop's gap-vs-ratchet rule
  (above) is the carefully-chosen approximation given this constraint.
* **No trailing take-profit.** Symmetric mirror-image of `tsl_stop` —
  not implemented; the only trailing semantics we model is the
  loss-side trail.
* **No borrow cost on shorts.** Shorts can carry brackets like longs,
  but the simulator does not model securities-lending fees or margin
  requirements.

## See also

* [Simulator guide](simulator.md) — the parent guide to all four
  entry points.
* `tests/unit/test_simulator_stops.py`,
  `tests/unit/test_simulator_take_profit.py`, and
  `tests/unit/test_simulator_trailing_stop.py` — exhaustive edge-case
  coverage.
* `tests/unit/test_sim_parity.py` — Rust ↔ fallback parity for
  bracket orders.
