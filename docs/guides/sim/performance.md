---
title: Simulator performance (Rust kernel)
description: When the Rust simulator kernel engages, what speedup to expect, and how to stay on the fast path.
---

# Simulator performance

## The fast path in one paragraph

When you call :meth:`Simulator.run_weights`, :meth:`Simulator.run_orders`, or :meth:`Simulator.run_signals` with Fundcloud's built-in cost, slippage, and execution models, the simulator dispatches to a **Rust kernel** that runs the full deterministic bar loop with the GIL released — no Python callback per bar, no pandas method dispatch, all work happens on flat NumPy arrays through a PyO3 boundary. The pure-Python fallback at `fundcloud.kernels._sim_pyfallback` is the parity reference: the two backends agree to `atol=1e-10` across every `(cost × slippage × execution)` combination (`tests/unit/test_sim_parity.py` enforces this).

## Speedup (`bench/bench_sim.py`)

```
size (bars × assets)   path           wall (s)            speedup
─────────────────────────────────────────────────────────────────
    500 ×   5          run_weights    py=0.005  rust=0.003    2.0×
   2000 ×  10          run_weights    py=0.024  rust=0.007    3.4×
   5000 ×  20          run_weights    py=0.105  rust=0.020    5.3×
  10000 ×  30          run_weights    py=0.300  rust=0.049    6.2×
```

Speedup scales with panel size because the Rust loop has a smaller per-bar constant than the NumPy-driven Python fallback. The Python fallback is itself NumPy-array-based (no pandas `.iloc` per bar), which is why the speedup sits around 2–6× rather than the 100× seen when replacing a pandas-row loop directly.

Run the benchmark yourself:

```bash
env -u CONDA_PREFIX DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
    uv run python bench/bench_sim.py
```

## When is the Rust kernel engaged?

**Three entry points** route to Rust:

- `Simulator.run_weights(target_weights)` — rebalance at each row of `target_weights`
- `Simulator.run_orders(orders)` — execute an explicit `ts/asset/side/qty` log
- `Simulator.run_signals(entries, exits)` — boolean entry/exit panels

**Not** `run_strategy` — that path calls a Python `BaseStrategy.decide(ctx)` per bar and can't release the GIL per iteration. It stays on the original Python loop.

Rust dispatch is **gated** on all three of `(costs, slippage, execution)` being built-ins:

| Model | Built-in |
|---|---|
| `costs` | `NoCost`, `FixedBps`, `PerShare` |
| `slippage` | `NoSlippage`, `HalfSpread` |
| `execution` | `NextBarOpen`, `SameBarClose` |

Any custom subclass silently falls back to the original Python `_drive` loop — the simulator stays correct, you just don't get the Rust speedup.

## Verify the fast path is active

```python
from fundcloud.kernels import HAS_RUST, _core
print(HAS_RUST, hasattr(_core, "sim_run_weights"))
# → True True   ← Rust kernel installed
```

If `HAS_RUST` is `False`, your wheel was built without the Rust extension — the simulator still runs correctly on the NumPy fallback. Rebuild the Rust extension with:

```bash
env -u CONDA_PREFIX DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
    uv run maturin develop --release
```

## Architecture

```
Simulator.run_weights ─┬─ built-in models?
                       │      │
                       │      ├─ yes → _run_weights_fast ─┐
                       │      │                           │
                       │      │          fundcloud.kernels._sim dispatcher
                       │      │                           │
                       │      │                           ├─ HAS_RUST → Rust kernel
                       │      │                           │             (crates/fundcloud-core/src/sim.rs)
                       │      │                           │
                       │      │                           └─ fallback  → _sim_pyfallback.py
                       │      │                                          (NumPy loop; parity reference)
                       │      │
                       │      └─ no  → _drive (original Python loop with per-bar callback)
```

The Rust kernel and the NumPy fallback share the exact same loop structure (drain pending → submit new orders → mark-to-market), same enum-tagged config, same struct-of-arrays output. Bugs fixed in one almost always apply to the other by close inspection; `tests/unit/test_sim_parity.py` enforces that they agree before a release ships.
