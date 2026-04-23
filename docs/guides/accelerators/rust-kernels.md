# Rust kernels

Portfolio research is dominated by rolling-window statistics and tail-risk calculations on wide panels. These are exactly the operations where NumPy's per-column Python-level dispatch becomes the bottleneck once you cross a few thousand bars or a few dozen assets. Fundcloud's answer is a small, opinionated Rust core: the tightly-looped numeric kernels live in the `crates/fundcloud-core` crate, are compiled into a PyO3 extension module (`fundcloud._core`), and are surfaced through [`fundcloud.kernels`](../../reference/kernels.md) — which dispatches to Rust when the extension is available and falls back to a pure-Python reference otherwise.

Two design constraints keep the system honest:

1. **Parity is a test, not a claim.** Every kernel has a pure-Python reference and a dedicated parity suite that asserts `atol ≤ 1e-10` across random panels. If the Rust code ever disagrees with NumPy by more than a rounding error, the test fails and the wheel does not ship.
2. **Fallback is always functional, never hidden.** When the Rust extension is missing, the same Python function runs — just slower. Nothing raises, nothing silently returns a stub; `HAS_RUST` and `kernel_version()` tell you exactly which backend is active.

## Which kernels?

- **Returns** — `returns_from_prices` (simple arithmetic return).
- **Rolling** — `rolling_mean`, `rolling_std`, and their 2-D batch variants
  (Rayon-parallel across columns).
- **Drawdown** — `drawdown_series`, `max_drawdown_batch`.
- **Moments** — `sharpe_batch`, `sortino_batch` (annualised, sample std).
- **Tail risk** — `var_batch`, `cvar_batch` (pandas-compatible quantile).

Every kernel:

- NaN-aware. Skips NaN samples in mean / std / quantile computations.
- Releases the GIL via `py.allow_threads` before calling into Rust.
- Uses Rayon to parallelise across columns for batch variants.

## Which backend am I using?

```python
from fundcloud import kernels
kernels.HAS_RUST            # True when the extension loaded
kernels.kernel_version()    # "0.1.0" on Rust; "python-fallback" without
```

## Parity {#parity}

The pure-Python reference lives in `fundcloud.kernels._pyfallback`.
A dedicated parity suite (`tests/unit/test_kernels_parity.py`) compares
the Rust and Python outputs on random panels of 500 × 10 at multiple
parameter settings — `atol = 1e-10` or better across every kernel.

## Benchmarks

Run the harness locally:

```bash
uv run python bench/kernels_bench.py
```

It prints a small table like:

| kernel | n × m | python (ms) | rust (ms) | speedup |
|---|---|---|---|---|
| rolling_mean_batch | 5000 × 20 | 4.8 | 0.9 | ≈ 5× |
| rolling_std_batch  | 5000 × 20 | 7.2 | 1.1 | ≈ 6× |
| sharpe_batch       | 5000 × 20 | 1.4 | 0.12 | ≈ 12× |
| cvar_batch         | 5000 × 20 | 3.9 | 0.20 | ≈ 19× |

Numbers vary by CPU; re-run on your target hardware. A larger "number of
strategies" axis (wider panels) widens the gap because Rayon scales.

## Fallback {#fallback}

If the Rust wheel isn't available for your platform (unusual — we ship
wheels for Linux/macOS/Windows × x86_64/aarch64 via abi3), Fundcloud
transparently falls back to NumPy/Pandas implementations. Correctness is
identical; speed is slower.
