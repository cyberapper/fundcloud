"""Simulator Rust-vs-NumPy-fallback benchmark.

Times the three deterministic entry points on panels of increasing size
and reports the speed-up factor. Run:

    env -u CONDA_PREFIX DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        uv run python bench/bench_sim.py

The script prints a simple CSV-like table; pipe to a file for tracking
regressions across releases.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fundcloud.kernels import HAS_RUST
from fundcloud.kernels import _sim as _dispatcher
from fundcloud.sim import FixedBps, HalfSpread, NextBarOpen, Simulator


def _synthetic_bars(n_bars: int, n_assets: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-02", periods=n_bars, freq="B")
    cols: dict[tuple[str, str], np.ndarray] = {}
    for j in range(n_assets):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n_bars)))
        cols[("open", f"A{j:02d}")] = close
        cols[("high", f"A{j:02d}")] = close * 1.001
        cols[("low", f"A{j:02d}")] = close * 0.999
        cols[("close", f"A{j:02d}")] = close
        cols[("volume", f"A{j:02d}")] = 1_000_000.0
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _time_once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _run_under_backend(fn, *, rust: bool) -> float:
    orig = _dispatcher._have_rust_sim
    _dispatcher._have_rust_sim = lambda: rust
    try:
        # Warm up + 3 repeats; take the median.
        fn()
        samples = sorted(_time_once(fn) for _ in range(3))
        return samples[1]
    finally:
        _dispatcher._have_rust_sim = orig


def _run_weights_case(bars: pd.DataFrame) -> object:
    assets = [f"A{j:02d}" for j in range(bars.columns.get_level_values(1).nunique())]
    target = pd.DataFrame(
        [[1.0 / len(assets)] * len(assets)] * 3,
        index=[bars.index[0], bars.index[len(bars) // 3], bars.index[2 * len(bars) // 3]],
        columns=assets,
    )

    def _call() -> object:
        return Simulator(
            bars,
            cash=1_000_000.0,
            costs=FixedBps(5.0),
            slippage=HalfSpread(spread_bps=3.0),
            execution=NextBarOpen(),
        ).run_weights(target)

    return _call


def main() -> int:
    if not HAS_RUST:
        print("Rust extension not built — run `maturin develop --release` first.")
        return 1
    print("size (bars x assets) | path          | wall (s) | speedup")
    print("-" * 64)
    for nb, na in [(500, 5), (2000, 10), (5000, 20), (10000, 30)]:
        bars = _synthetic_bars(nb, na)
        call = _run_weights_case(bars)
        t_py = _run_under_backend(call, rust=False)
        t_rust = _run_under_backend(call, rust=True)
        speed = t_py / t_rust if t_rust > 0 else float("inf")
        print(
            f"{nb:>5} x {na:<4}          | run_weights    | "
            f"py={t_py:.3f}  rust={t_rust:.3f}  | {speed:5.1f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
