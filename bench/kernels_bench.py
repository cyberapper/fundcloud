"""Micro-benchmarks for Fundcloud numerical kernels.

Usage::

    uv run python bench/kernels_bench.py

Prints a small table comparing the active Fundcloud kernel (Rust when
available) against the NumPy/pandas fallback across a handful of panel
sizes. Numbers vary by CPU; the relative speedup is what matters.

The script deliberately avoids pytest-style collection — we don't want the
benchmark to run on every unit-test invocation — and writes an optional
CSV for the release notes.
"""

from __future__ import annotations

import argparse
import csv
import sys
import timeit
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fundcloud import kernels
from fundcloud.kernels import _pyfallback

__all__ = ["Benchmark", "main"]


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One row of the benchmark table."""

    kernel: str
    n: int
    m: int
    python_ms: float
    rust_ms: float

    @property
    def speedup(self) -> float:
        return self.python_ms / self.rust_ms if self.rust_ms else float("inf")


# -------------------------------------------------------------------- harness


def _time(fn, *, repeat: int = 5, number: int = 1) -> float:
    """Wall time of the fastest of ``repeat`` invocations, in milliseconds."""
    best = min(timeit.repeat(fn, repeat=repeat, number=number)) / number
    return best * 1_000.0


def _sized_panel(n: int, m: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=(n, m)).astype(float)


def _sized_vector(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, size=n).astype(float)


# ---------------------------------------------------------------------- runs


def run_rolling_mean_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    window = 30
    return Benchmark(
        kernel="rolling_mean_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.rolling_mean_batch(x, window)),
        rust_ms=_time(lambda: kernels.rolling_mean_batch(x, window)),
    )


def run_rolling_std_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    window = 30
    return Benchmark(
        kernel="rolling_std_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.rolling_std_batch(x, window, 1)),
        rust_ms=_time(lambda: kernels.rolling_std_batch(x, window, 1)),
    )


def run_max_drawdown_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    return Benchmark(
        kernel="max_drawdown_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.max_drawdown_batch(x)),
        rust_ms=_time(lambda: kernels.max_drawdown_batch(x)),
    )


def run_sharpe_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    return Benchmark(
        kernel="sharpe_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.sharpe_batch(x, 0.0, 252.0)),
        rust_ms=_time(lambda: kernels.sharpe_batch(x, 0.0, 252.0)),
    )


def run_sortino_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    return Benchmark(
        kernel="sortino_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.sortino_batch(x, 0.0, 252.0)),
        rust_ms=_time(lambda: kernels.sortino_batch(x, 0.0, 252.0)),
    )


def run_cvar_batch(n: int, m: int) -> Benchmark:
    x = _sized_panel(n, m)
    return Benchmark(
        kernel="cvar_batch",
        n=n,
        m=m,
        python_ms=_time(lambda: _pyfallback.cvar_batch(x, 0.95)),
        rust_ms=_time(lambda: kernels.cvar_batch(x, 0.95)),
    )


_BENCH_FUNS = [
    run_rolling_mean_batch,
    run_rolling_std_batch,
    run_max_drawdown_batch,
    run_sharpe_batch,
    run_sortino_batch,
    run_cvar_batch,
]


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["2000x10", "5000x20", "10000x50"],
        help="Panel shapes as NxM (e.g. 5000x20).",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args(argv)

    print(f"backend: {'rust' if kernels.HAS_RUST else 'python-fallback'}")
    print(f"version: {kernels.kernel_version()}")
    print()

    rows: list[Benchmark] = []
    for size in args.sizes:
        try:
            n_str, m_str = size.lower().split("x")
            n, m = int(n_str), int(m_str)
        except ValueError:
            print(f"skipping invalid size {size!r}", file=sys.stderr)
            continue
        for fn in _BENCH_FUNS:
            rows.append(fn(n, m))

    _print_table(rows)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["kernel", "n", "m", "python_ms", "rust_ms", "speedup"])
            for row in rows:
                writer.writerow([
                    row.kernel,
                    row.n,
                    row.m,
                    f"{row.python_ms:.3f}",
                    f"{row.rust_ms:.3f}",
                    f"{row.speedup:.2f}",
                ])
        print(f"\nwrote {args.csv}")
    return 0


def _print_table(rows: list[Benchmark]) -> None:
    print(
        f"{'kernel':<22} {'n':>7} {'m':>5} {'python (ms)':>13} " f"{'rust (ms)':>11} {'speedup':>9}"
    )
    print("-" * 72)
    for row in rows:
        print(
            f"{row.kernel:<22} {row.n:>7} {row.m:>5} "
            f"{row.python_ms:>13.2f} {row.rust_ms:>11.2f} "
            f"{row.speedup:>8.1f}x"
        )


if __name__ == "__main__":
    raise SystemExit(main())
