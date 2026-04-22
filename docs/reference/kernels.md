# Kernels

`fundcloud.kernels` is the low-level numeric surface that every user-facing metric dispatches through. Most users will never import it directly — `returns.fc.sharpe()` and `pf.max_drawdown()` already go through the same kernels — but the symbols below are the supported public API for code that wants to call the accelerated primitives explicitly. `HAS_RUST` and `kernel_version()` report which backend is active; see the [Rust kernels guide](../guides/accelerators/rust-kernels.md) for the parity methodology and published benchmarks.

::: fundcloud.kernels
    options:
      members:
        - HAS_RUST
        - kernel_version
        - returns_from_prices
        - rolling_mean
        - rolling_std
        - rolling_mean_batch
        - rolling_std_batch
        - drawdown_series
        - max_drawdown_batch
        - sharpe_batch
        - sortino_batch
        - var_batch
        - cvar_batch
