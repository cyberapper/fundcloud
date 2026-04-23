# Metrics

All single-series metrics (`sharpe`, `sortino`, `calmar`, `omega`, `max_drawdown`, `ulcer_index`, `value_at_risk`, `cvar`) take a returns `Series` and a small number of keyword arguments (`periods_per_year`, `risk_free`, `alpha`, `target`) with sensible defaults for daily data. The `batch_*` variants accept a dict of named series or a wide DataFrame and dispatch to the Rust-accelerated kernel when available — see [Rust kernels](../guides/accelerators/rust-kernels.md). `returns_stats` / `batch_summary` produce the same `Series` / `DataFrame` shape used by the tear sheet, so custom reporting code can share the formatting layer.

::: fundcloud.metrics
    options:
      members:
        - sharpe
        - sortino
        - calmar
        - omega
        - drawdown_series
        - max_drawdown
        - ulcer_index
        - cvar
        - value_at_risk
        - returns_stats
        - batch_sharpe
        - batch_sortino
        - batch_max_drawdown
        - batch_cvar
        - batch_summary
