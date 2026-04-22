# Portfolio

`Portfolio` is Fundcloud's shared post-simulation object. Every entry path — `Simulator.run_strategy`, `run_weights`, `run_signals`, `run_orders`, and the skfolio round-trip via `from_skfolio` / `to_skfolio` — produces one. Metrics (`sharpe`, `max_drawdown`, `turnover`, `attribution`), the full `summary()` bundle, and the tear-sheet renderers all read from the same object, which keeps notebook exploration and production reporting numerically identical. `Population` holds a set of portfolios for cross-strategy comparison; `Position` is the per-asset record.

::: fundcloud.portfolio
    options:
      members:
        - Portfolio
        - Population
        - Position
