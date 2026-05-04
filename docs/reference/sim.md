# Simulator

The `fundcloud.sim` module is the execution engine: a single `Simulator` class with four entry points (`run_strategy`, `run_weights`, `run_signals`, `run_orders`), plus the small protocol classes that govern costs (`FixedBps`, `PerShare`, `NoCost`), slippage (`HalfSpread`, `NoSlippage`), and fill timing (`NextBarOpen`, `NextBarClose`). All paths return a `SimResult` that carries the post-run `Portfolio`, executed `Trade`s, full `Order` history, and per-bar equity curve. See the [Simulator guide](../guides/sim/simulator.md) for when to pick each entry point.

::: fundcloud.sim
    options:
      members:
        - Simulator
        - SimResult
        - Order
        - OrderSide
        - OrderKind
        - Trade
        - TradeReason
        - CostModel
        - FixedBps
        - PerShare
        - NoCost
        - SlippageModel
        - HalfSpread
        - NoSlippage
        - ExecutionModel
        - NextBarOpen
        - NextBarClose
