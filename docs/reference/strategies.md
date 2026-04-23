# Strategies

`fundcloud.strategies` exposes the abstract `BaseStrategy` together with two preset implementations (`Hold`, `DCA`), a per-bar `Context` object, and the `Cadence` / `Scheduler` primitives that govern when a strategy fires. The `@register_strategy` decorator participates in Catalog serialisation so a strategy name round-trips through YAML configs. For usage patterns and the custom-strategy worked example, see the [DCA & Hold guide](../guides/strategies/dca.md).

::: fundcloud.strategies
    options:
      members:
        - BaseStrategy
        - Context
        - Hold
        - RebalanceSpec
        - DCA
        - Scheduler
        - Cadence
        - register_strategy
        - registered_strategies
