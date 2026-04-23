# Optimize

Portfolio construction is exposed through a thin sklearn-compatible layer: optimisers implement `fit(X)` / `predict(X) -> Portfolio` / `score(X)`, which means they slot into `sklearn.pipeline.Pipeline` and `GridSearchCV` the same way any estimator does. When the `[pf]` extra is installed, skfolio's full family (`MeanRisk`, `HRP`, `HERC`, and friends) is reachable through Fundcloud with `RiskMeasure.VARIANCE`, `CVAR`, `EVAR`, `MDD` and related measures. Without the extra, `EqualWeighted`, `InverseVolatility`, and `MVO` remain available as pure-Python fallbacks — same interface, same estimator checks. See the [interop guide](../guides/interop/sklearn.md) for end-to-end pipelines.

::: fundcloud.optimize
    options:
      members:
        - MVO
        - EqualWeighted
        - InverseVolatility
        - BaseFallbackOptimizer
        - RiskMeasure
