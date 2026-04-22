# sklearn & skfolio interop

Every estimator-shaped class in Fundcloud is a **drop-in for both sklearn and skfolio** — no adapters required. The same `FeaturePipeline`, `MeanRisk`, `PurgedKFold`, and `Portfolio` objects you'd use in a notebook are the ones you compose inside a `sklearn.pipeline.Pipeline` or nest under `GridSearchCV`. Fit/predict semantics, parameter-name conventions (`__` separator for nested access), and `get_params` / `set_params` behaviour all match sklearn's `estimator_checks` suite, which means joblib serialisation, grid search, and cross-validated scoring round-trip cleanly.

!!! note "Why this matters for research-to-production"
    The path from a notebook backtest to a scheduled production pipeline is almost always the place where ad-hoc code accumulates. By keeping everything on the sklearn estimator contract, the same object graph survives the move — the notebook code becomes the production code, not a rewrite of it.

## Contracts

| Component | Base class | Implements |
|---|---|---|
| Transformer (`IndicatorSpec`, `FeaturePipeline`) | `sklearn.base.TransformerMixin, BaseEstimator` | `fit(X, y=None)`, `transform(X)` |
| CV splitter (`PurgedKFold`, `EmbargoedKFold`) | `sklearn.model_selection.BaseCrossValidator` | `split`, `get_n_splits` |
| Optimiser (`MeanRisk`, `HRP`, `EqualWeighted`, `MVO`) | skfolio `BaseOptimization` when `[pf]` is installed; sklearn `BaseEstimator` otherwise | `fit(X)`, `predict(X) -> Portfolio`, `score(X)` |

## Nested pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from fundcloud.features import FeaturePipeline
from fundcloud.features.indicators import RSI, SMA
from fundcloud.optimize import MeanRisk, RiskMeasure
from fundcloud.validate import EmbargoedKFold

pipe = Pipeline([
    ("features", FeaturePipeline([
        ("rsi", RSI(timeperiod=14)),
        ("sma", SMA(timeperiod=20)),
    ])),
    ("optim",    MeanRisk(risk_measure=RiskMeasure.CVAR)),
])

search = GridSearchCV(
    pipe,
    param_grid={"optim__min_weights": [0.0, 0.02, 0.05]},
    cv=EmbargoedKFold(n_splits=5, purge=3, embargo=1),
)
search.fit(returns_panel)
```

## skfolio round-trip

```python
from fundcloud.portfolio import Portfolio

# Fundcloud → skfolio
sk_pf = fc_pf.to_skfolio()

# skfolio → Fundcloud
fc_pf = Portfolio.from_skfolio(sk_pf)
```

`Portfolio.from_skfolio` preserves returns + weights (when skfolio exposes
a per-period weights matrix), and `Portfolio.to_skfolio` builds a fresh
`skfolio.Portfolio` with the correct shape.

## Without the `[pf]` extra

If skfolio isn't installed:

- `fundcloud.optimize.EqualWeighted` / `InverseVolatility` resolve to the
  pure-Python fallback class (same `fit`/`predict` contract, long-only
  + fully-invested constraints, `scipy.optimize.minimize`).
- `fundcloud.optimize.MVO` is always available as the pure-Python MVO.
- Every other skfolio optimiser raises `ImportError` with a clear install
  hint when first attribute-accessed.

The fallbacks also pass `sklearn.utils.estimator_checks.check_estimator`,
so `Pipeline` / `GridSearchCV` round-trips are unchanged.
