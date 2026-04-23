---
title: Using Python's help()
description: Every public Fundcloud function, class, and accessor method ships a NumPy-style docstring with Parameters / Returns / Examples. Use help() at the REPL or ? in Jupyter to read it.
---

# Using Python's `help()` at the REPL

You don't need to leave your terminal to discover Fundcloud's API. Every public class, function, and accessor method ships a NumPy-style docstring with **Parameters**, **Returns**, and **Examples** blocks — so the Python introspection tools you already know light up.

## The one-liner

```python
from fundcloud.metrics import metrics
help(metrics)
```

You get the full docstring: what `metrics(...)` computes, every parameter with its type, what the return value contains, and a runnable example.

## Deep-diving a method

```python
from fundcloud.sim import Simulator
help(Simulator.run_strategy)
help(Simulator.run_weights)
help(Simulator.run_signals)
help(Simulator.run_orders)
```

Same for the `.fc` accessor methods:

```python
import pandas as pd
import fundcloud  # registers the .fc accessor

help(pd.DataFrame.fc.run_dca)     # preset wrapper for DCA
help(pd.DataFrame.fc.render_pdf)  # tear-sheet PDF
help(pd.Series.fc.metrics)        # ~55-metric bundle
```

## Jupyter shortcut

In a notebook, a single `?` gives you the docstring; `??` shows the source:

```python
returns.fc.metrics?
fundcloud.metrics.probabilistic_sharpe??
```

## Discovering what's available

### Tab-completion at the REPL

```python
import fundcloud.data as fcd
dir(fcd)
# → ['AV', 'Backend', 'BaseBackend', 'Binance', 'CSV', 'Catalog',
#    'DatasetSpec', 'DuckDB', 'FMP', 'Memory', 'Parquet',
#    'ReadOnlyError', 'WriteMode', 'YF', ...]
```

The network-backed backends (`YF`, `FMP`, `AV`, `Binance`) are lazy-imported, but `__dir__` surfaces them anyway — so tab-completion "just works" in IPython, Jupyter, and modern editors.

The same pattern applies to `fundcloud.sim`, `fundcloud.optimize`, `fundcloud.validate`, and — most striking — `fundcloud.features.indicators`, where all 158 auto-wrapped TA-Lib indicators are tab-completable:

```python
from fundcloud.features import indicators
len([x for x in dir(indicators) if x.isupper()])
# → 158
```

### Listing every TA-Lib indicator

```python
from fundcloud.features.indicators import list_indicators, GROUPS

list_indicators()[:5]
# → ['ACOS', 'AD', 'ADD', 'ADOSC', 'ADX']

GROUPS["Momentum Indicators"][:3]
# → ['ADX', 'ADXR', 'APO']
```

### The full one-shot metrics bundle

```python
import pandas as pd
import fundcloud                                          # .fc accessor
returns = pd.read_csv("my_returns.csv", index_col="date",
                      parse_dates=True)["return"]
m = returns.fc.metrics()                                  # pd.Series of ~55 metrics
m.head()
```

`help(returns.fc.metrics)` shows what each row means.

## Where to go from `help()`

* **Metrics catalogue** — [Portfolio metrics](portfolio/metrics.md).
* **Simulator entry points** — [Simulator](sim/simulator.md).
* **Accessor index** — each `.fc.*` method links back to its free function in [API reference](../reference/metrics.md).

The convention: if a method shows up in `dir(fundcloud.something)`, it has a docstring worth reading.
