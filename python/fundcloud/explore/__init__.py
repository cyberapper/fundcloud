"""Exploratory data analysis — native plotly + jinja2, no external deps.

* :func:`describe` returns a one-row-per-column summary frame — a
  super-set of :meth:`pandas.DataFrame.describe` with finance extras.
* :func:`profile` renders a full profile — overview, per-column stats,
  histograms, Pearson/Spearman correlation, missing-value patterns, and
  rule-based alerts.
* :func:`compare` renders a two-dataset drift report — side-by-side
  overview, per-column KS + Wasserstein drift, overlay histograms,
  correlation delta, and alerts (including target-correlation shift when
  a ``target=`` column is supplied).

All three ship in the core install; no extras required. ``quickview`` is
a deprecated alias for ``describe`` — pending removal.
"""

from __future__ import annotations

from fundcloud.explore._report import ProfileReport
from fundcloud.explore.compare import compare
from fundcloud.explore.describe import describe
from fundcloud.explore.profile import profile
from fundcloud.explore.quickview import quickview

__all__ = ["ProfileReport", "compare", "describe", "profile", "quickview"]
