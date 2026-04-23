"""``Population`` — a comparison container for ``Portfolio`` objects.

Mirrors skfolio's ``Population`` so users' existing skfolio code keeps
working. Plot helpers live under :mod:`fundcloud.plots`; the population
object is responsible only for *deriving* the comparison frames.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd

from fundcloud.portfolio.portfolio import Portfolio

__all__ = ["Population"]


class Population:
    """A named bag of :class:`Portfolio` objects."""

    def __init__(self, portfolios: Sequence[Portfolio]) -> None:
        self._portfolios = list(portfolios)
        # Disambiguate name collisions so `summary` produces unique columns.
        counts: dict[str, int] = {}
        for p in self._portfolios:
            counts[p.name] = counts.get(p.name, 0) + 1
        seen: dict[str, int] = {}
        for p in self._portfolios:
            base = p.name
            if counts[base] > 1:
                seen[base] = seen.get(base, 0) + 1
                p.rename(f"{base}_{seen[base]}")

    # --------------------------------------------------------------- sequence

    def __iter__(self) -> Iterator[Portfolio]:
        return iter(self._portfolios)

    def __len__(self) -> int:
        return len(self._portfolios)

    def __getitem__(self, idx: int | str) -> Portfolio:
        if isinstance(idx, int):
            return self._portfolios[idx]
        for p in self._portfolios:
            if p.name == idx:
                return p
        raise KeyError(idx)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self._portfolios]

    # ----------------------------------------------------------------- views

    def summary(
        self,
        *,
        risk_free: float | None = None,
        periods_per_year: int | None = None,
        cvar_alpha: float = 0.95,
    ) -> pd.DataFrame:
        """Metric-by-portfolio comparison table (rows = metrics, cols = portfolios)."""
        cols = [
            p.summary(
                risk_free=risk_free,
                periods_per_year=periods_per_year,
                cvar_alpha=cvar_alpha,
            )
            for p in self._portfolios
        ]
        if not cols:
            return pd.DataFrame()
        return pd.concat(cols, axis=1)

    def cumulative_returns(self) -> pd.DataFrame:
        """Wide frame of cumulative (compounded) returns per portfolio."""
        if not self._portfolios:
            return pd.DataFrame()
        series = {}
        for p in self._portfolios:
            try:
                r = p.returns
            except ValueError:
                continue
            series[p.name] = (1.0 + r).cumprod()
        return pd.DataFrame(series)

    def composition(self) -> pd.DataFrame:
        """Latest weights per portfolio, as rows-per-portfolio × asset columns."""
        rows: dict[str, pd.Series] = {}
        for p in self._portfolios:
            w = p.weights
            if w is None or len(w) == 0:
                continue
            rows[p.name] = w.iloc[-1]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).T.fillna(0.0)

    # ------------------------------------------------------------------ dunder

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return f"Population(n={len(self._portfolios)}, names={self.names})"
