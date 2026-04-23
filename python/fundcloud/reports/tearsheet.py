"""``Tearsheet`` — one object, three output formats.

Built from a :class:`fundcloud.portfolio.Portfolio`. Rendering dispatches to
sibling modules (:mod:`fundcloud.reports.html`, :mod:`.pdf`, :mod:`.excel`).
Heavy dependencies are imported lazily so the default install stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from fundcloud._benchmark import resolve_benchmark
from fundcloud.portfolio import Portfolio

__all__ = ["Tearsheet"]


@dataclass(slots=True)
class Tearsheet:
    """A renderable tear sheet.

    ``benchmark`` accepts a :class:`pandas.Series` or a column name to look
    up on the portfolio's underlying returns (set at construction). The
    lookup runs in ``__post_init__`` so every downstream renderer sees a
    plain ``pd.Series``.
    """

    portfolio: Portfolio
    benchmark: pd.Series | str | None = None
    template: Literal["strategy"] = "strategy"
    title: str | None = None
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.benchmark, str):
            # Try to resolve against the portfolio's returns — for Series
            # the name is the only viable match; for aggregated portfolios
            # there is no named column so we raise a clear message.
            candidate = self.portfolio.returns
            if isinstance(candidate, pd.Series) and candidate.name == self.benchmark:
                self.benchmark = candidate.rename(self.benchmark)
            else:
                self.benchmark = resolve_benchmark(None, self.benchmark)  # will raise

    # ---------------------------------------------------------------- renderers

    def render_html(self, path: str | Path | None = None) -> str | Path:
        """Render to HTML.

        ``path`` is ``None``  → returns the HTML string.
        ``path`` is a path    → writes the file and returns the :class:`Path`.

        Returning the Path (instead of the 5 MB+ inline-plotly string) avoids
        flooding REPL / notebook output when a path is provided.
        """
        from fundcloud.reports import html as _html

        return _html.render(self, path=path)

    def render_pdf(
        self, path: str | Path, *, engine: Literal["matplotlib", "weasyprint"] | None = None
    ) -> Path:
        """Render a PDF.

        ``engine="matplotlib"`` (default) uses pure-Python matplotlib ``PdfPages`` —
        no system libraries required. ``engine="weasyprint"`` uses HTML + CSS
        paged media; needs Pango / GLib installed on the host.
        """
        from fundcloud.reports import pdf as _pdf

        return _pdf.render(self, path=Path(path), engine=engine)

    def render_excel(self, path: str | Path) -> Path:
        from fundcloud.reports import excel as _excel

        return _excel.render(self, path=Path(path))

    # ------------------------------------------------------------------ data

    def stats(self) -> pd.Series:
        return self.portfolio.summary()
