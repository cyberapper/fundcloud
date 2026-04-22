"""The :class:`ProfileReport` object returned by :func:`profile`.

Designed for a Python-first REPL workflow: ``report = profile(df)`` gives
you a rich object you can interrogate directly (``report.stats``,
``report.alerts``, ``report.correlations["pearson"]``) and render to HTML
only when you want to share.

Examples
--------
>>> import pandas as pd
>>> import numpy as np
>>> from fundcloud.explore import profile
>>> df = pd.DataFrame({"a": np.linspace(-0.01, 0.01, 100),
...                    "b": np.full(100, 0.0002)})
>>> report = profile(df)
>>> report.stats["mean"]["a"]      # doctest: +SKIP
>>> report.alerts                  # doctest: +SKIP
>>> report.to_html("out.html")     # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from fundcloud.explore._alerts import Alert

__all__ = ["ProfileReport"]


@dataclass
class ProfileReport:
    """Output of :func:`fundcloud.explore.profile` — a Python-first bundle.

    Attributes
    ----------
    overview
        Top-level summary (rows, cols, memory, duplicate rows, date range).
    stats
        Per-column descriptive statistics (same as :func:`describe`).
    correlations
        ``{"pearson": DataFrame, "spearman": DataFrame}`` when there are
        at least two numeric columns; empty dict otherwise.
    missing
        ``Series`` of missing-value counts per column.
    alerts
        Rule-based alerts (zero-variance, high correlation, skew /
        kurtosis, excessive missing, etc.).
    title
        Report title used by :meth:`to_html`.
    """

    overview: dict[str, Any]
    stats: pd.DataFrame
    correlations: dict[str, pd.DataFrame]
    missing: pd.Series
    alerts: list[Alert]
    title: str = "Fundcloud data profile"
    # Non-owning plumbing — kept private to avoid leaking implementation.
    _html_builder: Any = field(default=None, repr=False)

    # ----------------------------------------------------------- text / HTML

    def __repr__(self) -> str:
        lines = [f"ProfileReport({self.title!r})"]
        lines.append(
            f"  shape      : {self.overview.get('rows', '?')} rows, "
            f"{self.overview.get('cols', '?')} cols"
        )
        mem = self.overview.get("memory_bytes")
        if mem is not None:
            lines.append(f"  memory     : {_format_bytes(mem)}")
        dr_start = self.overview.get("date_start")
        dr_end = self.overview.get("date_end")
        if dr_start and dr_end:
            lines.append(f"  date range : {dr_start} → {dr_end}")
        lines.append(f"  missing    : {int(self.missing.sum())}")
        lines.append(f"  alerts     : {len(self.alerts)}")
        if self.alerts:
            for a in self.alerts[:5]:
                lines.append(f"    · [{a.severity}] {a.code}: {a.message}")
            if len(self.alerts) > 5:
                lines.append(f"    · ... and {len(self.alerts) - 5} more")
        lines.append("  call .stats / .correlations / .alerts / .to_html(...) for more.")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        # Keep the Jupyter inline preview small — show the stats table + a
        # compact alert summary. Full report via to_html().
        alert_html = ""
        if self.alerts:
            alert_html = "<ul style='margin-top:12px;'>" + "".join(
                f"<li><strong>{a.severity}</strong> · {a.code}: {a.message}</li>"
                for a in self.alerts[:10]
            ) + "</ul>"
        head = (
            f"<h4 style='margin:0 0 6px;'>{self.title}</h4>"
            f"<p style='color:#64748b;margin:0 0 10px;font-size:12px;'>"
            f"{self.overview.get('rows', '?')} rows, "
            f"{self.overview.get('cols', '?')} cols · "
            f"{len(self.alerts)} alerts · call "
            f"<code>.to_html('out.html')</code> for the full report.</p>"
        )
        return head + self.stats.to_html() + alert_html

    def to_html(
        self,
        path: str | Path | None = None,
        *,
        embed_plotlyjs: bool = False,
    ) -> Path | str:
        """Render the full HTML profile.

        When ``path`` is given, write to disk and return the :class:`Path`.
        Otherwise, return the HTML string.
        """
        if self._html_builder is None:
            msg = (
                "This ProfileReport was created without an HTML builder. "
                "Rebuild via fundcloud.explore.profile(df) to enable to_html()."
            )
            raise RuntimeError(msg)
        html = self._html_builder(embed_plotlyjs=embed_plotlyjs)
        if path is None:
            return html
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
        return p

    # ----------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (tables → ``records`` lists)."""
        return {
            "title": self.title,
            "overview": self.overview,
            "stats": self.stats.reset_index().to_dict(orient="records"),
            "correlations": {
                k: v.reset_index().to_dict(orient="records")
                for k, v in self.correlations.items()
            },
            "missing": self.missing.to_dict(),
            "alerts": [
                {
                    "severity": a.severity,
                    "code": a.code,
                    "message": a.message,
                    "columns": list(a.columns),
                }
                for a in self.alerts
            ],
        }


def _format_bytes(value: float) -> str:
    v = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024:
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"
