"""HTML tear-sheet smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet


@pytest.fixture
def portfolio() -> Portfolio:
    rng = np.random.default_rng(1)
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=400, freq="B").values)
    r = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx, name="demo")
    return Portfolio(returns=r, name="demo")


def test_render_html_writes_file(portfolio: Portfolio, tmp_path: Path) -> None:
    out = tmp_path / "demo.html"
    Tearsheet(portfolio, title="Demo").render_html(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Cumulative returns" in text
    assert "Drawdown" in text
    assert "Rolling Sharpe" in text
    # Plotly runtime is embedded inline (the first chart uses include_plotlyjs="inline").
    assert "Plotly" in text


def test_render_html_returns_string_when_no_path(portfolio: Portfolio) -> None:
    html = Tearsheet(portfolio).render_html()
    assert "<html" in html
    assert "Tear sheet" in html


def test_render_html_with_benchmark(portfolio: Portfolio, tmp_path: Path) -> None:
    bench = portfolio.returns * 0.4
    bench = bench.rename("benchmark")
    Tearsheet(portfolio, benchmark=bench).render_html(tmp_path / "demo.html")
    assert (tmp_path / "demo.html").exists()


def test_render_html_includes_new_performance_sections(
    portfolio: Portfolio, tmp_path: Path
) -> None:
    """Period performance, EOY bars+table, worst drawdowns & runups should render."""
    bench = portfolio.returns.rename("benchmark") * 0.4
    out = tmp_path / "demo.html"
    Tearsheet(portfolio, benchmark=bench).render_html(out)
    html = out.read_text(encoding="utf-8")
    # New section headings.
    assert "Period performance" in html
    assert "EOY returns" in html
    assert "Worst 10 drawdowns" in html
    assert "Top 10 runups" in html
    # Period rows.
    assert "MTD" in html
    assert "All-time (ann.)" in html
    # Table cells have % formatting.
    assert "%</td>" in html


def test_render_html_places_tables_in_sidebar_and_chart_on_left(
    portfolio: Portfolio, tmp_path: Path
) -> None:
    """Charts stay in the fc-charts column (left); the four numeric tables —
    period performance, yearly returns, worst drawdowns, top runups —
    render inside the fc-sidebar as collapsible ``fc-group`` accordions."""
    bench = portfolio.returns.rename("benchmark") * 0.4
    out = tmp_path / "demo.html"
    Tearsheet(portfolio, benchmark=bench).render_html(out)
    html = out.read_text(encoding="utf-8")
    # Slice the document between the two layout columns.
    charts_start = html.index('<div class="fc-charts">')
    sidebar_start = html.index('<aside class="fc-sidebar">')
    charts = html[charts_start:sidebar_start]
    sidebar = html[sidebar_start:]
    # Tables land in the sidebar.
    for marker in ("Period performance", "Worst 10 drawdowns", "Top 10 runups"):
        assert marker in sidebar
        assert marker not in charts
    assert "fc-sidebar-table" in sidebar
    # The EOY chart remains on the left (plotly title comes from the builder).
    assert "EOY Returns vs Benchmark" in charts


def test_render_html_cumulative_chart_uses_percent(portfolio: Portfolio, tmp_path: Path) -> None:
    out = tmp_path / "demo.html"
    Tearsheet(portfolio).render_html(out)
    html = out.read_text(encoding="utf-8")
    # Plotly tickformat ``.0%`` is inlined as a JSON attribute on the figure.
    assert '"tickformat":".0%"' in html
