"""Multi-asset tear-sheet smoke tests for :mod:`fundcloud.reports.multi`.

The single-asset path is covered separately by :mod:`test_reports_html`,
:mod:`test_reports_pdf`, and :mod:`test_reports_excel`. This module
exercises the multi-asset orchestrator: HTML / PDF / Excel rendering of
several named portfolios in one report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.reports import multi as _multi


def _portfolio(seed: int, *, periods: int, name: str, start: str = "2022-01-03") -> Portfolio:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.date_range(start, periods=periods, freq="B").values)
    r = pd.Series(rng.normal(0.0005, 0.01, periods), index=idx, name=name)
    return Portfolio(returns=r, name=name)


@pytest.fixture
def long_portfolios() -> list[tuple[str, Portfolio]]:
    """Two portfolios with > 60 days span — triggers the monthly heatmap path."""
    return [
        ("alpha", _portfolio(1, periods=300, name="alpha")),
        ("beta", _portfolio(2, periods=300, name="beta")),
    ]


@pytest.fixture
def short_portfolio() -> list[tuple[str, Portfolio]]:
    """One portfolio with < 10 rows — skips the monthly heatmap path."""
    return [("tiny", _portfolio(3, periods=8, name="tiny"))]


@pytest.fixture
def benchmark() -> pd.Series:
    rng = np.random.default_rng(99)
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=300, freq="B").values)
    return pd.Series(rng.normal(0.0003, 0.008, 300), index=idx, name="bench")


# --------------------------------------------------------------------- HTML


def test_render_html_returns_string_when_no_path(
    long_portfolios: list[tuple[str, Portfolio]],
) -> None:
    html = _multi.render_html(long_portfolios, title="Demo multi")
    assert isinstance(html, str)
    assert "<html" in html
    assert "Demo multi" in html
    # Both asset names appear (tabs).
    assert "alpha" in html
    assert "beta" in html


def test_render_html_writes_file_when_path_given(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    out = tmp_path / "multi.html"
    returned = _multi.render_html(long_portfolios, path=out)
    assert returned == out
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Plotly runtime is inlined exactly once on the first figure.
    assert "Plotly" in text
    # Default report title.
    assert "Fundcloud tear sheet" in text


def test_render_html_with_benchmark_includes_rolling_alpha_beta(
    long_portfolios: list[tuple[str, Portfolio]], benchmark: pd.Series, tmp_path: Path
) -> None:
    out = tmp_path / "multi_bench.html"
    _multi.render_html(long_portfolios, benchmark=benchmark, path=out)
    html = out.read_text(encoding="utf-8")
    # The benchmark label drives a rolling alpha/beta panel per section.
    assert "bench" in html


def test_render_html_with_unnamed_benchmark(
    long_portfolios: list[tuple[str, Portfolio]],
) -> None:
    """Benchmark without a `.name` falls back to the literal "benchmark"."""
    rng = np.random.default_rng(7)
    bench = pd.Series(
        rng.normal(0.0, 0.005, 300),
        index=pd.DatetimeIndex(pd.date_range("2022-01-03", periods=300, freq="B").values),
    )
    html = _multi.render_html(long_portfolios, benchmark=bench)
    assert isinstance(html, str)


def test_render_html_skips_monthly_heatmap_for_short_series(
    short_portfolio: list[tuple[str, Portfolio]],
) -> None:
    """`_has_span_of_months` returns False → no monthly heatmap is rendered."""
    html = _multi.render_html(short_portfolio)
    # The single-tab single-asset rendering still produces an HTML doc.
    assert "<html" in html
    assert "tiny" in html


def test_render_html_empty_portfolios_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _multi.render_html([])


# --------------------------------------------------------------------- PDF


def test_render_pdf_writes_valid_pdf(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    out = tmp_path / "multi.pdf"
    returned = _multi.render_pdf(long_portfolios, path=out, title="Multi PDF")
    assert returned == out
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    # Cover page + per-asset metric tables + chart pages → comfortably > 30 KB.
    assert len(data) > 30_000


def test_render_pdf_with_benchmark_adds_rolling_pages(
    long_portfolios: list[tuple[str, Portfolio]], benchmark: pd.Series, tmp_path: Path
) -> None:
    out = tmp_path / "multi_bench.pdf"
    _multi.render_pdf(long_portfolios, path=out, title="With bench", benchmark=benchmark)
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")


def test_render_pdf_default_title(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    """No `title=` → cover page uses the default 'Multi-asset tear sheet'."""
    out = tmp_path / "default.pdf"
    _multi.render_pdf(long_portfolios, path=out)
    assert out.exists()


def test_render_pdf_short_series_skips_monthly(
    short_portfolio: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    """A < 10-period series exercises the no-monthly-heatmap branch."""
    out = tmp_path / "short.pdf"
    _multi.render_pdf(short_portfolio, path=out)
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-")


def test_render_pdf_rejects_non_matplotlib_engine(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    with pytest.raises(NotImplementedError, match="matplotlib"):
        _multi.render_pdf(long_portfolios, path=tmp_path / "bad.pdf", engine="weasyprint")


# --------------------------------------------------------------------- Excel


def test_render_excel_writes_workbook_with_overview_and_per_asset_sheets(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    out = tmp_path / "multi.xlsx"
    returned = _multi.render_excel(long_portfolios, path=out, title="Multi Excel")
    assert returned == out
    assert out.exists()

    xls = pd.ExcelFile(out)
    sheets = set(xls.sheet_names)
    assert "Overview" in sheets
    # Per-asset Summary + Returns pair for each portfolio.
    assert "alpha_Summary" in sheets
    assert "alpha_Returns" in sheets
    assert "beta_Summary" in sheets
    assert "beta_Returns" in sheets

    # Returns sheet must have rows for every period.
    returns = pd.read_excel(out, sheet_name="alpha_Returns")
    assert len(returns) > 0


def test_render_excel_with_benchmark_overlays_columns(
    long_portfolios: list[tuple[str, Portfolio]], benchmark: pd.Series, tmp_path: Path
) -> None:
    out = tmp_path / "multi_bench.xlsx"
    _multi.render_excel(long_portfolios, path=out, benchmark=benchmark)
    assert out.exists()
    xls = pd.ExcelFile(out)
    assert "alpha_Summary" in xls.sheet_names


def test_render_excel_default_title(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    """Default title (no `title=`) still produces a usable workbook."""
    out = tmp_path / "default.xlsx"
    _multi.render_excel(long_portfolios, path=out)
    assert out.exists()


def test_render_excel_creates_parent_dir(
    long_portfolios: list[tuple[str, Portfolio]], tmp_path: Path
) -> None:
    """Path with a nonexistent parent directory is created on demand."""
    out = tmp_path / "nested" / "deep" / "multi.xlsx"
    _multi.render_excel(long_portfolios, path=out)
    assert out.exists()


# --------------------------------------------------------------------- helpers


def test_has_span_of_months_true_for_long_series() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=300, freq="B").values)
    s = pd.Series(np.zeros(300), index=idx)
    assert _multi._has_span_of_months(s) is True


def test_has_span_of_months_false_for_short_series() -> None:
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=5, freq="B").values)
    s = pd.Series(np.zeros(5), index=idx)
    assert _multi._has_span_of_months(s) is False


def test_has_span_of_months_false_for_narrow_window() -> None:
    """≥ 10 rows but < 60 days span → False."""
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=15, freq="B").values)
    s = pd.Series(np.zeros(15), index=idx)
    assert _multi._has_span_of_months(s) is False


def test_safe_sheet_name_strips_excel_invalid_chars() -> None:
    # Excel disallows []:/*?\ in sheet names.
    assert _multi._safe_sheet_name("a[b]c:d/e\\f*g?h") == "abcdefgh"


def test_safe_sheet_name_truncates_to_31_chars_with_suffix() -> None:
    out = _multi._safe_sheet_name("X" * 50, suffix="_Summary")
    assert len(out) == 31
    assert out.endswith("_Summary")


def test_fmt_pct_handles_valid_and_bad_values() -> None:
    assert _multi._fmt_pct(0.0123) == "+1.23%"
    assert _multi._fmt_pct(-0.05) == "-5.00%"
    assert _multi._fmt_pct(None) == "—"
    assert _multi._fmt_pct("not-a-number") == "—"


def test_fmt_num_handles_valid_and_bad_values() -> None:
    assert _multi._fmt_num(1.5) == "+1.50"
    assert _multi._fmt_num(-0.25) == "-0.25"
    assert _multi._fmt_num(None) == "—"
    assert _multi._fmt_num(float("nan")).startswith(("+nan", "-nan", "nan"))  # platform variant
