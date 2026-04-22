"""Tests for :mod:`fundcloud.explore`.

Covers both the unchanged :func:`quickview` layer and the native rewrites
of :func:`profile` / :func:`compare`. The rewrites depend only on the core
install (plotly + jinja2 + scipy) — no ``ydata-profiling`` or ``sweetviz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.explore import compare, describe, profile, quickview


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "y": rng.normal(5, 2, 100),
        "cat": ["a", "b"] * 50,
    })


# ---------------------------------------------------------------- describe


def test_describe_includes_pandas_rows(df: pd.DataFrame) -> None:
    out = describe(df)
    # pandas-compatible columns (numeric only)
    for col in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
        assert col in out.columns
    # Fundcloud extras
    for col in ("dtype", "missing", "unique", "skew", "kurtosis", "zeros_pct", "inf_pct"):
        assert col in out.columns


def test_describe_finance_rows_on_datetime_index() -> None:
    idx = pd.date_range("2024-01-02", periods=252, freq="B")
    df = pd.DataFrame({"a": np.linspace(-0.01, 0.01, 252)}, index=idx)
    out = describe(df)
    for col in ("sharpe", "cagr", "volatility", "max_drawdown"):
        assert col in out.columns
    assert np.isfinite(out.loc["a", "sharpe"])


def test_describe_honours_percentiles() -> None:
    df = pd.DataFrame({"a": np.arange(100)})
    out = describe(df, percentiles=[0.1, 0.5, 0.9])
    assert "10%" in out.columns
    assert "50%" in out.columns
    assert "90%" in out.columns


def test_describe_writes_html(df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "describe.html"
    describe(df, output=out, title="Describe demo")
    assert out.exists()
    assert "Describe demo" in out.read_text()


def test_quickview_is_deprecated_alias(df: pd.DataFrame) -> None:
    with pytest.warns(DeprecationWarning, match="describe"):
        quickview(df)


# ---------------------------------------------------------------- quickview legacy


def test_quickview_writes_html(df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "qv.html"
    with pytest.warns(DeprecationWarning, match="describe"):
        quickview(df, output=out, title="Quick")
    assert out.exists()
    text = out.read_text()
    assert "Quick" in text


# ------------------------------------------------------------------- profile


_PROFILE_SECTIONS = (
    "Overview",
    "Per-column stats",
    "Histograms",
    "Correlations",
    "Alerts",
)


def test_profile_returns_profile_report(df: pd.DataFrame) -> None:
    """Python-first: profile() returns a ProfileReport the caller can poke at."""
    from fundcloud.explore import ProfileReport

    report = profile(df, title="profile test")
    assert isinstance(report, ProfileReport)
    # Stats table mirrors describe-style output
    assert not report.stats.empty
    assert set(report.stats.index) == {"x", "y", "cat"}
    # REPL-friendly __repr__ stays short
    text = repr(report)
    assert "ProfileReport" in text
    assert "rows" in text


def test_profile_writes_html(df: pd.DataFrame, tmp_path: Path) -> None:
    """profile(df, output=...) still writes the file AND returns the report."""
    out = tmp_path / "profile.html"
    report = profile(df, output=out, title="profile test")
    assert out.exists()
    assert out.stat().st_size > 10_000
    # Report is still returned
    assert report.title == "profile test"
    # Full HTML has all sections
    text = out.read_text()
    for section in _PROFILE_SECTIONS:
        assert f">{section}</h2>" in text, f"missing section: {section}"


def test_profile_alerts_flag_known_patterns() -> None:
    rng = np.random.default_rng(0)
    n = 200
    bad = pd.DataFrame({
        "signal": rng.normal(size=n),
        "const": np.ones(n),
        "mostly_missing": [1.0 if i < 10 else np.nan for i in range(n)],
    })
    bad["dupe"] = bad["signal"]
    report = profile(bad)
    codes = {a.code for a in report.alerts}
    assert "zero_variance" in codes
    assert "high_missing" in codes
    assert "high_correlation" in codes


def test_profile_handles_datetime_index(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    frame = pd.DataFrame({"r": np.linspace(-0.01, 0.01, 50)}, index=idx)
    out = tmp_path / "dt.html"
    profile(frame, output=out)
    text = out.read_text()
    assert "Date range" in text
    assert "2020-01-01" in text


def test_profile_is_dependency_free(monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame) -> None:
    """Hide ydata_profiling / sweetviz from sys.modules; profile() still works."""
    monkeypatch.setitem(sys.modules, "ydata_profiling", None)
    monkeypatch.setitem(sys.modules, "sweetviz", None)
    report = profile(df)
    html = report.to_html()
    assert isinstance(html, str)
    assert "<h1>Fundcloud data profile</h1>" in html


def test_profile_report_to_dict_is_json_friendly(df: pd.DataFrame) -> None:
    import json

    report = profile(df)
    d = report.to_dict()
    # Round-trip through JSON to prove it's serialisable.
    round_tripped = json.loads(json.dumps(d, default=str))
    assert {"title", "overview", "stats", "correlations", "missing", "alerts"}.issubset(
        round_tripped.keys()
    )


def test_profile_html_loads_plotlyjs_before_plot_divs(tmp_path: Path) -> None:
    """Regression: plotly.js must load before inline Plotly.newPlot calls.

    If the script tag is deferred to <body> end, every histogram and
    heatmap div renders empty because the inline scripts run before
    window.Plotly is defined.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    out = tmp_path / "profile.html"
    profile(frame, output=out)
    html = out.read_text()
    plotly_pos = html.find("cdn.plot.ly")
    # Match the markup, not the CSS selector — the CSS contains
    # .plotly-graph-div { ... } and would produce a false earlier hit.
    first_plot_div = html.find('class="plotly-graph-div"')
    head_end = html.find("</head>")
    assert plotly_pos != -1, "plotly.js CDN tag missing"
    assert plotly_pos < head_end, "plotly.js CDN tag must be inside <head>"
    assert plotly_pos < first_plot_div, "plotly.js must load before any plot div"
    # Each figure generates one Plotly.newPlot call — profile() on a 3-col
    # numeric frame ships 3 histograms + 2 correlation heatmaps = 5.
    assert html.count("Plotly.newPlot") == 5


def test_profile_histograms_default_open_and_have_no_duplicate_titles(
    tmp_path: Path,
) -> None:
    """Regression: histograms render as ``<details open>`` so they're
    visible on scroll (important for >10 assets), and the figure's own
    title is suppressed because the ``<summary>`` already shows the
    asset name."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    out = tmp_path / "hist.html"
    profile(frame, output=out)
    html = out.read_text()
    # The Histograms section must expand every asset by default.
    hist_section_start = html.find("<h2>Histograms</h2>")
    assert hist_section_start > 0
    assert html.count("<details open>") >= 4, (
        "histograms should render expanded by default"
    )


def test_profile_histograms_stack_vertically(tmp_path: Path) -> None:
    """Regression: histograms must stack one per row so the layout
    scales to an arbitrary number of assets. A grid-based layout would
    squeeze wide histograms into narrow columns and hurt bin detail."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    out = tmp_path / "stack.html"
    profile(frame, output=out)
    html = out.read_text()
    assert ".grid { display: flex; flex-direction: column" in html, (
        "histograms should use a flex-column stack, not a responsive grid"
    )


def test_profile_missing_value_panel_single_asset_uses_timeline(tmp_path: Path) -> None:
    """Regression: for a 1-column frame, the missingness panel should NOT
    render as a 1-column heatmap (which looks like a bar). It should use
    a row-axis timeline instead."""
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    vals = np.where(np.arange(200) % 7 == 0, np.nan, 1.0)  # weekly gaps
    frame = pd.DataFrame({"only": vals}, index=idx)
    out = tmp_path / "single.html"
    profile(frame, output=out)
    html = out.read_text()
    # The h3 label advertises a timeline, not the multi-column map.
    assert "timeline" in html.lower()


def test_profile_missing_value_panel_aligned_ordering(tmp_path: Path) -> None:
    """Regression: bar chart and heatmap use the same column order so
    column N on the bar matches column N on the heatmap below."""
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "lots_missing": [np.nan if i % 2 == 0 else 1.0 for i in range(50)],
            "no_missing": rng.normal(size=50),
            "some_missing": [np.nan if i % 7 == 0 else 1.0 for i in range(50)],
        },
        index=idx,
    )
    out = tmp_path / "aligned.html"
    profile(frame, output=out)
    html = out.read_text()
    # Both charts include the same caption hint about column ordering.
    assert "column order matches bar chart above" in html


# ------------------------------------------------------------------- compare


_COMPARE_SECTIONS = (
    "Overview",
    "Per-column drift",
    "Distribution overlay",
    "Correlation delta",
    "Alerts",
)


def test_compare_writes_html(df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "cmp.html"
    path = compare(df.iloc[:50], df.iloc[50:], output=out, names=("first", "second"))
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 10_000
    text = out.read_text()
    for section in _COMPARE_SECTIONS:
        assert f">{section}</h2>" in text, f"missing section: {section}"


def test_compare_drift_alerts(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 500
    a = pd.DataFrame({"x": rng.normal(0, 1, n)})
    b = pd.DataFrame({"x": rng.normal(3, 1, n)})  # mean shift of 3σ
    out = tmp_path / "drift.html"
    compare(a, b, output=out)
    text = out.read_text()
    assert "distribution_shift" in text
    assert "mean_shift" in text


def test_compare_missing_column_alerts(df: pd.DataFrame, tmp_path: Path) -> None:
    a = df.copy()
    b = df.drop(columns=["cat"])
    b = b.assign(extra=1.0)
    out = tmp_path / "schema.html"
    compare(a, b, output=out)
    text = out.read_text()
    assert "column_removed" in text
    assert "column_added" in text


def test_compare_target_correlation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 400
    feature = rng.normal(size=n)
    # In "a", target tracks feature; in "b" the signal is wrecked.
    a = pd.DataFrame({"feature": feature, "target": feature * 1.1 + rng.normal(0, 0.1, n)})
    b = pd.DataFrame({"feature": feature, "target": rng.normal(0, 1, n)})
    out = tmp_path / "target.html"
    compare(a, b, output=out, target="target")
    text = out.read_text()
    assert "Target-correlation shifts" in text
    assert "target_correlation_shift" in text
