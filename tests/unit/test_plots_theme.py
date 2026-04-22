"""Tests for :mod:`fundcloud.plots.themes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from fundcloud.plots import plotly as plt_plot
from fundcloud.plots import themes


@pytest.fixture(autouse=True)
def _reset_theme():
    before = themes.get_theme()
    yield
    themes.set_theme(before)


@pytest.fixture
def returns() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=120, freq="B").values)
    return pd.Series(rng.normal(0.0005, 0.01, 120), index=idx, name="strategy")


@pytest.mark.parametrize(
    ("alias", "expected_template"),
    [
        ("default", None),
        ("white", "plotly_white"),
        ("dark", "plotly_dark"),
        ("ggplot2", "ggplot2"),
        ("seaborn", "seaborn"),
    ],
)
def test_alias_resolves_to_plotly_template(alias: str, expected_template: str | None) -> None:
    assert themes._resolve_template(alias) == expected_template


def test_set_and_get_round_trip() -> None:
    themes.set_theme("dark")
    assert themes.get_theme() == "dark"


def test_unknown_theme_raises() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        themes.set_theme("nope-this-is-not-real")


def test_user_registered_template_passes_through(returns: pd.Series) -> None:
    # Register a user template on the fly (mirrors the docstring example).
    pio.templates["fundcloud-test"] = pio.templates["plotly_dark"]
    try:
        themes.set_theme("fundcloud-test")
        assert themes._resolve_template(None) == "fundcloud-test"
        fig = plt_plot.cumulative(returns)
        # The figure picks up the selected template.
        assert fig.layout.template is not None
    finally:
        del pio.templates["fundcloud-test"]


def test_theme_kwarg_overrides_active(returns: pd.Series) -> None:
    themes.set_theme("default")
    fig = plt_plot.cumulative(returns, theme="dark")
    # plotly_dark sets a dark paper background — easier-to-pin signal than
    # comparing the full template object.
    assert fig.layout.template.layout.paper_bgcolor is not None


def test_default_theme_applies_fundcloud_fallback(returns: pd.Series) -> None:
    themes.set_theme("default")
    fig = plt_plot.cumulative(returns)
    # With no template selected, fundcloud still gives the figure its
    # signature white-paper look.
    assert fig.layout.paper_bgcolor == "white"


def test_top_level_reexport_is_same_object() -> None:
    # `import fundcloud as fc; fc.set_theme(...)` must hit the same setter
    # as `fundcloud.plots.set_theme` so users don't get two registries.
    import fundcloud as fc

    assert fc.set_theme is themes.set_theme
    assert fc.get_theme is themes.get_theme
