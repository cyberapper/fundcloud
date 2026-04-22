"""PDF tear-sheet tests — matplotlib default + opt-in WeasyPrint.

The matplotlib backend is pure-Python and always runs. The WeasyPrint
backend is only exercised when its native libraries (Pango / GLib / cairo)
are discoverable; otherwise the WeasyPrint-specific test is skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet


def _weasyprint_loads() -> bool:
    try:
        import weasyprint

        # WeasyPrint only fails at dlopen time when you build HTML objects,
        # so force that to exercise the native-library path.
        weasyprint.HTML(string="<html><body/></html>").render()
        return True
    except Exception:
        return False


skip_without_weasyprint = pytest.mark.skipif(
    not _weasyprint_loads(),
    reason="WeasyPrint native libs (Pango/glib/cairo) not discoverable",
)


@pytest.fixture
def portfolio() -> Portfolio:
    rng = np.random.default_rng(2)
    idx = pd.DatetimeIndex(pd.date_range("2022-01-03", periods=250, freq="B").values)
    r = pd.Series(rng.normal(0.0005, 0.01, 250), index=idx, name="demo")
    return Portfolio(returns=r, name="demo")


def test_render_pdf_matplotlib_default(portfolio: Portfolio, tmp_path: Path) -> None:
    """The matplotlib PdfPages backend is the default and has no system deps."""
    out = tmp_path / "demo.pdf"
    returned = Tearsheet(portfolio, title="PDF demo").render_pdf(out)
    assert returned == out
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-"), "output is not a valid PDF"
    # Multiple pages → comfortably > 20 KB even on a tiny portfolio.
    assert len(data) > 20_000


def test_render_pdf_explicit_matplotlib(portfolio: Portfolio, tmp_path: Path) -> None:
    out = tmp_path / "explicit.pdf"
    Tearsheet(portfolio, title="Explicit mpl").render_pdf(out, engine="matplotlib")
    assert out.read_bytes().startswith(b"%PDF-")


def test_render_pdf_rejects_unknown_engine(portfolio: Portfolio, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown PDF engine"):
        Tearsheet(portfolio).render_pdf(tmp_path / "bad.pdf", engine="chromium")  # type: ignore[arg-type]


@skip_without_weasyprint
def test_render_pdf_weasyprint_opt_in(portfolio: Portfolio, tmp_path: Path) -> None:
    out = tmp_path / "weasy.pdf"
    Tearsheet(portfolio, title="WP demo").render_pdf(out, engine="weasyprint")
    assert out.exists()
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    assert len(data) > 5_000
