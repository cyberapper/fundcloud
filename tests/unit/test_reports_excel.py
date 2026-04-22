"""Excel tear-sheet smoke test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def portfolio() -> Portfolio:
    rng = np.random.default_rng(3)
    idx = pd.DatetimeIndex(pd.date_range("2023-01-02", periods=120, freq="B").values)
    r = pd.Series(rng.normal(0.0005, 0.01, 120), index=idx, name="demo")
    w = pd.DataFrame({"A": [0.6] * 120, "B": [0.4] * 120}, index=idx)
    return Portfolio(returns=r, weights=w, name="demo")


def test_render_excel_produces_workbook(portfolio: Portfolio, tmp_path: Path) -> None:
    out = tmp_path / "demo.xlsx"
    Tearsheet(portfolio, title="Excel demo").render_excel(out)
    assert out.exists()
    wb = openpyxl.load_workbook(out, read_only=True)
    names = set(wb.sheetnames)
    assert {
        "Summary",
        "Period Returns",
        "Yearly Returns",
        "Drawdowns",
        "Runups",
        "Returns",
        "Weights",
    }.issubset(names)


def test_render_excel_period_returns_has_pct_format(portfolio: Portfolio, tmp_path: Path) -> None:
    bench = portfolio.returns.rename("SPY") * 0.5
    out = tmp_path / "demo.xlsx"
    Tearsheet(portfolio, benchmark=bench).render_excel(out)
    wb = openpyxl.load_workbook(out, read_only=False)
    sheet = wb["Period Returns"]
    # Row 1 = header, row 2 starts MTD. Column A = label, B = first data column.
    header_row = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    assert "Period" in header_row
    # The first data cell (B2 = MTD benchmark return) should have pct number format.
    cell = sheet.cell(row=2, column=2)
    assert cell.number_format.endswith("%")
