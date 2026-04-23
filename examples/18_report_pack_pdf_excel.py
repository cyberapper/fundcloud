"""18 — Stakeholder report pack: HTML + PDF + Excel from one run.

Trader scenario: you just finished a backtest and need to share results
with three different audiences:

* **Review / chat** — interactive self-contained HTML (plotly, readable
  on any phone or desktop browser).
* **Board deck / archive** — static, printable PDF built via
  matplotlib ``PdfPages``. Pure-Python, no system libraries needed.
* **Analysts** — editable Excel workbook with native XlsxWriter charts
  so numbers remain modifiable in-place.

``Tearsheet`` is one object with three renderers. This example drives
all three off the same ``Portfolio`` and reports the file sizes. The PDF
renderer has an optional ``engine="weasyprint"`` mode for users who want
the richer CSS-driven layout and already have Pango installed on the
host — shown at the bottom.

Run:
    uv add 'fundcloud[reports,viz]'
    uv run python examples/18_report_pack_pdf_excel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from _data import pull_closes
from fundcloud.portfolio import Portfolio
from fundcloud.reports import Tearsheet

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    closes = pull_closes({"SPY": "SPY", "AGG": "AGG"}, years=5)
    if closes is None or closes.empty:
        return 1
    returns = closes.pct_change().dropna()
    weights_now = pd.DataFrame(
        [[0.6, 0.4]] * len(returns),
        index=returns.index,
        columns=returns.columns,
    )
    benchmark = returns["SPY"].rename("SPY")

    # Strategy returns = 60/40 rebalance-every-day stand-in.
    strategy = (returns * weights_now).sum(axis=1).rename("60_40")
    portfolio = Portfolio(returns=strategy, weights=weights_now, benchmark=benchmark, name="60_40")

    tear = Tearsheet(portfolio, benchmark=benchmark, title="60/40 tear sheet — Q1 review")

    html_path = OUT / "18_report.html"
    pdf_path = OUT / "18_report.pdf"
    xlsx_path = OUT / "18_report.xlsx"

    tear.render_html(html_path)
    print(
        f"HTML:   {html_path.relative_to(HERE.parent)}  "
        f"({html_path.stat().st_size / 1024:.1f} KB, plotly inline)"
    )

    try:
        tear.render_pdf(pdf_path)
        print(
            f"PDF:    {pdf_path.relative_to(HERE.parent)}  "
            f"({pdf_path.stat().st_size / 1024:.1f} KB, matplotlib PdfPages)"
        )
    except ImportError as e:
        print(f"PDF:    skipped — {e}")
        print("        (need `uv add 'fundcloud[viz]'`)")

    # Optional: opt in to the WeasyPrint engine for the CSS-styled layout.
    # Only activates when WeasyPrint + Pango are installed on the host.
    weasy_path = OUT / "18_report_weasyprint.pdf"
    try:
        tear.render_pdf(weasy_path, engine="weasyprint")
        print(
            f"PDF*:   {weasy_path.relative_to(HERE.parent)}  "
            f"({weasy_path.stat().st_size / 1024:.1f} KB, WeasyPrint — opt-in)"
        )
    except ImportError as e:
        print(f"PDF*:   WeasyPrint engine unavailable (optional) — {type(e).__name__}")

    try:
        tear.render_excel(xlsx_path)
        print(
            f"Excel:  {xlsx_path.relative_to(HERE.parent)}  "
            f"({xlsx_path.stat().st_size / 1024:.1f} KB, XlsxWriter with native charts)"
        )
    except ImportError as e:
        print(f"Excel:  skipped — {e}")
        print("        (need `uv add 'fundcloud[reports]'`)")

    print("\nHow to read it:")
    print("  * HTML is self-contained (plotly embedded) — paste into Slack or email.")
    print("  * PDF uses matplotlib PdfPages by default — pure-Python, no system")
    print("    libraries required. One stat-table page, one chart per page, with")
    print("    deterministic margins so nothing gets cropped.")
    print("  * engine='weasyprint' is available when you want the richer CSS")
    print("    layout and have Pango installed on the host.")
    print("  * Excel ships real XlsxWriter charts; numbers remain editable, unlike")
    print("    an image-based export, so analysts can tweak and recompute in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
