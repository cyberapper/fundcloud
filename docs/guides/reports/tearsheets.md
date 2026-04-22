# Tear sheets

A Fundcloud tear sheet is a single rendered artefact that a non-technical stakeholder can read end-to-end — stat cards, cumulative return curve, drawdown, rolling Sharpe, a return distribution, and a monthly heatmap — with the exact run parameters embedded in the footer. Three output formats share the same content and formatting layer, so an HTML sent to a partner, a PDF attached to a monthly note, and an Excel workbook used for internal review are always consistent.

!!! tip "Pick the format that matches the recipient"
    HTML is best for interactive exploration and emails (self-contained, no runtime needed). PDF is the lowest-friction archive format (matplotlib default, no system dependencies). Excel is the right choice whenever the recipient will resize charts, change the colour of a series, or pivot the numbers themselves — XlsxWriter emits native charts, not screenshots.

```python
from fundcloud.reports import Tearsheet

ts = Tearsheet(portfolio, title="Demo", benchmark=bench_returns)
ts.render_html("demo.html")        # self-contained plotly — always available
ts.render_pdf("demo.pdf")          # matplotlib PdfPages — needs fundcloud[viz]
ts.render_excel("demo.xlsx")       # XlsxWriter — needs fundcloud[reports]

# Opt-in CSS-styled PDF (needs WeasyPrint + Pango):
ts.render_pdf("demo.pdf", engine="weasyprint")
```

Each render emits a `Path` (except `render_html` which also returns the
HTML string when `path=None`).

## What's inside

The HTML page is a two-column layout — **charts on the left**, **numeric tables (including the categorised metrics sidebar) on the right**:

| Left column — `fc-charts` | Right column — `fc-sidebar` |
| ------------------------- | --------------------------- |
| Cumulative return (%) | Categorised metrics (Return / Risk-adjusted / …) |
| Drawdown | Period performance table |
| Rolling Sharpe | EOY returns table |
| Rolling α / β (if benchmark) | Worst 10 drawdowns |
| Return distribution | Top 10 runups |
| Monthly heatmap | |
| EOY returns (paired bars) | |

Every sidebar section is a collapsible `<details class="fc-group">` accordion — the same styling for metric groups and for the new numeric tables. PDF and Excel render the same content laid out as separate pages / sheets (see below).

### Ingredients

- **Stat cards** — CAGR, Sharpe, max drawdown, CVaR-95 (top of the page).
- **Cumulative return (%)** chart — starts at 0% and is percent-formatted; overlays the benchmark as a dashed reference when supplied.
- **Drawdown**, **Rolling Sharpe** (63-bar), **Return distribution**, **Monthly heatmap** (year × month, when the history spans more than 60 days).
- **EOY returns** paired bar chart (benchmark amber, strategy blue, dashed mean-return reference line).
- **Period performance table** — MTD / 3M / 6M / YTD / 1Y / 3Y (ann.) / 5Y (ann.) / 10Y (ann.) / All-time (ann.). One column per series (benchmark prepended when supplied).
- **EOY returns table** — Year / Benchmark / Strategy.
- **Worst 10 drawdowns table** — Started / Recovered / Drawdown / Days, sorted by depth.
- **Top 10 runups table** — Started / Peaked / Runup / Days, sorted by magnitude.
- **Categorised metrics** — every metric from `fundcloud.metrics.metrics()` grouped by category (Return, Risk-adjusted, Risk, Drawdown, Distribution, Trade, Calendar, and Benchmark when a benchmark is given). The sidebar header reads `METRICS — {strategy} vs {benchmark}` so the two value columns are self-describing. Each row has a `?` badge; hover it for the definition and formula.
- **Benchmark rows** — when `benchmark=` is supplied: the cumulative curve gets a dashed overlay, the metrics sidebar grows a `Benchmark` section (with the benchmark's own values shown in a second column), and a rolling α/β panel joins the chart list. PDFs get a dedicated benchmark page; Excel Summary sheets mirror the categorised layout with a Strategy / Benchmark column pair, and a `Benchmark` sheet carries the aligned return series.

## HTML output

- Plotly runtime is embedded **inline** — the single file is
  self-contained, fine to email or commit.
- Expect ~5 MB per report because of the inline JS; drop to PDF for
  smaller artefacts.

## PDF output

**Default (pure-Python):** :class:`matplotlib.backends.backend_pdf.PdfPages`
— one stat-table page plus one chart per page. No system libraries
required; the charts are the same matplotlib figures used by every other
renderer, so the output is consistent and nothing gets cropped.

**Optional WeasyPrint engine:** pass `engine="weasyprint"` to
`render_pdf` when you want the HTML/CSS paged-media layout. WeasyPrint
needs Pango / GLib / cairo — on macOS that's `brew install pango` plus
`export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`; on Debian/Ubuntu,
`apt install libpango-1.0-0 libpangoft2-1.0-0`. If the WeasyPrint import
fails, Fundcloud raises a clear `ImportError` pointing back to the
matplotlib default.

## Excel output (optional extra)

- Uses [XlsxWriter](https://xlsxwriter.readthedocs.io/) with **native
  charts** — not images — so the numbers remain editable in Excel and
  the charts update live.
- Sheets: `Summary`, `Period Returns`, `Yearly Returns`, `Drawdowns`,
  `Runups`, `Returns`, and `Weights` (when a weights frame is available
  on the portfolio). Percentage columns and embedded cumulative charts
  use the workbook's shared `0.00%` format.

## One-figure alternative — `plots.summary`

If the recipient just wants the charts — no stat cards, no table, no Jinja template — call [`fundcloud.plots.summary`](../plots/summary.md) directly. It returns a single Plotly `Figure` with the same canonical panels (cumulative, drawdown, rolling Sharpe, distribution, monthly heatmap) plus an optional composition row, ready to write to HTML or a PNG:

```python
from fundcloud import plots

plots.summary(portfolio.returns, weights=portfolio.weights).write_html("quick.html")
```

Themes apply to both surfaces. Setting the theme before rendering re-themes the `Tearsheet` HTML too:

```python
import fundcloud as fc

fc.set_theme("dark")
Tearsheet(portfolio).render_html("dark.html")
```

See [Plots → Themes](../plots/themes.md) for the alias map.

## Custom templates

Drop a Jinja2 file at `fundcloud/reports/templates/<name>.html.j2`
(the same place the default `strategy.html.j2` lives), then
`Tearsheet(..., template="<name>")`. The shared
[`fundcloud.reports.formatting`](../../reference/reports.md) helpers
(`stat_cards`, `stats_rows`, `format_stat`) keep HTML and PDF output
consistent.
