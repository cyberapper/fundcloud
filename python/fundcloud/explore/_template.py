"""Jinja2 templates for :func:`profile` and :func:`compare` reports.

Kept inline so the package ships without template-file packaging concerns.
Everything renders through a single environment so macros (``format_int``,
``format_pct``) are shared between both reports.
"""

from __future__ import annotations

from jinja2 import Environment, select_autoescape

__all__ = ["COMPARE_TEMPLATE", "PROFILE_TEMPLATE", "env"]


def _format_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _format_float(value: object, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.{digits}g}"


def _format_pct(value: object, digits: int = 1) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.{digits}f}%"


def _format_bytes(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024:
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


env = Environment(autoescape=select_autoescape(["html", "xml"]))
env.filters["fmt_int"] = _format_int
env.filters["fmt_float"] = _format_float
env.filters["fmt_pct"] = _format_pct
env.filters["fmt_bytes"] = _format_bytes


_BASE_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: #0f172a;
  background: #f8fafc;
  margin: 0;
  padding: 32px 48px 64px;
  line-height: 1.5;
}
h1 { font-size: 26px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 32px 0 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h3 { font-size: 14px; margin: 16px 0 4px; color: #334155; }
.muted { color: #64748b; font-size: 13px; }
table.overview, table.stats {
  border-collapse: collapse;
  width: 100%;
  background: #ffffff;
  font-size: 13px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}
table.overview td, table.overview th, table.stats td, table.stats th {
  padding: 6px 10px;
  border-bottom: 1px solid #e2e8f0;
  text-align: right;
}
table.overview th:first-child, table.overview td:first-child,
table.stats th:first-child, table.stats td:first-child { text-align: left; }
table.stats thead th {
  background: #f1f5f9;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #475569;
}
/* Histogram / overlay stack — one chart per row, full-width. Scales to
   an arbitrary number of assets: each histogram keeps its horizontal
   resolution intact, and the reader just scrolls. A grid-based layout
   squeezed wide histograms into narrow columns, losing bin detail. */
.grid { display: flex; flex-direction: column; gap: 12px; }
.grid > details { padding: 4px 8px 8px; }
.grid .plotly-graph-div { width: 100%; }
details { background: #ffffff; border-radius: 6px; padding: 8px 12px; border: 1px solid #e2e8f0; }
details + details { margin-top: 8px; }
details[open] > summary { margin-bottom: 4px; }
summary { cursor: pointer; font-weight: 600; font-size: 13px; color: #334155; padding: 2px 0; }
/* Stack missing-value charts vertically so the bar chart's x-axis
   labels line up directly above the heatmap's columns — same order, same
   widths. */
.missing-stack { display: flex; flex-direction: column; gap: 12px; }
.missing-stack h3 { margin: 4px 0 2px; }
.missing-stack .plotly-graph-div { width: 100%; }
.alert-list { list-style: none; padding: 0; margin: 0; }
.alert-list li {
  padding: 8px 12px; margin-bottom: 6px; border-radius: 4px;
  background: #ffffff; border-left: 4px solid #cbd5e1;
  font-size: 13px;
}
.alert-critical { border-left-color: #dc2626; background: #fef2f2; }
.alert-warning { border-left-color: #f59e0b; background: #fffbeb; }
.alert-info { border-left-color: #3b82f6; background: #eff6ff; }
.tabs { display: flex; gap: 8px; margin: 8px 0; }
.tab-button {
  padding: 6px 12px; border: 1px solid #cbd5e1; background: #ffffff;
  cursor: pointer; border-radius: 4px; font-size: 12px;
}
.tab-button.active { background: #0f172a; color: #ffffff; border-color: #0f172a; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.footer { margin-top: 48px; font-size: 11px; color: #94a3b8; text-align: center; }
""".strip()


_TABS_JS = """
(function () {
  function resizePlots(container) {
    // Hidden plotly divs render at width 0; when the panel becomes visible
    // we need to tell plotly to recompute dimensions. Without this the
    // Spearman heatmap (or any tab that wasn't active at first paint)
    // shows up narrower than its Pearson neighbour.
    if (typeof Plotly === 'undefined') { return; }
    container.querySelectorAll('.plotly-graph-div').forEach((d) => {
      try { Plotly.Plots.resize(d); } catch (_e) { /* fig not ready yet */ }
    });
  }

  document.querySelectorAll('.tabs').forEach((tabs) => {
    const group = tabs.dataset.group;
    const buttons = tabs.querySelectorAll('.tab-button');
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        buttons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const target = btn.dataset.tab;
        document.querySelectorAll(`.tab-panel[data-group="${group}"]`).forEach((p) => {
          const nowActive = p.dataset.tab === target;
          p.classList.toggle('active', nowActive);
          if (nowActive) { resizePlots(p); }
        });
      });
    });
  });

  // Also handle <details> opening — plotly figures inside a collapsed
  // <details> element render at width 0 and stay that way when expanded
  // unless we explicitly resize them.
  document.querySelectorAll('details').forEach((d) => {
    d.addEventListener('toggle', () => {
      if (d.open) { resizePlots(d); }
    });
  });
})();
""".strip()


_HEAD = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<style>{{ css | safe }}</style>
{# Plotly must load BEFORE any inline Plotly.newPlot(...) calls in the #}
{# body; placing it in <head> guarantees that. If we defer it to the #}
{# footer, every histogram / heatmap div renders empty because Plotly #}
{# is still undefined when the figures try to self-register. #}
{% if plotlyjs_block %}{{ plotlyjs_block | safe }}{% endif %}
</head>
<body>
<h1>{{ title }}</h1>
<p class="muted">{{ subtitle }}</p>
"""


_FOOTER = """
<div class="footer">fundcloud.explore — generated {{ generated_at }}</div>
<script>{{ tabs_js | safe }}</script>
</body>
</html>
"""


PROFILE_TEMPLATE = (
    _HEAD
    + """

<h2>Overview</h2>
<table class="overview">
<tr><th>Rows</th><td>{{ overview.rows | fmt_int }}</td>
    <th>Columns</th><td>{{ overview.cols | fmt_int }}</td></tr>
<tr><th>Cells</th><td>{{ overview.cells | fmt_int }}</td>
    <th>Missing cells</th><td>{{ overview.missing_cells | fmt_int }}
        ({{ overview.missing_cells_pct | fmt_pct }})</td></tr>
<tr><th>Duplicate rows</th><td>{{ overview.duplicate_rows | fmt_int }}</td>
    <th>Memory</th><td>{{ overview.memory_bytes | fmt_bytes }}</td></tr>
<tr><th>Index</th><td>{{ overview.index_type }}</td>
    <th>Dtypes</th><td>{% for k, v in overview.dtypes.items() %}{{ k }}: {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}</td></tr>
{% if overview.date_start %}
<tr><th>Date range</th><td colspan="3">{{ overview.date_start }} → {{ overview.date_end }}</td></tr>
{% endif %}
</table>

<h2>Per-column stats</h2>
{{ stats_table_html | safe }}

{% if histograms_html %}
<h2>Histograms</h2>
<div class="grid">
{% for name, div in histograms_html %}
<details open><summary>{{ name }}</summary>{{ div | safe }}</details>
{% endfor %}
</div>
{% endif %}

{% if correlation_html %}
<h2>Correlations</h2>
<div class="tabs" data-group="corr">
<button class="tab-button active" data-tab="pearson">Pearson</button>
<button class="tab-button" data-tab="spearman">Spearman</button>
</div>
<div class="tab-panel active" data-group="corr" data-tab="pearson">{{ correlation_html.pearson | safe }}</div>
<div class="tab-panel" data-group="corr" data-tab="spearman">{{ correlation_html.spearman | safe }}</div>
{% endif %}

{% if missing_html %}
<h2>Missing-value patterns</h2>
<div class="missing-stack">
<h3>Missing values per column</h3>
{{ missing_html.bar | safe }}
{% if missing_html.heatmap %}
<h3>Missingness map</h3>
<p class="muted">{{ missing_html.caption }}</p>
{{ missing_html.heatmap | safe }}
{% endif %}
</div>
{% endif %}

<h2>Alerts</h2>
{% if alerts %}
<ul class="alert-list">
{% for alert in alerts %}
<li class="alert-{{ alert.severity }}">
<strong>{{ alert.code }}</strong> — {{ alert.message }}
</li>
{% endfor %}
</ul>
{% else %}
<p class="muted">No alerts fired.</p>
{% endif %}
"""
    + _FOOTER
)


COMPARE_TEMPLATE = (
    _HEAD
    + """

<h2>Overview</h2>
<table class="overview">
<tr><th></th><th>{{ names[0] }}</th><th>{{ names[1] }}</th></tr>
<tr><th>Rows</th><td>{{ overview.a.rows | fmt_int }}</td><td>{{ overview.b.rows | fmt_int }}</td></tr>
<tr><th>Columns</th><td>{{ overview.a.cols | fmt_int }}</td><td>{{ overview.b.cols | fmt_int }}</td></tr>
<tr><th>Missing cells</th><td>{{ overview.a.missing_cells | fmt_int }}</td><td>{{ overview.b.missing_cells | fmt_int }}</td></tr>
<tr><th>Memory</th><td>{{ overview.a.memory_bytes | fmt_bytes }}</td><td>{{ overview.b.memory_bytes | fmt_bytes }}</td></tr>
{% if overview.a.date_start %}
<tr><th>Date range</th><td>{{ overview.a.date_start }} → {{ overview.a.date_end }}</td><td>{{ overview.b.date_start }} → {{ overview.b.date_end }}</td></tr>
{% endif %}
</table>
<p class="muted">Shared columns: {{ shared | length }}. Unique to {{ names[0] }}: {{ only_a | length }}. Unique to {{ names[1] }}: {{ only_b | length }}.</p>

<h2>Per-column drift</h2>
{% if drift_table_html %}
{{ drift_table_html | safe }}
{% else %}
<p class="muted">No numeric columns in common — nothing to compare.</p>
{% endif %}

{% if overlay_histograms_html %}
<h2>Distribution overlay</h2>
<div class="grid">
{% for name, div in overlay_histograms_html %}
<details open><summary>{{ name }}</summary>{{ div | safe }}</details>
{% endfor %}
</div>
{% endif %}

{% if correlation_delta_html %}
<h2>Correlation delta</h2>
{{ correlation_delta_html | safe }}
{% endif %}

{% if target_shift_html %}
<h2>Target-correlation shifts</h2>
<p class="muted">Correlation of each feature with the target, both datasets; bigger |Δ| means the feature's signal against the target has moved.</p>
{{ target_shift_html | safe }}
{% endif %}

<h2>Alerts</h2>
{% if alerts %}
<ul class="alert-list">
{% for alert in alerts %}
<li class="alert-{{ alert.severity }}">
<strong>{{ alert.code }}</strong> — {{ alert.message }}
</li>
{% endfor %}
</ul>
{% else %}
<p class="muted">No alerts fired.</p>
{% endif %}
"""
    + _FOOTER
)
