# Reports

`Tearsheet` is the single report object — one class, three renderers (`render_html`, `render_pdf`, `render_excel`) — fed a `Portfolio` and an optional benchmark. The `fundcloud.reports.formatting` module exposes the shared building blocks (`StatCard`, `StatRow`, `stat_cards`, `stats_rows`, `format_stat`) so custom templates and bespoke reports can reuse the exact formatting the default tear sheet uses. See the [Tear sheets guide](../guides/reports/tearsheets.md) for output structure and the PDF/Excel extras.

::: fundcloud.reports
    options:
      members:
        - Tearsheet

::: fundcloud.reports.formatting
    options:
      members:
        - StatCard
        - StatRow
        - stat_cards
        - stats_rows
        - format_stat
