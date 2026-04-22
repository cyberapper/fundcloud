---
title: Themes
description: Switching Plotly templates via the fundcloud alias map or any registered plotly.io template.
---

# Themes

Theming in `fundcloud.plots` is a thin alias layer over Plotly's builtin templates. There is no custom theme object — if Plotly already ships the mechanism, fundcloud uses it directly.

## Alias map

| fundcloud alias | plotly template         |
| --------------- | ----------------------- |
| `default`       | *(no template applied)* |
| `white`         | `plotly_white`          |
| `dark`          | `plotly_dark`           |
| `ggplot2`       | `ggplot2`               |
| `seaborn`       | `seaborn`               |

```python
import fundcloud as fc

fc.set_theme("dark")                    # global default for subsequent figures
fc.plots.cumulative(returns).show()

# Per-figure override wins:
fc.plots.cumulative(returns, theme="white").show()
```

`set_theme` is re-exported at the top level (same object as `fundcloud.plots.set_theme`) so the common `import fundcloud as fc` idiom stays terse. Call `fc.get_theme()` to read the active alias (default: `"default"`).

## Using any registered Plotly template

`set_theme` accepts any name in `plotly.io.templates`, so user-defined templates work without a fundcloud-specific API:

```python
import plotly.io as pio
import plotly.graph_objects as go

# Your brand template
pio.templates["my-brand"] = go.layout.Template(
    layout={
        "colorway": ["#003a70", "#e6a817", "#6ab04c"],
        "paper_bgcolor": "#fafafa",
        "font": {"family": "Söhne, system-ui"},
    }
)

fc.set_theme("my-brand")
fc.plots.summary(returns).write_html("branded.html")
```

An unknown name raises `ValueError` listing every template available in the current process.

## HTML / PDF tear sheets pick up the active theme

`Tearsheet.render_html` builds its Plotly figures through these same builders — so switching theme before calling `render_html` rethemes the report:

```python
import fundcloud as fc
from fundcloud.reports import Tearsheet

fc.set_theme("dark")
Tearsheet(portfolio).render_html("dark.html")
```

`render_pdf` (matplotlib) is not theme-aware; matplotlib theming is not supported today and is explicitly [out of scope](#why-plotly-only).

## Why Plotly-only?

Fundcloud is plotly-first; matplotlib exists because WeasyPrint and PdfPages need it. Adding a second theme system would double the surface area for a secondary output path. If dark-mode PDFs become important later we'll add an `mpl_rc` alias map with the same five names — same alias, different backend.
