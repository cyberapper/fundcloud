"""26 — Plotly themes for fundcloud figures.

Five fundcloud aliases pass straight through to Plotly builtin templates:

    default → (no template override)
    white   → plotly_white
    dark    → plotly_dark
    ggplot2 → ggplot2
    seaborn → seaborn

Any name registered in ``plotly.io.templates`` is also accepted — useful
when you want to plug in your own brand template.

Run:
    uv run python examples/26_plots_theming.py
"""

from __future__ import annotations

from pathlib import Path

import fundcloud as fc
import plotly.graph_objects as go
import plotly.io as pio
from _synth import AssetProfile, close_returns, generate_ohlcv

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bars = generate_ohlcv(
        {"SPY": AssetProfile(mu=0.08, sigma=0.17, price0=450.0)},
        periods=504,
        seed=5,
    )
    returns = close_returns(bars)["SPY"].rename("SPY")

    # 1. One file per fundcloud alias — use the top-level shortcut.
    for alias in ("default", "white", "dark", "ggplot2", "seaborn"):
        fc.set_theme(alias)
        fig = fc.plots.cumulative(returns, annotations=True, title=f"alias: {alias}")
        out = OUT / f"26_theme_{alias}.html"
        fig.write_html(out)
        print(f"Wrote {out.relative_to(HERE.parent)}")

    # 2. A user-registered plotly template — works without any fundcloud API
    pio.templates["fundcloud-example-brand"] = go.layout.Template(
        layout={
            "colorway": ["#2ea8e5", "#ff8a3c", "#30b77a", "#ae60ff"],
            "paper_bgcolor": "#fbfaf5",
            "plot_bgcolor": "#fbfaf5",
            "font": {"family": "JetBrains Mono, ui-monospace, monospace"},
            "xaxis": {"gridcolor": "rgba(0,0,0,0.06)"},
            "yaxis": {"gridcolor": "rgba(0,0,0,0.06)"},
        }
    )
    fc.set_theme("fundcloud-example-brand")
    fig = fc.plots.summary(returns, title="user template: fundcloud-example-brand")
    out = OUT / "26_theme_brand.html"
    fig.write_html(out)
    print(f"Wrote {out.relative_to(HERE.parent)}")

    # Reset for good citizenship.
    fc.set_theme("default")
    del pio.templates["fundcloud-example-brand"]


if __name__ == "__main__":
    main()
