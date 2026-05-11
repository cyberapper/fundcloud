"""Pattern-detection plot builders.

Plotly candlestick charts that render chart-pattern events with their
pivots connected into the formation shape, trend lines, and the
holding-period horizon overlaid. Useful for eyeballing detections and
for sharing screenshots in research notes.

Three entry points:

* :func:`plot_pattern_event` — single detection, deeply annotated. The
  formation shape is drawn by connecting pivots in chronological order;
  vertical markers show ``breakout_ts`` and (when ``horizon`` is set)
  ``breakout_ts + horizon`` so you can see exactly where the
  feature-quality metric grades the trade.
* :func:`plot_patterns_overview` — every detection of *one* pattern on
  one asset, with formation shapes drawn. Best for spotting clustering.
* :func:`plot_asset_patterns` — every detection of *every* pattern on
  one asset, on a single chart with one legend group per pattern. Click
  a legend entry to toggle a pattern's traces on / off. This is the
  view you want when researching "what's been happening on AAPL?".

All three produce ``plotly.graph_objects.Figure`` instances using the
same theme machinery as the rest of :mod:`fundcloud.plots`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from fundcloud.plots.plotly import _style

__all__ = ["plot_asset_patterns", "plot_pattern_event", "plot_patterns_overview"]

# Marker / line aesthetics. Kept inline (rather than in a theme file)
# since these are pattern-specific and shouldn't drift with theme swaps.
_PIVOT_HIGH_COLOR = "#d62728"  # red
_PIVOT_LOW_COLOR = "#2ca02c"  # green
_NECKLINE_COLOR = "#1f77b4"  # blue
_TARGET_COLOR = "#2ca02c"
_STOP_COLOR = "#d62728"
_ENTRY_COLOR = "#1f77b4"
_FORMATION_FILL = "rgba(31, 119, 180, 0.06)"  # transparent blue band

# Distinct colour per pattern — used for the formation polyline so each
# detection is individually identifiable on the multi-pattern overview.
# Drawn from Plotly's D3 qualitative palette.
_PATTERN_COLORS: dict[str, str] = {
    "head_and_shoulders": "#d62728",  # red
    "inverse_head_and_shoulders": "#2ca02c",  # green
    "double_top": "#ff7f0e",  # orange
    "double_bottom": "#17becf",  # teal
    "triple_top": "#9467bd",  # purple
    "triple_bottom": "#1f77b4",  # blue
    "ascending_triangle": "#bcbd22",  # olive
    "descending_triangle": "#8c564b",  # brown
    "symmetrical_triangle": "#e377c2",  # pink
}

# Horizon-window shading: green-tinted for bullish, red-tinted for bearish.
_HORIZON_FILL_BULLISH = "rgba(44, 160, 44, 0.05)"
_HORIZON_FILL_BEARISH = "rgba(214, 39, 40, 0.05)"
_HORIZON_LINE = "rgba(0, 0, 0, 0.4)"


def _select_asset_ohlc(bars: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Pull OHLC for one asset out of a MultiIndex bars frame."""
    if not isinstance(bars.columns, pd.MultiIndex):
        msg = "bars must have MultiIndex (field, asset) columns"
        raise TypeError(msg)
    sub = bars.xs(asset, level=-1, axis=1)
    return sub.dropna(subset=["open", "high", "low", "close"])


def _slice_window(
    ohlc: pd.DataFrame,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
    *,
    padding: int,
) -> pd.DataFrame:
    """Return the OHLC slice from ``padding`` bars before formation_start
    through ``padding`` bars after formation_end. Bar-padded, not
    calendar-padded — consistent with how detectors think about windows.
    """
    idx = ohlc.index
    try:
        start_pos = idx.get_loc(formation_start)
        end_pos = idx.get_loc(formation_end)
    except KeyError:
        # Fallback: slice between timestamps directly. Less precise on
        # gappy series but never crashes.
        return ohlc.loc[
            formation_start - pd.Timedelta(days=padding) : formation_end
            + pd.Timedelta(days=padding)
        ]
    if isinstance(start_pos, slice):
        start_pos = start_pos.start
    if isinstance(end_pos, slice):
        end_pos = end_pos.stop - 1
    lo = max(0, int(start_pos) - padding)
    hi = min(len(idx) - 1, int(end_pos) + padding)
    return ohlc.iloc[lo : hi + 1]


def _add_candlestick(fig: go.Figure, ohlc: pd.DataFrame, asset: str) -> None:
    fig.add_trace(
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
            name=asset,
            showlegend=False,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )


def _add_pivots(fig: go.Figure, pivots: list[dict[str, Any]]) -> None:
    """Render each pivot as a triangle marker — pointing down for HIGH
    (visually sits above the bar high) and up for LOW (below the low).
    """
    if not pivots:
        return
    high_x = [p["ts"] for p in pivots if p.get("kind") == "HIGH"]
    high_y = [p["price"] for p in pivots if p.get("kind") == "HIGH"]
    low_x = [p["ts"] for p in pivots if p.get("kind") == "LOW"]
    low_y = [p["price"] for p in pivots if p.get("kind") == "LOW"]
    if high_x:
        fig.add_trace(
            go.Scatter(
                x=high_x,
                y=high_y,
                mode="markers",
                name="High pivot",
                marker={
                    "symbol": "triangle-down",
                    "size": 12,
                    "color": _PIVOT_HIGH_COLOR,
                    "line": {"width": 1, "color": "white"},
                },
            )
        )
    if low_x:
        fig.add_trace(
            go.Scatter(
                x=low_x,
                y=low_y,
                mode="markers",
                name="Low pivot",
                marker={
                    "symbol": "triangle-up",
                    "size": 12,
                    "color": _PIVOT_LOW_COLOR,
                    "line": {"width": 1, "color": "white"},
                },
            )
        )


def _add_trend_lines(
    fig: go.Figure,
    trend_lines: list[dict[str, Any]],
    ohlc_window: pd.DataFrame,
    bars_full: pd.DataFrame,
    asset: str,
) -> None:
    """Project each Rust trend line (slope/intercept on the *full* bars
    index) onto the slice we're plotting. ``trend_lines`` are dicts as
    emitted by the PyO3 layer.
    """
    if not trend_lines:
        return
    full_close = _select_asset_ohlc(bars_full, asset).index
    for tl in trend_lines:
        slope = float(tl.get("slope", 0.0))
        intercept = float(tl.get("intercept", 0.0))
        start_index = int(tl.get("start_index", 0))
        end_index = int(tl.get("end_index", 0))
        # Bar-position → timestamp; clip to the slice we're plotting.
        try:
            start_ts = full_close[start_index]
            end_ts = full_close[end_index]
        except IndexError:
            continue
        # Restrict drawing to the visible window (prevents off-screen
        # trace bbox from collapsing the y-axis).
        plot_start = max(start_ts, ohlc_window.index[0])
        plot_end = min(end_ts, ohlc_window.index[-1])
        if plot_start > plot_end:
            continue
        # Map back to bar positions for the price formula.
        try:
            ps_pos = full_close.get_loc(plot_start)
            pe_pos = full_close.get_loc(plot_end)
        except KeyError:
            continue
        if isinstance(ps_pos, slice):
            ps_pos = ps_pos.start
        if isinstance(pe_pos, slice):
            pe_pos = pe_pos.stop - 1
        x_vals = [plot_start, plot_end]
        y_vals = [slope * ps_pos + intercept, slope * pe_pos + intercept]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line={"color": _NECKLINE_COLOR, "width": 2, "dash": "dot"},
                name="Trend line",
                showlegend=False,
            )
        )


def _add_levels(fig: go.Figure, event: dict[str, Any], x_range: tuple[Any, Any]) -> None:
    """Horizontal lines for entry / target / stop. Drawn across the
    formation+padding range only, not the whole chart.
    """
    levels = [
        ("breakout", event.get("breakout_level"), _ENTRY_COLOR, "solid"),
        ("target", event.get("target_price"), _TARGET_COLOR, "dash"),
        ("stop", event.get("stop_price"), _STOP_COLOR, "dash"),
    ]
    seen_y: set[float] = set()
    for label, price, color, dash in levels:
        if price is None or pd.isna(price):
            continue
        # Avoid drawing entry + breakout twice when they coincide (which
        # is the v1 default).
        if round(float(price), 6) in seen_y:
            continue
        seen_y.add(round(float(price), 6))
        fig.add_trace(
            go.Scatter(
                x=list(x_range),
                y=[float(price), float(price)],
                mode="lines",
                line={"color": color, "width": 1.2, "dash": dash},
                name=label,
                hoverinfo="name+y",
            )
        )


def _shade_formation(
    fig: go.Figure,
    formation_start: pd.Timestamp,
    formation_end: pd.Timestamp,
) -> None:
    """Faint blue band over the formation window so the eye lands there."""
    fig.add_vrect(
        x0=formation_start,
        x1=formation_end,
        fillcolor=_FORMATION_FILL,
        line_width=0,
        layer="below",
    )


def _build_title(event: dict[str, Any]) -> str:
    """Single-line title: 'DOUBLE_TOP / SPY / 1995-04-12 / Q=68'.

    Direction is no longer included — events are direction-agnostic by
    detection contract; the strategy / direction-map decides direction
    elsewhere.
    """
    pattern = event.get("pattern")
    pattern_str = pattern.value if hasattr(pattern, "value") else str(pattern)
    asset = event.get("asset", "?")
    breakout = event.get("breakout_ts")
    breakout_str = pd.Timestamp(breakout).strftime("%Y-%m-%d") if breakout is not None else ""
    quality = event.get("quality")
    q_str = f"Q={quality:.0f}" if quality is not None and not pd.isna(quality) else ""
    variant = event.get("variant")
    bits = [pattern_str, asset, breakout_str, q_str]
    if variant:
        bits.append(str(variant))
    return " · ".join(b for b in bits if b)


def _event_to_dict(event: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(event, pd.Series):
        return event.to_dict()
    if isinstance(event, dict):
        return event
    msg = f"event must be a Series or dict, got {type(event).__name__}"
    raise TypeError(msg)


def _direction_str(event: dict[str, Any]) -> str:
    """Lowercase direction string for plot colouring.

    Detection emits no direction; this falls back to ``"bullish"`` so the
    horizon shading defaults to the long-side colour. Plot consumers
    that want bearish colouring can synthesise a ``direction`` field on
    the event dict before passing it in.
    """
    d = event.get("direction")
    if d is None:
        return "bullish"
    return d.value if hasattr(d, "value") else str(d).lower()


def _pattern_str(event: dict[str, Any]) -> str:
    p = event.get("pattern")
    return p.value if hasattr(p, "value") else str(p)


def _pattern_color(name: str) -> str:
    """Pick the legend / shape colour for a pattern name."""
    return _PATTERN_COLORS.get(name, "#7f7f7f")


def _bar_offset_ts(
    bars_full: pd.DataFrame,
    asset: str,
    breakout_ts: pd.Timestamp,
    horizon: int,
) -> pd.Timestamp | None:
    """Return the timestamp ``horizon`` bars after ``breakout_ts`` on the
    asset's bar series. Returns ``None`` when the lookahead exceeds the
    series — caller chooses a fallback.
    """
    idx = _select_asset_ohlc(bars_full, asset).index
    try:
        pos = idx.get_loc(breakout_ts)
    except KeyError:
        return None
    if isinstance(pos, slice):
        pos = pos.start
    target_pos = int(pos) + int(horizon)
    if target_pos >= len(idx):
        return None
    return idx[target_pos]


def _add_pattern_shape(
    fig: go.Figure,
    event: dict[str, Any],
    *,
    color: str | None = None,
    legendgroup: str | None = None,
    show_in_legend: bool = False,
    name: str | None = None,
) -> None:
    """Connect the event's pivots in chronological order so the formation
    is visually identifiable as a single object — not five disconnected
    triangle markers.

    The polyline is drawn on top of the candles with a slight transparency
    so OHLC bars stay readable under it. Hover shows pattern + direction
    + breakout date + quality + variant.
    """
    pivots = event.get("pivots") or []
    if len(pivots) < 2:
        return
    pivots_sorted = sorted(pivots, key=lambda p: p["ts"])
    xs = [p["ts"] for p in pivots_sorted]
    ys = [p["price"] for p in pivots_sorted]
    line_color = color or _pattern_color(_pattern_str(event))
    pattern_label = _pattern_str(event)
    breakout = pd.Timestamp(event["breakout_ts"]).strftime("%Y-%m-%d")
    quality = event.get("quality")
    q_str = f"Q={quality:.0f}" if quality is not None and not pd.isna(quality) else ""
    variant = event.get("variant") or ""
    hover = (
        f"<b>{pattern_label}</b> ({_direction_str(event)})<br>"
        f"breakout: {breakout}<br>"
        f"{q_str}{(' · ' + variant) if variant else ''}"
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": line_color, "width": 2},
            marker={"size": 7, "color": line_color, "line": {"width": 1, "color": "white"}},
            name=name or pattern_label,
            legendgroup=legendgroup or pattern_label,
            showlegend=show_in_legend,
            hovertemplate=hover + "<extra></extra>",
        )
    )


def _add_horizon_markers(
    fig: go.Figure,
    event: dict[str, Any],
    bars_full: pd.DataFrame,
    *,
    horizon: int,
    legendgroup: str | None = None,
) -> None:
    """Draw vertical lines at ``breakout_ts`` and ``breakout_ts + horizon``,
    plus a faint direction-coloured shade between them. Lets the reader
    see exactly where the metric grades the trade outcome.

    Uses ``add_shape`` rather than ``add_vline`` because the latter has a
    plotly bug with tz-aware datetime axes (TypeError on shape padding).
    """
    if horizon is None or horizon <= 0:
        return
    breakout = pd.Timestamp(event["breakout_ts"])
    horizon_end = _bar_offset_ts(bars_full, str(event["asset"]), breakout, horizon)
    if horizon_end is None:
        return
    direction = _direction_str(event)
    fill = _HORIZON_FILL_BULLISH if direction == "bullish" else _HORIZON_FILL_BEARISH
    fig.add_vrect(
        x0=breakout,
        x1=horizon_end,
        fillcolor=fill,
        line_width=0,
        layer="below",
    )
    # Vertical lines via add_shape — paper-y coordinates (0..1) avoid the
    # need to know the data y range and dodge the add_vline tz bug.
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=breakout,
        x1=breakout,
        y0=0,
        y1=1,
        line={"color": _HORIZON_LINE, "width": 1, "dash": "solid"},
        layer="above",
    )
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=horizon_end,
        x1=horizon_end,
        y0=0,
        y1=1,
        line={"color": _HORIZON_LINE, "width": 1, "dash": "dash"},
        layer="above",
    )
    # Top-aligned annotations explain the lines so the reader knows what
    # they mean without a tooltip.
    fig.add_annotation(
        x=breakout,
        y=1.0,
        xref="x",
        yref="paper",
        text="breakout",
        showarrow=False,
        font={"size": 10, "color": "rgba(0,0,0,0.7)"},
        yanchor="bottom",
        xanchor="left",
    )
    fig.add_annotation(
        x=horizon_end,
        y=1.0,
        xref="x",
        yref="paper",
        text=f"h={horizon}",
        showarrow=False,
        font={"size": 10, "color": "rgba(0,0,0,0.7)"},
        yanchor="bottom",
        xanchor="left",
    )


def plot_pattern_event(
    event: pd.Series | dict[str, Any],
    bars: pd.DataFrame,
    *,
    padding: int = 20,
    show_levels: bool = True,
    horizon: int | None = 20,
    theme: str | None = None,
) -> go.Figure:
    """Plot one detected pattern on a candlestick chart with annotations.

    Parameters
    ----------
    event
        A single row from the events table (``pd.Series`` from
        ``events.iloc[i]``) or a dict with the same keys.
    bars
        The MultiIndex Bars frame the event came from.
    padding
        Bars of context to show before formation_start and after
        formation_end. Defaults to 20 (≈ a month on daily data).
        Increased automatically when ``horizon`` exceeds it so the
        horizon-end marker stays visible.
    show_levels
        If True, overlay horizontal lines for ``breakout_level`` /
        ``target_price`` / ``stop_price`` (when filled).
    horizon
        If set, mark ``breakout_ts`` with a solid vertical line and
        ``breakout_ts + horizon`` with a dashed line; shade the holding
        window faintly. Set to ``None`` to suppress. Defaults to 20 to
        match :func:`fundcloud.metrics.feature_quality.evaluate`'s
        headline horizon.
    theme
        Plotly template alias — see :mod:`fundcloud.plots.themes`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    ev = _event_to_dict(event)
    asset = str(ev["asset"])
    ohlc_full = _select_asset_ohlc(bars, asset)
    formation_start = pd.Timestamp(ev["formation_start"])
    formation_end = pd.Timestamp(ev["formation_end"])
    # Stretch padding so the horizon-end marker stays in frame.
    effective_padding = max(int(padding), int(horizon) + 5) if horizon else int(padding)
    window = _slice_window(ohlc_full, formation_start, formation_end, padding=effective_padding)
    if window.empty:
        msg = f"no bars in plotting window for {asset} around {formation_start}"
        raise ValueError(msg)

    fig = go.Figure()
    _shade_formation(fig, formation_start, formation_end)
    _add_candlestick(fig, window, asset)
    # Draw the formation shape FIRST so pivot markers sit on top of it.
    _add_pattern_shape(fig, ev)
    _add_pivots(fig, ev.get("pivots") or [])
    trend_lines = (ev.get("meta") or {}).get("trend_lines") or []
    _add_trend_lines(fig, trend_lines, window, bars, asset)
    if show_levels:
        _add_levels(fig, ev, x_range=(window.index[0], window.index[-1]))
    if horizon:
        _add_horizon_markers(fig, ev, bars, horizon=horizon)

    fig.update_layout(xaxis_rangeslider_visible=False)
    return _style(fig, title=_build_title(ev), theme=theme)


def plot_asset_patterns(
    bars: pd.DataFrame,
    asset: str,
    *,
    patterns: Iterable[Any] | None = None,
    min_quality: float = 50.0,
    horizon: int | None = None,
    show_horizon_for_top: int = 10,
    theme: str | None = None,
) -> go.Figure:
    """Single candlestick chart for one asset with **every** pattern's
    detections drawn on top, grouped by pattern in the legend.

    This is the "what's been happening on AAPL?" view. Each pattern gets
    a coloured polyline through its pivots so you can pick the formation
    shape out by colour; clicking a legend entry hides / shows that
    pattern's traces.

    Parameters
    ----------
    bars
        MultiIndex Bars frame.
    asset
        Asset to plot. Must be in ``bars``.
    patterns
        Iterable of :class:`fundcloud.features.patterns.Pattern` enum
        values or stable name strings to include. ``None`` (default)
        means all 9 patterns.
    min_quality
        Drop detections below this geometric quality. Defaults to 50,
        matching :func:`fundcloud.features.patterns.PatternIndicator`.
    horizon
        If set, shade the ``[breakout_ts, breakout_ts + horizon]``
        window faintly (direction-coloured) for the most-recent
        ``show_horizon_for_top`` events. ``None`` (default) suppresses
        all shading — recommended on multi-pattern charts because
        drawing per-event horizon rectangles for hundreds of events
        produces a "barcode" background that loses individual meaning.
    show_horizon_for_top
        When ``horizon`` is set, cap how many of the most-recent events
        get a horizon shade. Defaults to 10. Set higher to extend the
        shading further back, set to ``0`` for no cap (every event
        shaded — only useful when there are few events).
    theme
        Plotly template alias.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> fig = bars.fc.plot_asset_patterns("AAPL")               # doctest: +SKIP
    >>> fig.show()                                              # doctest: +SKIP
    >>> fig = bars.fc.plot_asset_patterns(                      # doctest: +SKIP
    ...     "META",
    ...     patterns=[Pattern.TRIPLE_TOP, Pattern.DOUBLE_TOP],
    ...     min_quality=70,
    ... )
    """
    from fundcloud.features.indicators.base import _REGISTRY
    from fundcloud.features.patterns import Pattern

    if patterns is None:
        target_patterns: list[Pattern] = list(Pattern)
    else:
        target_patterns = []
        for p in patterns:
            target_patterns.append(p if isinstance(p, Pattern) else Pattern(str(p)))

    ohlc = _select_asset_ohlc(bars, asset)
    if ohlc.empty:
        msg = f"no bars for asset {asset!r}"
        raise ValueError(msg)
    # Slice down to the requested asset once so each detector scans a
    # single-asset frame instead of the full universe.
    asset_bars = bars.loc[:, pd.IndexSlice[:, [asset]]]

    fig = go.Figure()
    _add_candlestick(fig, ohlc, asset)

    # Pre-collect every event from every pattern so we can compute a
    # single "most recent N events globally" cutoff for horizon shading
    # — otherwise the cap would be spent on whichever pattern iterates
    # first instead of on the most-recent events across all patterns.
    per_pattern_events: list[tuple[Any, pd.DataFrame]] = []
    all_breakouts: list[pd.Timestamp] = []
    for pattern in target_patterns:
        if pattern.value not in _REGISTRY:
            continue
        Cls = _REGISTRY[pattern.value]
        events = (
            Cls(min_quality=min_quality)
            .events(asset_bars)
            .sort_values("breakout_ts", ascending=False)
        )
        if events.empty:
            continue
        per_pattern_events.append((pattern, events))
        all_breakouts.extend(events["breakout_ts"].tolist())

    horizon_cutoff: pd.Timestamp | None = None
    horizon_cap = (
        int(show_horizon_for_top) if show_horizon_for_top and show_horizon_for_top > 0 else None
    )
    if horizon and horizon_cap is not None and all_breakouts:
        sorted_breakouts = sorted(all_breakouts, reverse=True)
        cutoff_idx = min(horizon_cap, len(sorted_breakouts)) - 1
        horizon_cutoff = sorted_breakouts[cutoff_idx]

    legend_seen: set[str] = set()
    horizon_drawn = 0
    for pattern, events in per_pattern_events:
        color = _pattern_color(pattern.value)
        for _, ev in events.iterrows():
            ev_dict = ev.to_dict()
            show_in_legend = pattern.value not in legend_seen
            legend_seen.add(pattern.value)
            label = f"{pattern.value} ({len(events)})"
            _add_pattern_shape(
                fig,
                ev_dict,
                color=color,
                legendgroup=pattern.value,
                show_in_legend=show_in_legend,
                name=label,
            )
            if horizon:
                eligible = horizon_cutoff is None or ev["breakout_ts"] >= horizon_cutoff
                if eligible and (horizon_cap is None or horizon_drawn < horizon_cap):
                    _add_horizon_markers_silent(fig, ev_dict, bars, horizon=horizon)
                    horizon_drawn += 1

    title = f"{asset} — pattern timeline"
    if horizon:
        title += f" · h={horizon} (shaded for {horizon_drawn} most recent)"
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="closest")
    return _style(fig, title=title, theme=theme)


def _add_horizon_markers_silent(
    fig: go.Figure,
    event: dict[str, Any],
    bars_full: pd.DataFrame,
    *,
    horizon: int,
) -> None:
    """Like ``_add_horizon_markers`` but without the per-event vertical
    lines + annotations (those would clutter a multi-pattern overview).
    Just the faint direction-coloured shade.
    """
    breakout = pd.Timestamp(event["breakout_ts"])
    horizon_end = _bar_offset_ts(bars_full, str(event["asset"]), breakout, horizon)
    if horizon_end is None:
        return
    direction = _direction_str(event)
    fill = _HORIZON_FILL_BULLISH if direction == "bullish" else _HORIZON_FILL_BEARISH
    fig.add_vrect(
        x0=breakout,
        x1=horizon_end,
        fillcolor=fill,
        line_width=0,
        layer="below",
    )


def plot_patterns_overview(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    asset: str,
    *,
    max_events: int | None = None,
    horizon: int | None = None,
    show_horizon_for_top: int = 10,
    theme: str | None = None,
) -> go.Figure:
    """Continuous candlestick chart for one asset with each event's
    formation shape drawn (pivots connected by a coloured polyline) and
    optional horizon shading.

    Useful for spotting clustering — a dozen Triple Tops bunched in a
    six-month window often signals a regime, not nine independent setups.

    Parameters
    ----------
    events
        Events table (as returned by ``indicator.events(bars)`` or
        ``bars.fc.pattern_events(...)``).
    bars
        The full MultiIndex Bars frame.
    asset
        Asset to plot. Events are filtered to this asset.
    max_events
        Cap on number of events drawn — useful for sparing the renderer
        on assets with hundreds of events. ``None`` (default) plots all.
    horizon
        If set, lightly shade ``[breakout_ts, breakout_ts + horizon]``
        per event so the metric grading window is visible. Set to
        ``None`` to suppress.
    theme
        Plotly template alias.
    """
    asset_events = events[events["asset"] == asset].copy()
    asset_events = asset_events.sort_values("breakout_ts")
    if max_events is not None:
        asset_events = asset_events.head(int(max_events))

    ohlc = _select_asset_ohlc(bars, asset)
    fig = go.Figure()
    _add_candlestick(fig, ohlc, asset)
    if asset_events.empty:
        return _style(
            fig,
            title=f"{asset} — no events",
            theme=theme,
        )

    # Cap horizon shading to the most-recent N events. With hundreds of
    # events on a long timeline, drawing a 20-bar shade per event covers
    # virtually the whole background and the stripes lose individual
    # meaning ("barcode" effect).
    n_total = len(asset_events)
    horizon_cutoff_idx = n_total - int(show_horizon_for_top) if horizon else n_total

    # One legend entry per pattern present in the slice. Each event's
    # formation shape (pivots connected) goes in that pattern's group,
    # only the first event in the group is added to the legend.
    seen_patterns: set[str] = set()
    for i, (_, ev) in enumerate(asset_events.iterrows()):
        ev_dict = ev.to_dict()
        p_name = _pattern_str(ev_dict)
        show = p_name not in seen_patterns
        seen_patterns.add(p_name)
        _add_pattern_shape(
            fig,
            ev_dict,
            color=_pattern_color(p_name),
            legendgroup=p_name,
            show_in_legend=show,
            name=p_name,
        )
        # Sorted ascending by breakout_ts → the *last* N rows are the
        # most recent. Shade only those when horizon is enabled.
        if horizon and i >= horizon_cutoff_idx:
            _add_horizon_markers_silent(fig, ev_dict, bars, horizon=horizon)

    title = f"{asset} — {len(asset_events)} events"
    if "pattern" in asset_events.columns and asset_events["pattern"].nunique() == 1:
        p = asset_events["pattern"].iloc[0]
        p_str = getattr(p, "value", str(p))
        title = f"{asset} — {p_str} ({len(asset_events)} events)"
    if horizon:
        n_shaded = max(0, n_total - horizon_cutoff_idx)
        title += f" · h={horizon} (shaded for {n_shaded} most recent)"

    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="closest")
    return _style(fig, title=title, theme=theme)
