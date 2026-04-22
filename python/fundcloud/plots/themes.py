"""Plotly theme selection for fundcloud figures.

Thin alias layer over ``plotly.io.templates`` — no custom dataclass, no
registry mutation. The five fundcloud aliases map to plotly's builtin
templates:

============  ==================
fundcloud     plotly template
============  ==================
``default``   ``None`` (plotly's own default)
``white``     ``plotly_white``
``dark``      ``plotly_dark``
``ggplot2``   ``ggplot2``
``seaborn``   ``seaborn``
============  ==================

Any template name registered in ``plotly.io.templates`` is also accepted,
so user-defined templates plug in transparently::

    import plotly.io as pio
    pio.templates["my-brand"] = my_template
    fundcloud.plots.set_theme("my-brand")
"""

from __future__ import annotations

__all__ = [
    "ALIASES",
    "get_theme",
    "set_theme",
]


ALIASES: dict[str, str | None] = {
    "default": None,
    "white": "plotly_white",
    "dark": "plotly_dark",
    "ggplot2": "ggplot2",
    "seaborn": "seaborn",
}

_active: str = "default"


def set_theme(name: str) -> None:
    """Select the plotly template used by subsequent fundcloud figures.

    Parameters
    ----------
    name
        A fundcloud alias (``"default"``, ``"white"``, ``"dark"``,
        ``"ggplot2"``, ``"seaborn"``) or any name registered in
        ``plotly.io.templates``.
    """
    global _active
    _validate(name)
    _active = name


def get_theme() -> str:
    """Return the currently active theme name."""
    return _active


def _validate(name: str) -> None:
    if name in ALIASES:
        return
    # Lazy import: fundcloud.plots.__init__ may be imported before plotly is
    # touched for real work; importing plotly here is cheap once it's loaded.
    import plotly.io as pio

    if name not in pio.templates:
        known = sorted(set(ALIASES) | set(pio.templates))
        msg = f"unknown theme {name!r}; known themes: {known}"
        raise ValueError(msg)


def _resolve_template(theme: str | None) -> str | None:
    """Return the plotly template name for ``theme`` (or ``None``).

    ``theme=None`` uses the currently active theme (see :func:`set_theme`).
    """
    name = theme if theme is not None else _active
    if name in ALIASES:
        return ALIASES[name]
    # User-registered plotly template — pass straight through.
    return name
