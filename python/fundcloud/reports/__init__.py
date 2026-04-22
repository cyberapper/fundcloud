"""Tear-sheet output — HTML, PDF, Excel.

The primary public object is :class:`Tearsheet`. Renderer modules are
lazy-loaded via :class:`Tearsheet.render_*` methods, so installing
``fundcloud`` without ``[reports]`` stays cheap.
"""

from __future__ import annotations

from fundcloud.reports.tearsheet import Tearsheet

__all__ = ["Tearsheet"]
