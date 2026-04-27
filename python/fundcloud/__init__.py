"""Fundcloud — portfolio research, end-to-end, with a Rust core.

Importing this package registers the ``.fc`` pandas accessor on
:class:`pandas.Series` and :class:`pandas.DataFrame` as a side effect.
"""

from __future__ import annotations

# Side effect: register pandas accessors. Safe to import repeatedly — pandas
# raises UserWarning rather than failing when the namespace is already taken.
from fundcloud import accessors as _accessors  # noqa: F401
from fundcloud import accounts, errors
from fundcloud._version import __version__
from fundcloud.plots import get_theme, set_theme

__all__ = ["__version__", "accounts", "errors", "get_theme", "set_theme"]
