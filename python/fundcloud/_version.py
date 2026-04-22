"""Single source of truth for the library version.

Kept in its own module so ``python -c "import fundcloud._version"`` can be run
without importing the whole public API (useful for tooling).
"""

from __future__ import annotations

__version__ = "0.1.0"
