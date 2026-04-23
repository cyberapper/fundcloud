"""``Catalog`` — name → (source ``Backend``, sink ``Backend`` key, refresh policy).

A ``Catalog`` is the orchestrator that binds a user-facing dataset name to a
read backend (the source) and a key inside the catalog's sink backend
(the cache). Refreshes call :meth:`Backend.sync_to` with ``mode='upsert'``
so overlapping rows from re-pulls dedup on the timestamp index.

Per-dataset overrides are persisted in :attr:`DatasetSpec.refresh_kwargs`
with these recognised keys:

- ``start``: minimum date to pull on initial load.
- ``end``: maximum date (rare; usually omitted).
- ``lookback``: ``pd.Timedelta``-compatible window subtracted from the sink
  watermark on :meth:`refresh`. Used to re-pull recent rows that upstream
  may correct (corporate actions, restatements, exchange revisions).

The spec format is a plain Python dict — YAML is out of scope on purpose
(callers do ``yaml.safe_load(path.read_text())`` and pass the dict in).
See :meth:`Catalog.to_spec` / :meth:`Catalog.from_spec` for round-trippable
serialisation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

import pandas as pd

from fundcloud.data._base import Backend

__all__ = ["Catalog", "DatasetSpec"]


@dataclass(slots=True)
class DatasetSpec:
    """Declarative binding for a single dataset."""

    name: str
    source: Backend
    store_key: str
    refresh_kwargs: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


class Catalog:
    """A named collection of datasets sharing a single sink :class:`Backend`."""

    def __init__(self, store: Backend) -> None:
        self._store = store
        self._specs: dict[str, DatasetSpec] = {}

    # ------------------------------------------------------------------ specs

    def register(
        self,
        name: str,
        source: Backend,
        *,
        store_key: str | None = None,
        refresh_kwargs: Mapping[str, Any] | None = None,
        tags: tuple[str, ...] = (),
    ) -> DatasetSpec:
        """Register a dataset. Returns the resulting :class:`DatasetSpec`."""
        if name in self._specs:
            msg = f"dataset {name!r} already registered"
            raise ValueError(msg)
        spec = DatasetSpec(
            name=name,
            source=source,
            store_key=store_key or name,
            refresh_kwargs=dict(refresh_kwargs or {}),
            tags=tuple(tags),
        )
        self._specs[name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self) -> Any:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def store(self) -> Backend:
        return self._store

    # ------------------------------------------------------------------ access

    def spec(self, name: str) -> DatasetSpec:
        try:
            return self._specs[name]
        except KeyError as e:
            msg = f"dataset {name!r} is not registered"
            raise KeyError(msg) from e

    def load(
        self,
        name: str,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        prefer_store: bool = True,
    ) -> pd.DataFrame:
        """Return rows for ``name``.

        When ``prefer_store=True`` and the sink already has the dataset, the
        sink is the source of truth (callers who want fresh data should call
        :meth:`refresh` first). Otherwise the source is pulled and the result
        is persisted before being returned.

        Call-site ``start`` / ``end`` win over ``refresh_kwargs.start`` /
        ``refresh_kwargs.end``; both are forwarded to the underlying
        :meth:`Backend.read`.
        """
        spec = self.spec(name)
        kw = spec.refresh_kwargs
        eff_start = start if start is not None else kw.get("start")
        eff_end = end if end is not None else kw.get("end")

        if prefer_store and self._store.exists(spec.store_key):
            return self._store.read(spec.store_key, start=eff_start, end=eff_end)
        df = spec.source.read(start=eff_start, end=eff_end)
        if start is None and end is None:
            self._store.write(spec.store_key, df, mode="overwrite")
        return df

    # ------------------------------------------------------------------ refresh

    def refresh(
        self,
        name: str,
        *,
        end: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Pull incremental rows for ``name`` and upsert them into the sink.

        The sink watermark (``last_index``) is the default ``start``. If
        :attr:`DatasetSpec.refresh_kwargs` carries a ``lookback`` window, it
        is subtracted from the watermark so recently-corrected rows get
        re-pulled and deduplicated by ``mode='upsert'``.
        """
        spec = self.spec(name)
        last = self._store.last_index(spec.store_key)
        kw = spec.refresh_kwargs

        if last is not None:
            lookback = pd.Timedelta(kw.get("lookback", 0))
            start: pd.Timestamp | str | None = last - lookback
        else:
            start = kw.get("start")

        eff_end = end if end is not None else kw.get("end")
        return spec.source.sync_to(
            self._store,
            key=spec.store_key,
            start=start,
            end=eff_end,
            mode="upsert",
        )

    def refresh_all(
        self,
        *,
        end: pd.Timestamp | str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Refresh every (optionally tag-filtered) dataset."""
        out: dict[str, pd.DataFrame] = {}
        for name, spec in self._specs.items():
            if tags is not None and not set(tags).issubset(set(spec.tags)):
                continue
            out[name] = self.refresh(name, end=end)
        return out

    # ------------------------------------------------------------------ describe

    def describe(self, name: str | None = None) -> pd.DataFrame:
        """Produce a one-row-per-dataset summary frame."""
        specs = [self._specs[name]] if name is not None else list(self._specs.values())
        rows = []
        now = datetime.now(timezone.utc)
        for spec in specs:
            last = self._store.last_index(spec.store_key)
            if last is not None:
                ts = pd.Timestamp(last)
                utc_ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
                lag = now - utc_ts
            else:
                lag = None
            rows.append({
                "name": spec.name,
                "source": _qualname(spec.source),
                "symbols": getattr(spec.source, "symbols", []),
                "interval": getattr(spec.source, "interval", None),
                "store_key": spec.store_key,
                "last_index": last,
                "lag": lag,
                "tags": list(spec.tags),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ spec I/O

    def to_spec(self) -> dict[str, dict[str, Any]]:
        """Serialise the catalog into a dict that ``from_spec`` can read.

        Sources are referenced by their fully-qualified class name plus their
        constructor kwargs. Callers are responsible for making sure the
        referenced modules are importable when ``from_spec`` runs.
        """
        out: dict[str, dict[str, Any]] = {}
        for name, spec in self._specs.items():
            out[name] = {
                "source": _qualname(spec.source),
                "source_kwargs": _source_kwargs(spec.source),
                "store_key": spec.store_key,
                "refresh_kwargs": dict(spec.refresh_kwargs),
                "tags": list(spec.tags),
            }
        return out

    @classmethod
    def from_spec(cls, store: Backend, spec: Mapping[str, Mapping[str, Any]]) -> Catalog:
        """Build a catalog from a mapping of ``{name: {source, source_kwargs, ...}}``."""
        cat = cls(store=store)
        for name, row in spec.items():
            source_cls = _import_dotted(row["source"])
            source = source_cls(**row.get("source_kwargs", {}))
            cat.register(
                name,
                source,
                store_key=row.get("store_key"),
                refresh_kwargs=row.get("refresh_kwargs", {}),
                tags=tuple(row.get("tags", [])),
            )
        return cat


# ---------------------------------------------------------------------- helpers


def _qualname(obj: object) -> str:
    klass = type(obj) if not isinstance(obj, type) else obj
    return f"{klass.__module__}.{klass.__qualname__}"


def _import_dotted(path: str) -> type:
    module_name, _, cls_name = path.rpartition(".")
    if not module_name:
        msg = f"expected fully-qualified class path, got {path!r}"
        raise ValueError(msg)
    module = import_module(module_name)
    return getattr(module, cls_name)


def _source_kwargs(source: Backend) -> dict[str, Any]:
    """Best-effort reconstruction of kwargs needed to rebuild a source.

    Exposed as a convenience for ``to_spec`` / ``from_spec`` round-trips
    used by tests and lightweight config files. For anything more complex,
    callers should serialise their own config.
    """
    captured: dict[str, Any] = {}
    for attr in ("symbols", "interval", "path", "root", "table", "query", "date_col"):
        if hasattr(source, attr):
            val = getattr(source, attr)
            if val is None:
                continue
            captured[attr] = str(val) if attr in ("path", "root") else val
    return captured
