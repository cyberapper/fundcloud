"""Integration-test fixtures.

Spawns ephemeral service containers (ClickHouse, …) on demand, seeds them
with deterministic synthetic data, and tears them down at session end.
Tests that depend on these fixtures must be marked ``@pytest.mark.docker``
so they're skipped by default; run them explicitly with::

    uv run pytest tests/integration -m docker -q

A test using ``clickhouse_backend`` will be skipped at collection time
when:

* ``testcontainers`` isn't installed,
* ``clickhouse_connect`` isn't installed,
* the Docker daemon isn't reachable.

That keeps ``pytest -m docker`` runnable on any developer's machine that
has Docker, while still degrading gracefully elsewhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

# ----------------------------------------------------------------- skip helpers


def _skip_if_docker_unavailable() -> None:
    """Skip with a friendly message when Docker / testcontainers are missing."""
    pytest.importorskip(
        "testcontainers.clickhouse",
        reason="testcontainers not installed — skipping docker-marked test",
    )
    pytest.importorskip(
        "clickhouse_connect",
        reason="clickhouse-connect not installed — skipping docker-marked test",
    )
    try:
        import docker
    except ImportError:  # pragma: no cover — covered by the importorskip above
        pytest.skip("docker python sdk missing")
    try:
        docker.from_env().ping()
    except Exception as e:
        pytest.skip(f"Docker daemon not reachable: {e}")


# ----------------------------------------------------------------- container


@pytest.fixture(scope="session")
def clickhouse_container() -> Iterator[object]:
    """Session-scoped Clickhouse container, seeded with the fixture schema."""
    _skip_if_docker_unavailable()
    from testcontainers.clickhouse import ClickHouseContainer

    container = ClickHouseContainer("clickhouse/clickhouse-server:24.8")
    container.start()
    try:
        _seed(container)
        yield container
    finally:
        container.stop()


def _seed(container: object) -> None:
    """Create the ``bars`` table and insert deterministic synthetic OHLCV+features."""
    import clickhouse_connect

    host = container.get_container_host_ip()  # type: ignore[attr-defined]
    port = int(container.get_exposed_port(8123))  # type: ignore[attr-defined]
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=container.username,  # type: ignore[attr-defined]
        password=container.password,  # type: ignore[attr-defined]
        database=container.dbname,  # type: ignore[attr-defined]
        secure=False,
    )
    try:
        client.command("DROP TABLE IF EXISTS bars")
        client.command("""
            CREATE TABLE bars (
                ts DateTime64(3),
                prefix String,
                code String,
                tf String,
                o Float64,
                h Float64,
                l Float64,
                c Float64,
                v Float64,
                rsi_14 Float64,
                sentiment Float64
            )
            ENGINE = MergeTree()
            ORDER BY (prefix, code, tf, ts)
        """)

        rng = np.random.default_rng(42)
        rows: list[tuple[object, ...]] = []
        # tz-aware UTC: clickhouse-connect would otherwise interpret naive
        # datetimes via the *client* timezone and shift them on insert,
        # which makes the test results depend on where the runner sits.
        ts_index = pd.date_range("2024-01-02", periods=60, freq="1D", tz="UTC")
        for prefix, code in [
            ("HKEX", "0001"),
            ("HKEX", "0002"),
            ("TSE", "7203"),
            ("TSE", "6758"),
        ]:
            for tf in ("1d", "1h"):
                base = 100.0 + rng.normal(0, 5)
                drift = rng.normal(0, 1, len(ts_index))
                close = base + np.cumsum(drift)
                open_ = close + rng.normal(0, 0.2, len(ts_index))
                high = np.maximum(open_, close) + rng.uniform(0, 0.5, len(ts_index))
                low = np.minimum(open_, close) - rng.uniform(0, 0.5, len(ts_index))
                vol = rng.integers(100_000, 200_000, len(ts_index)).astype(float)
                rsi = rng.uniform(20, 80, len(ts_index))
                sent = rng.normal(0, 0.3, len(ts_index))
                for i, ts in enumerate(ts_index):
                    rows.append((
                        ts.to_pydatetime(),
                        prefix,
                        code,
                        tf,
                        float(open_[i]),
                        float(high[i]),
                        float(low[i]),
                        float(close[i]),
                        float(vol[i]),
                        float(rsi[i]),
                        float(sent[i]),
                    ))

        client.insert(
            "bars",
            rows,
            column_names=[
                "ts",
                "prefix",
                "code",
                "tf",
                "o",
                "h",
                "l",
                "c",
                "v",
                "rsi_14",
                "sentiment",
            ],
        )
    finally:
        client.close()


# ----------------------------------------------------------------- backend


@pytest.fixture
def clickhouse_kwargs(clickhouse_container: object) -> dict[str, object]:
    """Connection params resolved from the running container."""
    return {
        "host": clickhouse_container.get_container_host_ip(),  # type: ignore[attr-defined]
        "port": int(clickhouse_container.get_exposed_port(8123)),  # type: ignore[attr-defined]
        "user": clickhouse_container.username,  # type: ignore[attr-defined]
        "password": clickhouse_container.password,  # type: ignore[attr-defined]
        "database": clickhouse_container.dbname,  # type: ignore[attr-defined]
        "ssl": False,
    }


@pytest.fixture
def clickhouse_multi_asset(clickhouse_kwargs: dict[str, object]) -> object:
    """ClickHouse backend configured for the seeded ``bars`` table, multi-asset mode."""
    from fundcloud.data import ClickHouse

    return ClickHouse(
        table="bars",
        asset_cols=["prefix", "code"],
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        **clickhouse_kwargs,
    )


@pytest.fixture
def clickhouse_single_asset(clickhouse_kwargs: dict[str, object]) -> object:
    """ClickHouse backend configured to read one (prefix, code) slice as a single asset."""
    from fundcloud.data import ClickHouse

    return ClickHouse(
        table="bars",
        timestamp_col="ts",
        timeframe_col="tf",
        timeframe="1d",
        ohlcv_map={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"},
        where="prefix = 'HKEX' AND code = '0001'",
        **clickhouse_kwargs,
    )
