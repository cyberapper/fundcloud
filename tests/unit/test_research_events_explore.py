"""Tests for the exploration views (forward paths, outcome profile, portfolio).

The load-bearing property is that side is decided by the *data*, not the
detector's geometric branch: a synthetic up-drift fixture must read as "up" and
suggest long, and ``side="auto"`` must pick the profitable side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fundcloud.research.events.explore import (
    event_portfolio,
    evidence_table,
    forward_paths,
    outcome_profile,
    portfolio_by_event,
    return_distribution,
    tag_episodes,
)
from fundcloud.research.events.schema import build_observations


def _panel_up_drift(n: int = 80) -> pd.DataFrame:
    """A single-symbol panel that drifts up ~0.5%/bar with a steady ATR."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 * (1.005 ** np.arange(n))
    high = close * 1.01
    low = close * 0.99
    open_ = close / 1.005
    data = {
        ("open", "AAA"): pd.Series(open_, index=idx),
        ("high", "AAA"): pd.Series(high, index=idx),
        ("low", "AAA"): pd.Series(low, index=idx),
        ("close", "AAA"): pd.Series(close, index=idx),
        ("volume", "AAA"): pd.Series(np.ones(n), index=idx),
    }
    panel = pd.DataFrame(data, index=idx)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    return panel.sort_index(axis=1)


def _obs_on(panel: pd.DataFrame, positions: list[int]) -> pd.DataFrame:
    """Observations confirmed at the given bar positions, with next-bar fills."""
    idx = panel.index
    rows = []
    for p in positions:
        rows.append({
            "event_id": "ev_test",
            "asset": "AAA",
            "timeframe": "1D",
            "formation_end_ts": idx[p],
            "confirmed_ts": idx[p],
            "execution_ts": idx[p + 1],
            "params": {},
            "logic_version": 1,
            "params_hash": "h",
            "entry_ref_price": float(panel[("open", "AAA")].iloc[p + 1]),
            "stop_ref_price": float("nan"),
            "zone_lo": float("nan"),
            "zone_hi": float("nan"),
            "quality": float("nan"),
            "atr_at_confirm": 1.0,
        })
    return build_observations(rows)


def test_forward_paths_shape_and_leak_free_horizons() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, [10, 20, 30])

    paths = forward_paths(obs, panel, horizons=(1, 5, 10))

    assert set(paths.columns) >= {"event_id", "horizon", "ret", "net_atr", "up_atr", "dn_atr"}
    # 3 events x 3 horizons, all with lookahead available.
    assert len(paths) == 9
    # Up-drift fixture -> positive returns; the side is read from the data only.
    assert (paths["ret"] > 0).all()


def test_outcome_profile_reads_up_and_suggests_long_despite_label() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, [10, 20, 30, 40])

    prof = outcome_profile(forward_paths(obs, panel, horizons=(10,)), by=("event_id",))

    assert len(prof) == 1
    row = prof.iloc[0]
    assert row["frac_up"] == 1.0
    assert row["frac_down"] == 0.0
    assert row["suggested"] == "long"  # decided by the data's forward path
    # up/down/steady fractions partition the sample.
    assert abs(row["frac_up"] + row["frac_down"] + row["frac_steady"] - 1.0) < 1e-9


def test_return_distribution_percentiles_are_ordered() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, list(range(10, 40)))

    dist = return_distribution(forward_paths(obs, panel, horizons=(5,)), by=("event_id",))

    row = dist.iloc[0]
    assert row["ret_p05"] <= row["ret_p25"] <= row["ret_median"] <= row["ret_p75"] <= row["ret_p95"]


def test_event_portfolio_auto_picks_profitable_side() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, list(range(5, 60, 5)))

    equity, stats = event_portfolio(obs, panel, horizon=10, side="auto")

    # Up-drift -> auto must go long (+1) and beat zero.
    assert stats["side"] == 1.0
    assert stats["cagr"] > 0
    assert stats["max_drawdown"] <= 0.0
    assert equity["strategy"].iloc[-1] > 1.0
    assert {"strategy", "benchmark"} == set(equity.columns)


def test_event_portfolio_entry_day_fills_at_open_matches_forward_path() -> None:
    # FIX 2: a single long episode's portfolio total return at horizon h must
    # equal its forward_paths ret (open[exec] -> close[exec + h - 1]). With one
    # symbol, one event and zero cost the equal-weight strategy is exactly that
    # episode's path, so the entry bar must book open->close, not prev_close->close.
    panel = _panel_up_drift()
    obs = _obs_on(panel, [10])
    horizon = 7

    equity, stats = event_portfolio(obs, panel, horizon=horizon, side="long", cost_bps=0.0)
    paths = forward_paths(obs, panel, horizons=(horizon,))

    port_total_return = float(equity["strategy"].iloc[-1]) - 1.0
    expected = float(paths["ret"].iloc[0])
    assert port_total_return == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert stats["side"] == 1.0


def test_event_portfolio_charges_cost_on_exit_day() -> None:
    # FIX 3: cost is charged on the day a position drops to cash (full exit), not
    # dropped when n_active hits 0. Compare two zero-vs-nonzero cost runs: the only
    # difference is transaction cost, and it must include the exit-bar weight change.
    panel = _panel_up_drift()
    obs = _obs_on(panel, [10])  # single long episode, isolated so exit is a clean 1->0
    horizon = 5

    eq_free, _ = event_portfolio(obs, panel, horizon=horizon, side="long", cost_bps=0.0)
    eq_cost, _ = event_portfolio(obs, panel, horizon=horizon, side="long", cost_bps=6.0)

    free = eq_free["strategy"].to_numpy()
    paid = eq_cost["strategy"].to_numpy()
    one_way = 3.0 / 10_000.0  # half of 6 bps round-trip

    # Entry day: weight goes 0 -> 1, costing one_way.
    entry_day = 11  # execution bar (confirmed at 10, fills next bar)
    daily_free = free[entry_day] / free[entry_day - 1] - 1.0
    daily_paid = paid[entry_day] / paid[entry_day - 1] - 1.0
    assert daily_free - daily_paid == pytest.approx(one_way, rel=1e-9)

    # Exit day: weight goes 1 -> 0, which must also cost one_way (the dropped-exit bug).
    exit_day = entry_day + horizon  # first cash day after the hold
    daily_free_x = free[exit_day] / free[exit_day - 1] - 1.0
    daily_paid_x = paid[exit_day] / paid[exit_day - 1] - 1.0
    assert daily_free_x - daily_paid_x == pytest.approx(one_way, rel=1e-9)


def test_event_portfolio_auto_uses_episode_mean_not_raw_mean() -> None:
    # FIX 5: side="auto" decides from the episode mean, not the raw overlapping
    # mean. Build a panel where one short-lived UP episode (few bars) coexists with
    # a long persistent DOWN run (many overlapping detections): the raw mean is
    # dragged negative by the dense down detections, but the episode mean (one vote
    # per persistent signal) is positive, so auto must go long.
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = np.full(n, 100.0)
    # A short up-leg over [10, 25): strong daily up move (one episode).
    close[10:25] = 100.0 * (1.03 ** np.arange(15))
    up_end = close[24]
    # A long down-leg over [30, 75): mild daily decline but many overlapping firings.
    close[30:] = up_end * (0.999 ** np.arange(n - 30))
    high = close * 1.005
    low = close * 0.995
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    data = {
        ("open", "AAA"): pd.Series(open_, index=idx),
        ("high", "AAA"): pd.Series(high, index=idx),
        ("low", "AAA"): pd.Series(low, index=idx),
        ("close", "AAA"): pd.Series(close, index=idx),
        ("volume", "AAA"): pd.Series(np.ones(n), index=idx),
    }
    panel = pd.DataFrame(data, index=idx)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    panel = panel.sort_index(axis=1)

    horizon = 3
    # One up detection at the start of the up-leg; many dense down detections.
    up_positions = [10]
    down_positions = list(range(31, 70))  # heavily overlapping -> few episodes, many raw
    obs = _obs_on(panel, up_positions + down_positions)

    paths = forward_paths(obs, panel, horizons=(horizon,))
    raw_mean = float(paths["ret"].mean())
    episode_mean = float(
        tag_episodes(paths).loc[lambda t: t["is_episode"], "ret"].mean()
    )
    # The two disagree in sign: raw is dragged down by overlap, episodes are up.
    assert raw_mean < 0 < episode_mean

    _, stats = event_portfolio(obs, panel, horizon=horizon, side="auto")
    assert stats["side"] == 1.0  # episode mean (the correct vote) wins


def test_event_portfolio_short_side_loses_on_up_drift() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, list(range(5, 60, 5)))

    _, stats = event_portfolio(obs, panel, horizon=10, side="short")

    assert stats["side"] == -1.0
    assert stats["cagr"] < 0  # shorting an up-drift loses


def test_portfolio_by_event_one_row_per_group() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, list(range(5, 60, 5)))

    table = portfolio_by_event(obs, panel, horizon=10, by=("event_id",))

    assert len(table) == 1
    assert table.iloc[0]["event_id"] == "ev_test"
    assert table.iloc[0]["n_events"] == len(range(5, 60, 5))
    assert "max_drawdown" in table.columns


def test_tag_episodes_collapses_overlapping_detections() -> None:
    panel = _panel_up_drift()
    # Three detections one bar apart -> windows overlap heavily -> one episode.
    obs = _obs_on(panel, [10, 11, 12])
    paths = forward_paths(obs, panel, horizons=(10,))

    tagged = tag_episodes(paths)

    assert tagged["is_episode"].sum() == 1  # only the first appearance
    assert tagged["concurrency"].max() == 3  # all three overlap at the dense bar
    # ESS (sum of uniqueness) is between 1 and the raw count.
    ess = float(tagged["weight"].sum())
    assert 1.0 <= ess < 3.0


def test_outcome_profile_consolidates_and_reports_ess() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, [10, 11, 12])
    paths = forward_paths(obs, panel, horizons=(10,))

    prof = outcome_profile(paths, by=("event_id",))
    row = prof.iloc[0]

    assert row["n"] == 1  # episodes
    assert row["n_raw"] == 3  # raw detections
    assert 1.0 <= row["ess"] < 3.0
    # consolidate=False falls back to raw count.
    raw_prof = outcome_profile(paths, by=("event_id",), consolidate=False)
    assert raw_prof.iloc[0]["n"] == 3


def test_evidence_table_merges_outcome_and_portfolio() -> None:
    panel = _panel_up_drift()
    obs = _obs_on(panel, list(range(5, 60, 5)))

    table = evidence_table(obs, panel, horizon=10, by=("event_id",))

    assert len(table) == 1
    row = table.iloc[0]
    # Outcome side of the table (decided by the data's forward path).
    assert row["frac_up"] == 1.0
    assert row["suggested"] == "long"
    # Portfolio side of the table.
    assert row["side"] == 1.0
    assert row["cagr"] > 0
    assert row["max_drawdown"] <= 0.0
    assert {"frac_up", "frac_down", "frac_steady", "sharpe", "corr_to_bh",
            "avg_breadth"} <= set(table.columns)


def test_empty_obs_yields_empty_views() -> None:
    panel = _panel_up_drift()
    empty = build_observations([])

    assert forward_paths(empty, panel).empty
    assert outcome_profile(forward_paths(empty, panel)).empty
    assert return_distribution(forward_paths(empty, panel)).empty
    assert portfolio_by_event(empty, panel, horizon=10).empty
    assert evidence_table(empty, panel, horizon=10).empty
