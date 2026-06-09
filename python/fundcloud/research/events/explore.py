"""Exploration views over detected events — observe, don't gate.

At this research stage the goal is to *understand* what an event does, not to
accept/reject it, and **not** to assume a trade direction from the detector's
geometric branch. Each detector splits its two branches into distinct
``event_id``\\ s (``_up`` / ``_dn`` — the detection equation, not a trade
mandate): we group by them so we can check empirically whether an "up" setup
actually leads to up moves — but the action (long / short / skip) is decided from
the realised forward path, never from the suffix.

Every view reads the leak-free forward path of each event (measured strictly
after ``execution_ts``, the next-bar-open fill):

* :func:`forward_paths` — the shared foundation: one tidy row per
  ``(event, horizon)`` with the raw forward return, the net move in ATR units,
  and the up/down excursions in ATR units. Everything else aggregates this.
* :func:`outcome_profile` — the **data-driven** answer to "if this event
  happens, does price move up, down, or stay steady?". Per event (and
  parameterisation) it reports the up / down / steady fractions and the side the
  data implies — independent of the geometric branch.
* :func:`return_distribution` — the full shape (median, quartiles, tails), not
  just the mean, because event payoffs are heavy-tailed.
* :func:`event_portfolio` / :func:`portfolio_by_event` — a daily-position equity
  curve versus equal-weight buy-and-hold (CAGR, max drawdown, Sharpe,
  correlation to B&H). The trade side is a *decision input* (``"long"``,
  ``"short"`` or ``"auto"`` = inferred from the data), not the geometric branch.
  This is what answers "lower return but smaller drawdown is still good": an
  event is judged at the **portfolio** level.

None of these apply a pass/fail threshold — that is deliberate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundcloud.research.events.study import STUDY_HORIZONS, decode_params

__all__ = [
    "event_portfolio",
    "evidence_table",
    "forward_paths",
    "outcome_profile",
    "portfolio_by_event",
    "return_distribution",
    "tag_episodes",
    "variant_leaderboard",
]

_TRADING_DAYS = 252


def _asset_frame(panel: pd.DataFrame, asset: str) -> pd.DataFrame:
    """One asset's OHLC frame from a ``(field, symbol)`` panel, NaN rows dropped."""
    sub = panel.xs(asset, level=-1, axis=1)
    return sub.dropna(subset=["open", "high", "low", "close"])


def forward_paths(
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = STUDY_HORIZONS,
) -> pd.DataFrame:
    """Expand each observation into its forward path, per horizon.

    For every event the entry is ``entry_ref_price`` (the next-bar-open fill at
    ``execution_ts``) and the ATR unit is ``atr_at_confirm``. For each requested
    horizon ``h`` the forward window is the ``h`` bars from the execution bar; an
    event is dropped at horizons that run past the end of its series.

    The event's geometric branch lives in its ``event_id`` (``_up`` / ``_dn``
    suffix) — it is the detection equation, not a trade decision. Returns are raw
    (``ret > 0`` means price rose), never signed by the branch.

    Parameters
    ----------
    obs
        Observation frame (:func:`fundcloud.research.events.scan_panel` output).
    panel
        The same ``(field, symbol)`` panel the events were detected on.
    horizons
        Forward horizons in bars.

    Returns
    -------
    pandas.DataFrame
        One row per ``(event, horizon)`` with: ``event_id``, ``asset``,
        ``confirmed_ts``, ``params_hash``, ``horizon``,
        ``exec_pos`` (integer bar index of the execution bar in the asset's
        series — the unit overlap/episodes are measured in), ``entry``, ``ret``
        (raw close-to-entry return), ``net_atr`` (signed close move / ATR),
        ``up_atr`` (max up excursion / ATR), ``dn_atr`` (max down excursion /
        ATR, ``>= 0``). Empty obs yields an empty frame.
    """
    cols = ["event_id", "asset", "confirmed_ts", "params_hash", "horizon",
            "exec_pos", "entry", "ret", "net_atr", "up_atr", "dn_atr"]
    if obs.empty:
        return pd.DataFrame(columns=cols)

    rows: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}

    for _, ev in obs.iterrows():
        asset = str(ev["asset"])
        if asset not in frames:
            frames[asset] = _asset_frame(panel, asset)
        ab = frames[asset]

        entry = ev["entry_ref_price"]
        atr = ev["atr_at_confirm"]
        exec_ts = ev["execution_ts"]
        if pd.isna(entry) or pd.isna(atr) or atr <= 0 or pd.isna(exec_ts):
            continue
        try:
            pos = ab.index.get_loc(exec_ts)
        except KeyError:
            continue
        if isinstance(pos, slice):
            pos = pos.start

        high = ab["high"].to_numpy(np.float64)
        low = ab["low"].to_numpy(np.float64)
        close = ab["close"].to_numpy(np.float64)
        entry_f = float(entry)
        atr_f = float(atr)
        n = len(ab)

        for h in horizons:
            end = pos + h  # window is [pos, pos+h)
            if end > n:
                continue
            window_hi = high[pos:end]
            window_lo = low[pos:end]
            final = close[end - 1]
            rows.append({
                "event_id": ev["event_id"],
                "asset": asset,
                "confirmed_ts": ev["confirmed_ts"],
                "params_hash": ev["params_hash"],
                "horizon": int(h),
                "exec_pos": int(pos),
                "entry": entry_f,
                "ret": (final - entry_f) / entry_f,
                "net_atr": (final - entry_f) / atr_f,
                "up_atr": max(float(window_hi.max()) - entry_f, 0.0) / atr_f,
                "dn_atr": max(entry_f - float(window_lo.min()), 0.0) / atr_f,
            })

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols)


#: Keys defining one "signal episode" unit: repeated firings of the *same* event
#: (same ``event_id`` *and* ``params_hash``) on the *same* asset are one
#: persistent state, not N independent events. Overlap/uniqueness are always
#: computed within these keys.
_EPISODE_KEYS: tuple[str, ...] = ("asset", "event_id", "params_hash", "horizon")


def tag_episodes(paths: pd.DataFrame) -> pd.DataFrame:
    """Tag overlapping detections with an uniqueness weight and an episode flag.

    Overlapping forward windows mean detections are not independent samples (see
    ``docs/guides/research/event-registry.md`` and López de Prado, *Advances in
    Financial ML*, ch. 4). Within each :data:`_EPISODE_KEYS` group this adds:

    * ``concurrency`` — how many of the group's windows cover this event's
      execution bar's span (``>= 1``; itself counts).
    * ``weight`` — average uniqueness ``mean(1 / concurrency)`` over the event's
      ``[exec_pos, exec_pos + horizon)`` window. The group's ``weight`` sum is its
      effective independent-observation count (ESS).
    * ``is_episode`` — ``True`` for first-appearance entries under a
      ``cooldown = horizon`` no-re-entry rule: the execution model where you enter
      once when the signal first appears and hold ``horizon`` bars rather than
      re-entering every bar the condition persists.

    Parameters
    ----------
    paths
        :func:`forward_paths` output (must carry ``exec_pos`` and ``horizon``).

    Returns
    -------
    pandas.DataFrame
        ``paths`` with ``concurrency``, ``weight`` and ``is_episode`` added.
    """
    if paths.empty:
        return paths.assign(concurrency=pd.Series(dtype=float),
                            weight=pd.Series(dtype=float),
                            is_episode=pd.Series(dtype=bool))

    parts: list[pd.DataFrame] = []
    for (_, _, _, h), grp in paths.groupby(list(_EPISODE_KEYS), sort=False):
        g = grp.sort_values("exec_pos")
        pos = g["exec_pos"].to_numpy(np.int64)
        horizon = int(h)
        lo = int(pos.min())
        cover = np.zeros(int(pos.max()) + horizon - lo, dtype=np.int64)
        for p in pos:
            cover[p - lo:p - lo + horizon] += 1
        own = np.array([float(cover[p - lo]) for p in pos])  # concurrency at entry bar
        weight = np.array([float(np.mean(1.0 / cover[p - lo:p - lo + horizon])) for p in pos])
        is_episode = np.zeros(len(pos), dtype=bool)
        last = -(10**9)
        for i, p in enumerate(pos):
            if p >= last + horizon:
                is_episode[i] = True
                last = int(p)
        parts.append(g.assign(concurrency=own, weight=weight, is_episode=is_episode))
    return pd.concat(parts).loc[paths.index]


def outcome_profile(
    paths: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("event_id", "params_hash"),
    steady_band: float = 0.5,
    decide_tol: float = 0.05,
    consolidate: bool = True,
) -> pd.DataFrame:
    """Data-driven up / down / steady profile per event and horizon.

    Answers "if this event happens, does price move up, down, or stay steady?"
    using only realised paths — the geometric branch never signs anything. Each
    (episode) net move (in ATR units) is classified:

    * **up**  — ``net_atr > steady_band``
    * **down** — ``net_atr < -steady_band``
    * **steady** — ``|net_atr| <= steady_band`` (no actionable move)

    The suggested side is inferred from the *mean* net move over episodes:
    ``"long"`` if it exceeds ``decide_tol`` ATR, ``"short"`` if below
    ``-decide_tol``, else ``"neutral"``.

    Overlapping detections are not independent. With ``consolidate=True``
    (default) the fractions/means are computed over **episodes** (first-appearance
    entries, one per persistent signal — see :func:`tag_episodes`); ``n`` is the
    episode count, ``n_raw`` the raw detections, and ``ess`` the uniqueness-based
    effective sample size as a cross-check.

    Parameters
    ----------
    paths
        :func:`forward_paths` output.
    by
        Grouping keys. Default ``("event_id", "params_hash")`` keeps each
        parameterisation of each geometric branch separate (so pooled variants
        never mix). Pass ``("event_id",)`` to pool parameterisations and judge an
        event id as a whole.
    steady_band
        Half-width (ATR units) of the "steady" zone for the up/down/steady split.
    decide_tol
        Mean-net-ATR magnitude below which the suggested side is ``"neutral"``.
    consolidate
        If ``True`` (default), aggregate over episodes; if ``False``, over all raw
        detections (overstating sample size).

    Returns
    -------
    pandas.DataFrame
        One row per group × horizon with ``n`` (episodes if consolidated),
        ``n_raw``, ``ess``, ``frac_up``, ``frac_down``, ``frac_steady``,
        ``mean_net_atr``, ``median_net_atr``, ``mean_ret``, ``suggested``.
    """
    group_cols = [*by, "horizon"]
    out_cols = [*group_cols, "n", "n_raw", "ess", "frac_up", "frac_down",
                "frac_steady", "mean_net_atr", "median_net_atr", "mean_ret", "suggested"]
    if paths.empty:
        return pd.DataFrame(columns=out_cols)

    tagged = tag_episodes(paths)

    def _agg(g: pd.DataFrame) -> pd.Series:
        sample = g[g["is_episode"]] if consolidate else g
        net = sample["net_atr"].to_numpy(np.float64)
        mean_net = float(net.mean())
        suggested = "long" if mean_net > decide_tol else "short" if mean_net < -decide_tol else "neutral"
        return pd.Series({
            "n": len(sample),
            "n_raw": len(g),
            "ess": float(g["weight"].sum()),
            "frac_up": float((net > steady_band).mean()),
            "frac_down": float((net < -steady_band).mean()),
            "frac_steady": float((np.abs(net) <= steady_band).mean()),
            "mean_net_atr": mean_net,
            "median_net_atr": float(np.median(net)),
            "mean_ret": float(sample["ret"].mean()),
            "suggested": suggested,
        })

    out = tagged.groupby(list(group_cols), as_index=False).apply(_agg, include_groups=False)
    return out.sort_values(group_cols).reset_index(drop=True)[out_cols]


def return_distribution(
    paths: pd.DataFrame,
    *,
    by: tuple[str, ...] = ("event_id", "params_hash"),
    consolidate: bool = True,
) -> pd.DataFrame:
    """Full distribution of forward returns and excursions per group and horizon.

    Means hide heavy tails; this reports percentiles instead. For each group:
    return p05 / p25 / median / p75 / p95 and the mean/p95 of the up and down
    excursions (ATR units). Like :func:`outcome_profile`, ``consolidate=True``
    (default) computes over episodes (one per persistent signal), reporting
    ``n`` (episodes), ``n_raw`` and ``ess``.

    Parameters
    ----------
    paths
        :func:`forward_paths` output.
    by
        Grouping keys (default ``("event_id", "params_hash")``).
    consolidate
        If ``True`` (default), aggregate over episodes; else over raw detections.

    Returns
    -------
    pandas.DataFrame
        One row per group × horizon.
    """
    group_cols = [*by, "horizon"]
    out_cols = [*group_cols, "n", "n_raw", "ess", "ret_p05", "ret_p25",
                "ret_median", "ret_p75", "ret_p95", "up_atr_mean", "up_atr_p95",
                "dn_atr_mean", "dn_atr_p95"]
    if paths.empty:
        return pd.DataFrame(columns=out_cols)

    tagged = tag_episodes(paths)

    def _agg(g: pd.DataFrame) -> pd.Series:
        sample = g[g["is_episode"]] if consolidate else g
        ret = sample["ret"].to_numpy(np.float64)
        up = sample["up_atr"].to_numpy(np.float64)
        dn = sample["dn_atr"].to_numpy(np.float64)
        return pd.Series({
            "n": len(sample),
            "n_raw": len(g),
            "ess": float(g["weight"].sum()),
            "ret_p05": float(np.quantile(ret, 0.05)),
            "ret_p25": float(np.quantile(ret, 0.25)),
            "ret_median": float(np.quantile(ret, 0.50)),
            "ret_p75": float(np.quantile(ret, 0.75)),
            "ret_p95": float(np.quantile(ret, 0.95)),
            "up_atr_mean": float(up.mean()),
            "up_atr_p95": float(np.quantile(up, 0.95)),
            "dn_atr_mean": float(dn.mean()),
            "dn_atr_p95": float(np.quantile(dn, 0.95)),
        })

    out = tagged.groupby(list(group_cols), as_index=False).apply(_agg, include_groups=False)
    return out.sort_values(group_cols).reset_index(drop=True)[out_cols]


def _daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close daily returns per symbol, aligned on the panel index."""
    close = panel.xs("close", level=0, axis=1)
    return close.pct_change(fill_method=None)


def _drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve (a non-positive number)."""
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def _stats(strat: np.ndarray, bench: np.ndarray, n_days: int, in_market: float) -> dict[str, float]:
    """CAGR / max-drawdown / Sharpe / correlation for a strategy vs benchmark."""
    eq_s = np.cumprod(1.0 + strat)
    eq_b = np.cumprod(1.0 + bench)
    years = n_days / _TRADING_DAYS
    cagr_s = float(eq_s[-1] ** (1.0 / years) - 1.0) if years > 0 and eq_s[-1] > 0 else float("nan")
    cagr_b = float(eq_b[-1] ** (1.0 / years) - 1.0) if years > 0 and eq_b[-1] > 0 else float("nan")
    sd = strat.std()
    sharpe = float(strat.mean() / sd * np.sqrt(_TRADING_DAYS)) if sd > 0 else float("nan")
    corr = (
        float(np.corrcoef(strat, bench)[0, 1])
        if strat.std() > 0 and bench.std() > 0
        else float("nan")
    )
    return {
        "cagr": cagr_s,
        "max_drawdown": _drawdown(eq_s),
        "sharpe": sharpe,
        "corr_to_bh": corr,
        "in_market": in_market,
        "bh_cagr": cagr_b,
        "bh_max_drawdown": _drawdown(eq_b),
    }


def _resolve_side(obs: pd.DataFrame, panel: pd.DataFrame, horizon: int, side: str) -> float:
    """Map a side request to a ±1 sign; ``"auto"`` infers it from the data.

    For ``"auto"`` the sign is the *episode* mean forward return, not the raw
    mean over all overlapping detections. Overlapping firings of one persistent
    signal are not independent (see :func:`tag_episodes`), so a raw ``ret.mean()``
    lets a single long-lived signal dominate the decision. Taking the mean over
    first-appearance episodes (one per ``cooldown = horizon`` no-re-entry window)
    weights every distinct signal once. Falls back to long (``+1``) when there is
    no usable episode return.
    """
    if side == "long":
        return 1.0
    if side == "short":
        return -1.0
    if side == "auto":
        paths = forward_paths(obs, panel, horizons=(horizon,))
        if paths.empty:
            return 1.0
        episode_ret = tag_episodes(paths).loc[lambda t: t["is_episode"], "ret"]
        if episode_ret.empty or bool(episode_ret.isna().all()):
            return 1.0
        return 1.0 if float(episode_ret.mean()) >= 0 else -1.0
    msg = f"unknown side: {side!r}; valid: 'long', 'short', 'auto'"
    raise ValueError(msg)


def event_portfolio(
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int,
    side: str = "auto",
    cost_bps: float = 6.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Daily-position equity curve for an event set versus buy-and-hold.

    Each event holds its asset for ``horizon`` bars from the execution bar, signed
    by ``side`` (**not** by the textbook label); overlapping events on one asset
    cap the position at ``±1``. Each day the strategy is equal-weight across the
    assets with an active position (cash otherwise).

    The fill is the **execution-bar open** (``open[execution_ts]``, per the
    registry contract), so the first held bar of every position books its
    ``open → close`` return, not the untraded ``prev_close → close`` overnight gap;
    subsequent held bars book the normal close-to-close return. A single long
    episode's total return at horizon ``h`` therefore equals its
    :func:`forward_paths` ``ret`` (``open[exec] → close[exec + h - 1]``).

    Round-trip cost ``cost_bps`` is charged on changes in the *actual* portfolio
    weights ``w = position / n_active`` — including the full exit when a position
    drops to cash and the reweighting churn as the active set changes — so no
    trade goes uncharged. The benchmark is equal-weight, fully-invested
    buy-and-hold over the same assets and dates.

    Parameters
    ----------
    obs
        Observation frame; typically one event group (filter before calling).
    panel
        The ``(field, symbol)`` panel.
    horizon
        Holding length in bars.
    side
        ``"long"``, ``"short"``, or ``"auto"`` (infer the side from the data's
        mean forward return). The default ``"auto"`` embodies "let the data
        decide", not the geometric label.
    cost_bps
        Round-trip transaction cost in basis points (default 6).

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        ``(equity, stats)`` — ``equity`` indexed by date with ``strategy`` and
        ``benchmark`` columns (both start at 1.0); ``stats`` holds ``cagr``,
        ``max_drawdown``, ``sharpe``, ``corr_to_bh``, ``in_market`` (fraction of
        days with ≥1 symbol held), ``avg_breadth`` (mean fraction of symbols
        held), ``avg_active_when_in`` (mean symbols held on active days),
        ``bh_cagr``, ``bh_max_drawdown`` plus ``side`` (the resolved ±1 sign).
        Empty obs yields an empty equity frame and NaN stats.
    """
    rets = _daily_returns(panel)
    dates = rets.index
    symbols = list(rets.columns)
    n = len(dates)
    one_way = (cost_bps / 2.0) / 10_000.0
    sign = _resolve_side(obs, panel, horizon, side) if not obs.empty else 1.0

    pos = pd.DataFrame(0.0, index=dates, columns=symbols)
    if not obs.empty:
        for _, ev in obs.iterrows():
            asset = str(ev["asset"])
            if asset not in pos.columns or pd.isna(ev["execution_ts"]):
                continue
            try:
                start = dates.get_loc(ev["execution_ts"])
            except KeyError:
                continue
            if isinstance(start, slice):
                start = start.start
            end = min(start + horizon, n)
            pos.iloc[start:end, pos.columns.get_loc(asset)] += sign

    pos = pos.clip(-1.0, 1.0)
    pos_arr = pos.to_numpy(np.float64)
    ret_raw = rets.to_numpy(np.float64)

    # FIX 2: the first held bar of a position is filled at open[execution_ts], so
    # it books open->close, not the untraded prev_close->close overnight gap.
    # Build a per-cell return whose entry bars (0 -> held transitions) carry the
    # open->close return and all later held bars the normal close-to-close return.
    open_df = pd.DataFrame(panel.xs("open", level=0, axis=1))
    close_df = pd.DataFrame(panel.xs("close", level=0, axis=1))
    open_arr = open_df.reindex(columns=symbols).to_numpy(np.float64)
    close_arr = close_df.reindex(columns=symbols).to_numpy(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        open_to_close = close_arr / open_arr - 1.0
    held = pos_arr != 0.0
    prev_held = np.zeros_like(held)
    prev_held[1:] = held[:-1]
    starts = held & ~prev_held  # 0 -> held transitions (incl. a day-0 entry)
    ret_arr = np.where(starts, open_to_close, ret_raw)
    ret_arr = np.nan_to_num(ret_arr, nan=0.0)

    n_active = (pos_arr != 0.0).sum(axis=1)
    denom = np.maximum(n_active, 1)
    gross = np.where(n_active > 0, (pos_arr * ret_arr).sum(axis=1) / denom, 0.0)

    # FIX 3: charge cost on changes in the actual portfolio weights
    # w = position / n_active (equal-weight among active names, 0 when flat). This
    # captures the full exit (w -> 0) and the reweighting churn as the active set
    # changes, on every day — not just entry bars, and never divided away.
    weights = pos_arr / denom[:, None]
    weight_turnover = np.abs(np.diff(weights, axis=0, prepend=0.0))
    cost = (weight_turnover * one_way).sum(axis=1)
    strat = gross - cost

    valid = ~np.isnan(ret_raw)
    valid_counts = valid.sum(axis=1)
    bench = np.where(valid_counts > 0, np.nansum(ret_raw, axis=1) / np.maximum(valid_counts, 1), 0.0)
    in_market = float((n_active > 0).mean()) if n > 0 else float("nan")
    n_symbols = len(symbols)
    active_days = n_active > 0

    if n > 0:
        stats = _stats(strat, bench, n, in_market)
        # Breadth separates "≥1 symbol active" (in_market) from real exposure:
        # avg_breadth = mean fraction of symbols held; avg_active_when_in = on an
        # active day, how many symbols are held (1 means a concentrated single-name bet).
        stats["avg_breadth"] = float((n_active / n_symbols).mean())
        stats["avg_active_when_in"] = (
            float(n_active[active_days].mean()) if active_days.any() else float("nan")
        )
    else:
        keys = ("cagr", "max_drawdown", "sharpe", "corr_to_bh", "in_market",
                "bh_cagr", "bh_max_drawdown", "avg_breadth", "avg_active_when_in")
        stats = dict.fromkeys(keys, float("nan"))
    stats["side"] = sign

    equity = pd.DataFrame(
        {"strategy": np.cumprod(1.0 + strat), "benchmark": np.cumprod(1.0 + bench)},
        index=dates,
    )
    return equity, stats


def portfolio_by_event(
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int,
    by: tuple[str, ...] = ("event_id", "params_hash"),
    side: str = "auto",
    cost_bps: float = 6.0,
) -> pd.DataFrame:
    """Run :func:`event_portfolio` per group and stack the stats.

    Parameters
    ----------
    obs
        Observation frame.
    panel
        The ``(field, symbol)`` panel.
    horizon
        Holding length in bars.
    by
        Grouping keys (default ``("event_id", "params_hash")``).
    side
        Forwarded to :func:`event_portfolio` (``"auto"`` decides per group).
    cost_bps
        Round-trip cost in basis points.

    Returns
    -------
    pandas.DataFrame
        One row per group with the :func:`event_portfolio` stats plus the group
        keys and an ``n_events`` count. Empty obs yields an empty frame.
    """
    out_cols = [*by, "n_events", "side", "cagr", "max_drawdown", "sharpe",
                "corr_to_bh", "in_market", "avg_breadth", "avg_active_when_in",
                "bh_cagr", "bh_max_drawdown"]
    if obs.empty:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, object]] = []
    for keys, group in obs.groupby(list(by)):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        _, stats = event_portfolio(group, panel, horizon=horizon, side=side, cost_bps=cost_bps)
        rows.append({**dict(zip(by, key_tuple, strict=True)), "n_events": len(group), **stats})
    return pd.DataFrame(rows, columns=out_cols).sort_values(list(by)).reset_index(drop=True)


def evidence_table(
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    by: tuple[str, ...] = ("event_id", "params_hash"),
    steady_band: float = 0.5,
    cost_bps: float = 6.0,
) -> pd.DataFrame:
    """The one evidence-driven summary: what each event does + its portfolio impact.

    Combines, at a single ``horizon`` and per group, the :func:`outcome_profile`
    answer ("does price go up / down / steady, and which side does the data
    suggest?") with the :func:`portfolio_by_event` answer ("what does acting on it
    — side decided by the data, ``side="auto"`` — do to a portfolio versus
    buy-and-hold?"). Nothing here uses the geometric branch to pick a side and
    nothing applies a pass/fail threshold — it is purely descriptive evidence.

    Parameters
    ----------
    obs
        Observation frame (e.g. :func:`fundcloud.research.events.scan_variants`).
    panel
        The ``(field, symbol)`` panel.
    horizon
        The forward/holding horizon in bars (default 20).
    by
        Grouping keys (default ``("event_id", "params_hash")``; the event_id's
        ``_up`` / ``_dn`` suffix is a geometric branch, never a trade decision).
    steady_band
        Half-width (ATR units) of the "steady" zone, forwarded to
        :func:`outcome_profile`.
    cost_bps
        Round-trip cost in basis points, forwarded to :func:`portfolio_by_event`.

    Returns
    -------
    pandas.DataFrame
        One row per group with ``horizon``, the outcome columns (``n`` =
        episodes, ``n_raw``, ``ess``, ``frac_up``, ``frac_down``,
        ``frac_steady``, ``mean_net_atr``, ``suggested``) and the portfolio
        columns (``side``, ``cagr``,
        ``max_drawdown``, ``sharpe``, ``corr_to_bh``, ``avg_breadth``,
        ``bh_cagr``, ``bh_max_drawdown``). Empty obs yields an empty frame.
    """
    out_cols = [*by, "horizon", "n", "n_raw", "ess", "frac_up", "frac_down",
                "frac_steady", "mean_net_atr", "suggested", "side", "cagr",
                "max_drawdown", "sharpe", "corr_to_bh", "avg_breadth",
                "bh_cagr", "bh_max_drawdown"]
    if obs.empty:
        return pd.DataFrame(columns=out_cols)

    paths = forward_paths(obs, panel, horizons=(horizon,))
    profile = outcome_profile(paths, by=by, steady_band=steady_band)
    pf = portfolio_by_event(obs, panel, horizon=horizon, by=by, side="auto", cost_bps=cost_bps)

    keep_profile = [*by, "horizon", "n", "n_raw", "ess", "frac_up", "frac_down",
                    "frac_steady", "mean_net_atr", "suggested"]
    keep_pf = [*by, "side", "cagr", "max_drawdown", "sharpe", "corr_to_bh",
               "avg_breadth", "bh_cagr", "bh_max_drawdown"]
    merged = profile[keep_profile].merge(pf[keep_pf], on=list(by), how="outer")
    return merged.sort_values(list(by)).reset_index(drop=True)[out_cols]


def variant_leaderboard(
    obs: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    horizon: int = 20,
    event_id: str | None = None,
    sort_by: str = "sharpe",
    ascending: bool = False,
    steady_band: float = 0.5,
    cost_bps: float = 6.0,
) -> pd.DataFrame:
    """Rank parameter variants of an event with their params decoded into columns.

    This is the **legible** answer to "scan event/param1 and event/param2 and
    evaluate them separately": the evidence layer already keeps variants apart by
    ``params_hash``, but a hash is unreadable. This calls :func:`evidence_table`
    (one row per ``(event_id, params_hash)`` — the existing math, not recomputed)
    and left-joins :func:`fundcloud.research.events.decode_params` so the swept
    parameters appear as their own columns next to the metrics, then sorts by a
    chosen metric. A human reads ``z_body=1.0 → sharpe 0.3`` vs ``z_body=1.5 →
    sharpe 0.6`` directly off the table.

    Parameters
    ----------
    obs
        Observation frame (e.g. :func:`fundcloud.research.events.scan_variants`),
        typically pooling several variants.
    panel
        The ``(field, symbol)`` panel the events were detected on.
    horizon
        Forward/holding horizon in bars (default 20).
    event_id
        If given, restrict to this ``event_id`` so the parameter columns are
        homogeneous (every variant of one event shares the same param keys). When
        ``None`` the table pools detectors and the non-shared param columns read
        ``NaN`` for variants that do not use them — honest, not hidden.
    sort_by
        Metric column to rank by (default ``"sharpe"``). Raises if absent.
    ascending
        Sort direction (default ``False`` → best metric on top); ``NaN`` keys sink.
    steady_band
        Half-width (ATR units) of the steady zone, forwarded to the outcome view.
    cost_bps
        Round-trip cost in basis points, forwarded to the portfolio view.

    Returns
    -------
    pandas.DataFrame
        One row per ``(event_id, params_hash)``, column-ordered
        ``event_id, params_hash, <decoded params>, <evidence + portfolio metrics>``
        and sorted by ``sort_by``. Empty obs yields the empty
        :func:`evidence_table` frame (the parameter columns are unknowable with no
        rows).

    Raises
    ------
    ValueError
        If ``sort_by`` is not a column of the assembled table.
    """
    by = ("event_id", "params_hash")
    scoped = obs if event_id is None else obs[obs["event_id"] == event_id]
    table = evidence_table(scoped, panel, horizon=horizon, by=by,
                           steady_band=steady_band, cost_bps=cost_bps)
    if table.empty:
        return table

    key_cols = ["event_id", "params_hash"]
    metric_cols = [c for c in table.columns if c not in key_cols]

    decoded = decode_params(scoped)
    # A parameter can share a name with a metric column — e.g. the NR lookback
    # ``n`` collides with the episode-count metric ``n``. Prefix only the
    # colliding param(s) with ``param_`` so the join stays unambiguous and no
    # metric is shadowed; every other param keeps its plain, readable name.
    collisions = {c for c in decoded.columns if c in set(metric_cols)}
    if collisions:
        decoded = decoded.rename(columns={c: f"param_{c}" for c in collisions})
    param_cols = [c for c in decoded.columns if c != "params_hash"]
    merged = table.merge(decoded, on="params_hash", how="left")
    merged = merged[[*key_cols, *param_cols, *metric_cols]]

    if sort_by not in merged.columns:
        msg = f"unknown sort_by: {sort_by!r}; valid metric columns: {metric_cols}"
        raise ValueError(msg)
    return merged.sort_values(sort_by, ascending=ascending, na_position="last").reset_index(
        drop=True
    )
