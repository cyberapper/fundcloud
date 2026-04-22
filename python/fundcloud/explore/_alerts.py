"""Rule-based alerts for :func:`profile` and :func:`compare`.

Kept as plain dataclasses so the renderer can group by severity and the
test suite can assert on ``code`` strings without string-matching prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["Alert", "compare_alerts", "profile_alerts"]


Severity = Literal["info", "warning", "critical"]


@dataclass(frozen=True)
class Alert:
    severity: Severity
    code: str
    message: str
    columns: tuple[str, ...] = ()


_HIGH_CORR = 0.95
_HIGH_MISSING_PCT = 30.0
_HIGH_ZEROS_PCT = 50.0
_HIGH_SKEW = 2.0
_HIGH_KURTOSIS = 7.0
_QUASI_CONST_PCT = 95.0
_KS_CRITICAL = 0.2
_MEAN_SHIFT_SIGMA = 1.0


def profile_alerts(stats: pd.DataFrame, corr: pd.DataFrame | None) -> list[Alert]:
    """Evaluate profile-report rules against ``per_column_stats`` output."""
    alerts: list[Alert] = []
    for col, row in stats.iterrows():
        name = str(col)
        std = row.get("std", float("nan"))
        if pd.notna(std) and std == 0.0:
            alerts.append(
                Alert("critical", "zero_variance", f"Column {name!r} is constant (std=0).", (name,))
            )
        missing_pct = row.get("missing_pct", float("nan"))
        if pd.notna(missing_pct) and missing_pct > _HIGH_MISSING_PCT:
            alerts.append(
                Alert(
                    "warning",
                    "high_missing",
                    f"Column {name!r} is {missing_pct:.1f}% missing.",
                    (name,),
                )
            )
        zeros_pct = row.get("zeros_pct", float("nan"))
        if pd.notna(zeros_pct) and zeros_pct > _HIGH_ZEROS_PCT:
            alerts.append(
                Alert(
                    "info",
                    "excessive_zeros",
                    f"Column {name!r} is {zeros_pct:.1f}% zeros.",
                    (name,),
                )
            )
        skew = row.get("skew", float("nan"))
        if pd.notna(skew) and abs(skew) > _HIGH_SKEW:
            alerts.append(
                Alert("info", "high_skew", f"Column {name!r} has skew {skew:.2f}.", (name,))
            )
        kurt = row.get("kurtosis", float("nan"))
        if pd.notna(kurt) and abs(kurt) > _HIGH_KURTOSIS:
            alerts.append(
                Alert(
                    "info", "high_kurtosis", f"Column {name!r} has kurtosis {kurt:.2f}.", (name,)
                )
            )
        mode_freq = row.get("mode_freq", float("nan"))
        if pd.notna(mode_freq) and mode_freq > _QUASI_CONST_PCT:
            alerts.append(
                Alert(
                    "warning",
                    "quasi_constant",
                    f"Column {name!r} mode covers {mode_freq:.1f}% of non-null rows.",
                    (name,),
                )
            )
        inf_pct = row.get("inf_pct", float("nan"))
        if pd.notna(inf_pct) and inf_pct > 0.0:
            alerts.append(
                Alert(
                    "warning",
                    "infinite_values",
                    f"Column {name!r} contains {inf_pct:.2f}% ±inf.",
                    (name,),
                )
            )
    alerts.extend(_high_correlation_alerts(corr))
    return alerts


def _high_correlation_alerts(corr: pd.DataFrame | None) -> list[Alert]:
    if corr is None or corr.empty or corr.shape[0] < 2:
        return []
    abs_corr = corr.abs().to_numpy(copy=True, dtype=float)
    np.fill_diagonal(abs_corr, 0.0)
    names = [str(c) for c in corr.columns]
    alerts: list[Alert] = []
    seen: set[frozenset[str]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = abs_corr[i, j]
            if np.isnan(v) or v <= _HIGH_CORR:
                continue
            pair = frozenset({names[i], names[j]})
            if pair in seen:
                continue
            seen.add(pair)
            alerts.append(
                Alert(
                    "warning",
                    "high_correlation",
                    f"|corr({names[i]}, {names[j]})| = {v:.3f}.",
                    (names[i], names[j]),
                )
            )
    return alerts


def compare_alerts(
    drift: pd.DataFrame,
    cols_a: list[str],
    cols_b: list[str],
    *,
    target_shifts: dict[str, float] | None = None,
) -> list[Alert]:
    """Evaluate compare-report rules: schema changes, drift, target shift."""
    a_set = set(cols_a)
    b_set = set(cols_b)
    alerts: list[Alert] = []
    for c in sorted(a_set - b_set):
        alerts.append(
            Alert("warning", "column_removed", f"Column {c!r} present in a, missing in b.", (c,))
        )
    for c in sorted(b_set - a_set):
        alerts.append(Alert("info", "column_added", f"Column {c!r} new in b.", (c,)))
    if not drift.empty:
        for col, row in drift.iterrows():
            name = str(col)
            ks = row.get("ks_stat", float("nan"))
            if pd.notna(ks) and ks > _KS_CRITICAL:
                alerts.append(
                    Alert(
                        "critical",
                        "distribution_shift",
                        f"{name!r}: KS={ks:.3f} (p={row.get('ks_pvalue', float('nan')):.3g}).",
                        (name,),
                    )
                )
            std_a = row.get("std_a", float("nan"))
            mean_shift = row.get("mean_shift", float("nan"))
            if pd.notna(std_a) and std_a > 0.0 and pd.notna(mean_shift):
                sigmas = mean_shift / std_a
                if abs(sigmas) > _MEAN_SHIFT_SIGMA:
                    alerts.append(
                        Alert(
                            "critical",
                            "mean_shift",
                            f"{name!r}: mean shift {mean_shift:+.3g} ({sigmas:+.2f} std_a).",
                            (name,),
                        )
                    )
    if target_shifts:
        for col, delta in target_shifts.items():
            if pd.notna(delta) and abs(delta) > 0.2:
                alerts.append(
                    Alert(
                        "warning",
                        "target_correlation_shift",
                        f"{col!r}: d_corr(target)={delta:+.3f}.",
                        (col,),
                    )
                )
    return alerts
