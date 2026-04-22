"""Two-sample drift metrics for :func:`compare`.

Each shared numeric column yields a row: means, stds, KS two-sample
statistic (+p-value), Wasserstein (earth-mover) distance, and
missing-rate delta. Everything runs on ``scipy.stats``; no external
lib needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

__all__ = ["drift_table"]


def drift_table(a: pd.DataFrame, b: pd.DataFrame, shared: list[str]) -> pd.DataFrame:
    """Per-column drift stats for the columns present in both frames."""
    rows: list[dict[str, float | str]] = []
    for col in shared:
        col_a = a[col]
        col_b = b[col]
        if not (pd.api.types.is_numeric_dtype(col_a) and pd.api.types.is_numeric_dtype(col_b)):
            continue
        ac = col_a.dropna().to_numpy(dtype=float)
        bc = col_b.dropna().to_numpy(dtype=float)
        ac = ac[np.isfinite(ac)]
        bc = bc[np.isfinite(bc)]
        if len(ac) < 2 or len(bc) < 2:
            continue
        ks = ks_2samp(ac, bc, alternative="two-sided")
        w = float(wasserstein_distance(ac, bc))
        mean_a = float(ac.mean())
        mean_b = float(bc.mean())
        std_a = float(ac.std(ddof=1))
        std_b = float(bc.std(ddof=1))
        rows.append(
            {
                "column": str(col),
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_shift": mean_b - mean_a,
                "std_a": std_a,
                "std_b": std_b,
                "std_ratio": std_b / std_a if std_a else float("nan"),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein": w,
                "missing_delta_pct": float(
                    (col_b.isna().mean() - col_a.isna().mean()) * 100
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "mean_a",
                "mean_b",
                "mean_shift",
                "std_a",
                "std_b",
                "std_ratio",
                "ks_stat",
                "ks_pvalue",
                "wasserstein",
                "missing_delta_pct",
            ]
        )
    return pd.DataFrame(rows).set_index("column")
