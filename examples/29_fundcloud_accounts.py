"""29 — Analyse a FundCloud-tracked fund (NAV, positions, capital flows).

Pulls NAV and capital flows from the FundCloud platform, builds a
Portfolio with correct distribution-adjusted total-return semantics,
and renders a Tearsheet against an SPY benchmark.

Trader question answered: "What's my FundCloud fund's investor-level
total return — and how does it stack up against buy-and-hold SPY?"

Run:
    uv add 'fundcloud[reports]'           # core ships with FundCloud support
    export FUNDCLOUD_API_KEY=fc_live_...
    uv run python examples/29_fundcloud_accounts.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import fundcloud as fc

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    if not os.environ.get("FUNDCLOUD_API_KEY"):
        print(
            "skip: FUNDCLOUD_API_KEY not set. "
            "Mint one at https://app.fundcloud.com/settings → API Key."
        )
        return 0

    src = fc.accounts.FundCloud()

    # 1. Browse available funds / accounts so the user can confirm what's visible.
    funds = src.list_funds()
    print(f"Funds visible ({len(funds)}):")
    print(funds[["fund_id", "name", "currency", "aum", "status"]].to_string(index=False))
    print()

    if funds.empty:
        print("No funds visible to this API key — nothing to analyse.")
        return 0

    accounts_df = src.list_accounts()
    print(f"Linked accounts ({len(accounts_df)}):")
    cols = ["account_id", "account_name", "fund_name", "currency"]
    print(accounts_df[cols].to_string(index=False))
    print()

    # 2. Build a Portfolio for each fund, with the default
    #    (nav_per_share + total_return, distributions added back).
    for _, fund in funds.iterrows():
        fund_id = fund["fund_id"]
        display = fund["short_name"] or fund["name"]

        pf = src.to_portfolio(fund_id=fund_id, name=display)
        print(f"--- {display} ({fund_id}) ---")
        print(
            f"  Periods: {len(pf.returns)}  "
            f"Sharpe: {pf.sharpe():.2f}  "
            f"MaxDD: {pf.max_drawdown():.2%}"
        )

        # 3. Try for a benchmark in the fund's currency.
        try:
            spy = fc.data.FundCloud("SPY", period="2Y").read(start=pf.returns.index[0])
            benchmark = spy.xs("close", level=0, axis=1).iloc[:, 0]
        except Exception as e:  # example tolerates any lookup failure
            print(f"  (benchmark skipped: {e})")
            benchmark = None

        out_path = OUT / f"fundcloud_{display.replace(' ', '_').lower()}.html"
        fc.reports.Tearsheet(pf, benchmark=benchmark).render_html(out_path)
        print(f"  Tearsheet → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
