"""30 — Analyse an IB brokerage account from a Flex Query CSV export.

Loads a Flex Query CSV (single file or list of files), builds a
Portfolio with AUM-basis Modified Dietz semantics (the GIPS-standard
TWR for a brokerage account), and renders a Tearsheet against an SPY
benchmark.

Trader question answered: "What's my IB account's investment-only
return — i.e., stripping out my own deposit and withdrawal timing?"

Run:
    uv add 'fundcloud[viz,reports]'      # core ships with IB support
    uv run python examples/30_ib_flex_csv.py path/to/export.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import fundcloud as fc

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        # Default to the repo's anonymised sample if present.
        candidate = HERE.parent / "temp" / "ib_fundcloud_example.csv"
        if not candidate.exists():
            print(
                "usage: python examples/30_ib_flex_csv.py <flex_export.csv>",
                file=sys.stderr,
            )
            return 1
        path = candidate
        print(f"(no path given — using sample at {path})")
    else:
        path = Path(argv[1])

    src = fc.accounts.IB(path)

    funds = src.list_funds()
    print(f"Accounts visible ({len(funds)}):")
    print(funds[["fund_id", "currency", "aum", "inception_date"]].to_string(index=False))
    print()

    if funds.empty:
        print("No accounts in CSV — nothing to analyse.")
        return 0

    for _, row in funds.iterrows():
        acct = row["fund_id"]
        pf = src.to_portfolio(account_id=acct, name=acct)
        print(f"--- {acct} ---")
        print(
            f"  periods: {len(pf.returns)}  "
            f"sharpe: {pf.sharpe(periods_per_year=252):.2f}  "
            f"max_dd: {pf.max_drawdown():.2%}"
        )

        # Try a benchmark (best-effort — skip if YF/network unavailable).
        benchmark = None
        if pf.returns.empty:
            print("  (benchmark skipped: empty return series)")
        else:
            try:
                from fundcloud.data import YF

                spy = YF("SPY", start=str(pf.returns.index[0].date())).read()
                benchmark = spy.xs("close", level=0, axis=1).iloc[:, 0]
            except Exception as e:  # noqa: BLE001 — example tolerates lookup failure
                print(f"  (benchmark skipped: {type(e).__name__}: {e})")

        out_path = OUT / f"ib_{acct.replace(' ', '_').lower()}.html"
        fc.reports.Tearsheet(pf, benchmark=benchmark).render_html(out_path)
        print(f"  tearsheet → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
