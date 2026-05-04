---
title: Analysing a FundCloud fund
description: Pull your fund's NAV, positions, trades, and capital flows from the FundCloud platform — and run the same analytics you'd run on a backtest.
---

# Analysing a FundCloud fund

If you run a fund on the [FundCloud platform](https://app.fundcloud.com),
`fundcloud.accounts.FundCloud` lets you treat your live track record like
any other return series: tearsheets, drawdown tables, period returns,
benchmark comparisons, and skfolio-compatible attribution all work
exactly the same as they do for a backtested strategy. The only thing
that changes is where the NAV came from.

This guide is written for portfolio managers and traders. If you'd
rather see the protocol reference, head to [Reference → Accounts](../../reference/accounts.md).

---

## 60-second tour

```python
import fundcloud as fc

src = fc.accounts.FundCloud()              # reads FUNDCLOUD_API_KEY from env
pf  = src.to_portfolio()                   # one fund visible? you're done
fc.reports.Tearsheet(pf).render_html("my_fund.html")
```

That's the headline use case — three lines from "I have a FundCloud
account" to "here's a fully formatted PDF/HTML tearsheet I can send to
investors". Everything below is for when you need more control.

---

## Set up

`fundcloud.accounts.FundCloud` and `fundcloud.data.FundCloud` ship in
the core install — no extras flag required:

```bash
uv add fundcloud
```

Mint an API key in **Settings → API Key** at
[app.fundcloud.com](https://app.fundcloud.com/settings) and export it:

```bash
export FUNDCLOUD_API_KEY=fc_live_...
```

The provider picks the env var up automatically. Pass `api_key=`
explicitly if you'd rather load credentials from a vault or pin a
specific key for a given run.

The same key works for [`fundcloud.data.FundCloud`](../../reference/data.md)
(market-data backend), so you can pull benchmark prices through the
same credential.

!!! tip "Treat the key like a password"
    `fc_live_...` keys grant read access to every fund the user is
    a member of. Rotate every 90 days; revoke immediately if a key
    leaks. The library never logs the key — but be careful with
    `print(src._client.__dict__)` and similar debugging tricks.

---

## What's visible to me?

Before analysing, see what your credential can reach. The IDs and
numbers below are illustrative placeholders — yours will look
different.

```python
src.list_funds()
#                  fund_id                 name short_name currency  inception_date  status      aum  total_shares
#   <fund_id_alpha>     Global Multi-Asset Fund  GMAF      USD       2023-01-01  ACTIVE  2.50e+09     2.50e+07
#   <fund_id_beta>      Alpha Equity             AEQ       USD       2024-03-15  ACTIVE  1.20e+08     1.20e+06
#   <fund_id_gamma>     Macro Discretionary      MAC       USD       2024-09-01  ACTIVE  4.50e+08     4.50e+06

src.list_accounts()
#         account_id     account_name              fund_id     fund_name           ...  latest_nav
#   <acct_id_1>          IBKR-Primary       <fund_id_alpha>    Global Multi-Asset  ...   1.50e+09
#   <acct_id_2>          IBKR-Secondary     <fund_id_alpha>    Global Multi-Asset  ...   6.00e+08
#   <acct_id_3>          Binance-Spot       <fund_id_alpha>    Global Multi-Asset  ...   4.00e+08
```

`list_accounts(fund_id=...)` filters to one fund.

A FundCloud **fund** is the top-level book — what you'd report to investors. A FundCloud **account** is a linked broker / wallet under that book (an IBKR sub-account, a Binance API key, etc.). Most managers think in terms of "the fund's performance" — which is the default — but you can drill into a single account when you need to attribute return to a specific broker, sleeve, or strategy.

!!! note "Why are some accounts not in the list?"
    `list_accounts()` reads the FundCloud platform's most-recent
    aggregated NAV breakdown. Accounts that have been unlinked, or
    that received capital flows before any NAV records were
    published, may not appear. They still exist in `capital_flows()`
    history, and you can query them by `account_id` directly — the
    library will resolve the parent fund automatically.

---

## The headline workflow — fund-level tearsheet

For most portfolio managers, the daily question is "how did the fund
do?". One call:

```python
import fundcloud as fc

src = fc.accounts.FundCloud()
pf  = src.to_portfolio(fund_id="<fund_id_alpha>")
```

`pf` is a regular `Portfolio` object — the same one a backtest
produces. Every `.fc` accessor method, every metrics function, every
report renderer works on it:

=== "Quick metrics"

    ```python
    pf.summary()           # 11-metric snapshot
    pf.sharpe()            # single number
    pf.max_drawdown()      # peak-to-trough loss
    pf.cagr()              # compound annual growth rate
    pf.cvar(alpha=0.95)    # tail-risk measure
    ```

    Same surface as `Series.fc.metrics()` — the underlying
    implementation is the same.

=== "Full tearsheet"

    ```python
    spy = fc.data.FundCloud("SPY", period="2Y").read()["close"]
    fc.reports.Tearsheet(pf, benchmark=spy).render_html("apr_2026_review.html")
    ```

    Produces a rendered tearsheet with cumulative-return chart,
    drawdown table, monthly-return heatmap, period-return table, and
    benchmark comparison. Send to investors.

=== "Drawdown table"

    ```python
    pf.worst_drawdowns(top=5)
    #     Started     Recovered  Drawdown   Days
    #   2025-10-23   2026-01-08    -0.0418     78
    #   2025-12-04   2025-12-19    -0.0237     16
    #   2026-02-14   2026-03-02    -0.0193     17
    ```

    `Started → Recovered` is what investors and risk committees
    actually ask about. `pf.drawdown_details()` returns the full
    DataFrame including `valley` and `days_to_recover`.

=== "Period returns"

    ```python
    pf.period_returns(benchmark=spy)
    #                       SPY    Strategy
    # MTD                0.0148      0.0289
    # 3M                 0.0521      0.1147
    # YTD                0.0964      0.1832
    # 1Y (ann.)          0.1212      0.2401
    ```

    Standard monthly-report layout. 3Y/5Y/10Y rows are annualised
    — directly comparable to 1Y.

---

## Attribution — drilling into a single account

When the fund is up but you suspect one broker is dragging, drill in:

```python
pf_spot   = src.to_portfolio(account_id="<acct_id_3>")    # Binance spot
pf_ibkr_p = src.to_portfolio(account_id="<acct_id_1>")    # IBKR primary

pf_spot.cagr()              # how much did spot contribute?
pf_ibkr_p.max_drawdown()    # how deep was IBKR's worst patch?
```

Notice that we passed only `account_id` — no `fund_id`. The library
resolves the parent fund automatically (one extra request the first
time, then cached for the provider's lifetime).

If you want all accounts side-by-side as columns of a single DataFrame:

```python
import pandas as pd

panel = pd.DataFrame({
    a["account_name"]: src.to_portfolio(account_id=a["account_id"]).returns
    for _, a in src.list_accounts(fund_id="<fund_id_alpha>").iterrows()
})

panel.fc.summary()              # 11-metric table, one column per account
panel.fc.plot_cumulative()      # one line per account
```

!!! tip "Account-level Sharpe is a noisy diagnostic"
    A single account's Sharpe can swing wildly if it's small or
    recently funded. Use account-level breakdowns to find *direction*
    of attribution (which sleeves are dragging), not as standalone
    performance grades.

---

## How capital flows affect returns — and why the default is "right"

Investors put money in, take money out, and receive distributions.
These flows move AUM up and down without being investment performance.
If you naïvely take `pct_change()` of AUM, an injection looks like a
+10% day; a withdrawal looks like a -10% day. Neither is real return.

FundCloud tracks three flow types:

| Flow           | AUM | Shares    | NAV/share   | Counts as return? |
|----------------|-----|-----------|-------------|-------------------|
| `INJECTION`    | ↑   | ↑         | invariant   | No — new shares issued at current NAV |
| `WITHDRAWAL`   | ↓   | ↓         | invariant   | No — shares redeemed at current NAV |
| `DISTRIBUTION` | ↓   | unchanged | **↓**       | No — but per-share NAV understates total return unless added back |

`to_portfolio()` handles all three correctly **by default**. You don't
need to do anything — the next two sections explain *what* it's doing
in case you ever need to override.

### Default — NAV-per-share + total return (recommended)

```python
pf = src.to_portfolio(fund_id="<fund_id_alpha>")
# equivalent to:
pf = src.to_portfolio(fund_id="<fund_id_alpha>", basis="nav_per_share", method="total_return")
```

- Uses **per-share NAV** as the price series.
- Adds back **only** `DISTRIBUTION` flows, scaled to per-share
  (`distribution_amount / shares_outstanding_on_that_day`).
- Injections and withdrawals don't appear — they're NAV-per-share-invariant.

**This is what a public mutual fund quotes when it tells you its
return.** It's the metric an investor in your fund actually
experienced. Use it for performance reporting, marketing materials,
and benchmark comparisons.

### When to switch — AUM + Modified Dietz (institutional/GIPS)

```python
pf = src.to_portfolio(
    fund_id="<fund_id_alpha>",
    basis="aum",
    method="modified_dietz",
)
```

- Uses **total AUM** as the price series.
- Signs every flow type (`INJECTION` positive, `WITHDRAWAL` /
  `DISTRIBUTION` negative).
- Applies the GIPS-standard Modified Dietz formula:

  $$
  r_t = \frac{\mathrm{AUM}_t - \mathrm{AUM}_{t-1} - F_t}{\mathrm{AUM}_{t-1} + 0.5 \cdot F_t}
  $$

Use this when you need to **report to allocators** under GIPS
conventions, when you're computing returns net of management fees that
trigger AUM movements, or when you specifically need the AUM-weighted
TWR.

For daily-precision flow timing, swap `method="daily_twr"` —
`(AUM_t − AUM_{t-1} − F_t) / AUM_{t-1}` — it assumes flows happen at
period start, which is more accurate when you have daily flow data.

!!! warning "Don't mix bases mid-report"
    A 3-month return computed `nav_per_share + total_return` is not
    the same number as the same period computed `aum + modified_dietz`
    when flows occurred — they're answering different questions.
    Pick one for any given report and stick with it.

---

## Daily P&L check

```python
import fundcloud as fc
import pandas as pd

src = fc.accounts.FundCloud()
pf  = src.to_portfolio(fund_id="<fund_id_alpha>")

today  = pf.returns.iloc[-1]
mtd    = (1 + pf.returns[pf.returns.index.month == pd.Timestamp.now().month]).prod() - 1
ytd    = (1 + pf.returns[pf.returns.index.year  == pd.Timestamp.now().year]).prod()  - 1

print(f"Today:  {today:+.2%}")
print(f"MTD:    {mtd:+.2%}")
print(f"YTD:    {ytd:+.2%}")
```

For a one-call version, `pf.period_returns()` produces MTD / 3M / YTD /
1Y / 3Y / 5Y / All-time in one go.

---

## Reconciliation — sanity-checking flows

Before showing numbers to a CFO or an LP, reconcile what the library
ingested against what your records say:

```python
flows = src.capital_flows(fund_id="<fund_id_alpha>")
flows
#              flow_type    amount currency  account_id    notes
# flow_date
# 2024-09-15   INJECTION  5_000_000      USD  <acct_id_1>   None
# 2024-09-15  WITHDRAWAL  2_500_000      USD  <acct_id_1>   None
# 2024-09-15   INJECTION  2_500_000      USD  <acct_id_2>   None
```

The two `INJECTION + WITHDRAWAL` rows on the same day for one account
are a re-allocation between sleeves — net zero AUM impact. Modified
Dietz handles that correctly because it sums signed flows. If a row
looks wrong, fix it in the platform before re-running the report.

To prove your return-calc is consistent, compare adjusted vs raw NAV
on a flow date:

```python
raw  = src.nav(fund_id="<fund_id_alpha>", adjust_for_flows=False, start="2024-09-14")
adj  = src.nav(fund_id="<fund_id_alpha>", adjust_for_flows=True,  start="2024-09-14")
common = raw.index.intersection(adj.index)
(adj.loc[common, "nav"] - raw.loc[common, "nav"]).abs().describe()
```

The non-zero days are exactly where the platform smoothed a flow
event. If those dates don't match your `capital_flows()` table, raise
it with the FundCloud platform team — that's a data issue, not a
library issue.

---

## Raw data — when you need to compute something we haven't shipped

Every method returns a plain pandas DataFrame, ready for whatever
custom analytic you need:

| Method | What it gives you |
|---|---|
| `src.nav()` | Daily `nav` (per-share), `aum`, `shares`, `daily_return`, `fill_type`. Server-adjusted by default. |
| `src.capital_flows()` | One row per flow event: `flow_type`, `amount` (always positive — sign is in the type), `currency`, `account_id`, `notes`. |
| `src.positions()` | Current open positions. Per-symbol quantity, market value, weight, unrealized PnL. |
| `src.trades()` | Executed fills. `symbol`, `side`, `quantity`, `price`, `amount`, `fee`, `broker`, `status`. |

All four take optional `fund_id`, `account_id`, `start`, `end`. NAV
also takes `adjust_for_flows`.

### Defaults you should know about

- **`start` defaults to one year ago.** Same convention as
  [`fundcloud.data.FMP`](../../reference/data.md), `AV`, etc. Pass
  `start=` for longer history:
  ```python
  long = src.nav(start="2020-01-01")
  ```
- **`nav()` returns flow-adjusted values by default.** Pass
  `adjust_for_flows=False` for raw values — useful when reconciling.
- **`page_size` is aggressive** (1000 rows/request). Most funds fit
  in a single round-trip. The library drains pagination automatically
  and caps at 10,000 pages as a runaway guard.

### Bring-your-own-formula

If you need a return convention we don't ship, drop down to the
primitive:

```python
nav   = src.nav(fund_id="<fund_id_alpha>", adjust_for_flows=False)
flows = src.capital_flows(fund_id="<fund_id_alpha>")

# Compute returns however you like, then build a Portfolio
returns = fc.metrics.returns_from_nav(
    nav["nav"],
    distributions=...,                     # see returns_from_nav docstring
    method="total_return",
)
pf = fc.Portfolio.from_nav(nav["nav"], distributions=..., name="custom")
```

See [`returns_from_nav`](../../reference/metrics.md) and
[`Portfolio.from_nav`](../../reference/portfolio.md) for the full
parameter menu.

---

## Currency and FX

The accounts package **does not** convert currencies.

- A fund's NAV is denominated in the fund's `currency` (USD for the
  examples above).
- Each `position` carries its own `currency` (a USD fund holding GBP
  gilts will have `currency=GBP` on those rows).
- Capital flows are reported in their native currency.

If you analyse a multi-currency fund, ensure your benchmark, your
display formatting, and any cross-fund roll-ups are all in the same
currency.

---

## Errors you might see

Every error from `fc.accounts.*` is rooted at
`fundcloud.errors.FundcloudError`, so a single `try`/`except` covers
the whole library:

```python
try:
    pf = src.to_portfolio()
except fc.errors.AmbiguousError:
    # Multiple funds visible and no fund_id/account_id passed.
    # Inspect src.list_funds() and pick one.
    ...
except fc.errors.NotFoundError:
    # Fund or account doesn't exist (or was unlinked).
    ...
except fc.errors.AuthError:
    # Missing or invalid API key. Check FUNDCLOUD_API_KEY.
    ...
except fc.errors.QuotaError:
    # Rate limit or daily quota exhausted on the FundCloud plan.
    # Wait or upgrade.
    ...
except fc.errors.TransientError:
    # 5xx or 429 retries exhausted. Usually safe to retry the call.
    ...
```

See [Reference → Errors](../../reference/errors.md) for the full hierarchy.

---

## What's next

- [`fundcloud.metrics`](../portfolio/metrics.md) — every metric `pf.summary()` and `pf.metrics()` exposes.
- [`Returns analysis`](../portfolio/returns-analysis.md) — the broader trader-focused walkthrough of the analytics surface.
- [`Tearsheets`](../reports/tearsheets.md) — customizing the rendered HTML / PDF / Excel output.
- [`fundcloud.data.FundCloud`](../../reference/data.md) — pull market data through the same FundCloud credential for benchmarks.
