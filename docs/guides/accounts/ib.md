---
title: Analysing an Interactive Brokers account
description: Drop an Interactive Brokers Flex Query CSV into the same `Portfolio` analytics surface you'd use for a backtest. Tearsheets, drawdown tables, attribution — all work without changes.
---

# Analysing an Interactive Brokers account

If you trade through **Interactive Brokers**, your account's daily NAV
plus its deposit / withdrawal history is one Flex Query export away.
Drop the CSV into `fundcloud.accounts.IB` and your live track record
gets the same `Portfolio`-flavoured treatment as a backtested
strategy: drawdown tables, period returns, monthly heatmaps,
benchmark comparisons, skfolio-compatible attribution — all
unchanged from how they work for a simulated book.

This guide is written for portfolio managers and traders. For the
protocol-level reference, see [Reference → Accounts](../../reference/accounts.md).

---

## 60-second tour

```python
import fundcloud as fc

src = fc.accounts.IB("flex_export.csv")
pf  = src.to_portfolio()
fc.reports.Tearsheet(pf).render_html("ib_review.html")
```

Three lines from "I have a Flex CSV" to "here's an investor-grade
tearsheet". Everything below is for when you need more control or
need to set up the export from scratch.

!!! note "Why is the class called `IB`, not `IBKR`?"
    `IB` is the call-site shorthand. The full broker name —
    Interactive Brokers — is what you'd write in a marketing deck;
    `IB` is what you want to type in a notebook 50 times a day.

---

## Setting up the Flex Query in Interactive Brokers

You only do this once — IB saves the query template and re-runs it on
demand.

1. Log in to **Client Portal** at <https://www.interactivebrokers.com>.
2. Go to **Reports → Flex Queries → Activity Flex Query** and click
   the **+** button to create a new query.
3. Name it anything (`Fundcloud` is a fine convention).
4. Tick the two sections we use: **Net Asset Value (NAV) in Base**
   and **Cash Transactions**.
5. Set the **Delivery Configuration** to match the parser defaults:

   | Setting | Value |
   |---|---|
   | Format | `CSV` |
   | Period | `Last 365 Calendar Days` (or longer) |
   | Include header and trailer records? | `No` |
   | Include column headers? | `Yes` |
   | Display single column header row? | `No` |
   | Include section code and line descriptor? | `No` |
   | Date Format | `yyyyMMdd` |
   | Time Format | `HHmmss` |
   | Date/Time Separator | `; (semi-colon)` |

6. Save, then run and download the CSV.

### Which fields to tick — two paths

#### Path A — "Just tick everything" (zero thinking)

In each of the two sections, click **Select All** on the field grid.
The parser ignores extra columns, so a maximalist export works out
of the box. Recommended unless you specifically want a leaner
download.

In the **Cash Transactions** section's **Options** panel, tick
everything if you don't want to think — but if you want the smallest
possible export, the only option that matters is **Deposits &
Withdrawals** (plus **Detail** on the right column). The parser
drops every other event type — dividends, broker interest,
withholding tax — automatically.

#### Path B — Minimal essentials (smaller export, faster download)

Everything below is optional except the listed fields.

**Net Asset Value (NAV) in Base — required fields**

| Field | Why we need it |
|---|---|
| `Account ID` | Identifies the row's account (`ClientAccountID`). |
| `Currency` | Base currency for the NAV (`CurrencyPrimary`). |
| `Report Date` | Date of the NAV row. |
| `Cash` | Used to disambiguate this section from Cash Transactions during parsing. Doesn't have to be non-zero — its presence in the header is what matters. |
| `Total` | The AUM in base currency. **This is the NAV value we read.** |

**Cash Transactions — required Options + fields**

Under the section's **Options** panel, tick at minimum:

- **Deposits & Withdrawals** (the only event type we care about)
- **Detail** (we need row-level data, not the Summary aggregate)

Under the field grid:

| Field | Why we need it |
|---|---|
| `Account ID` | Identifies the row's account. |
| `Currency` | Native currency of the flow (e.g., `HKD`, `USD`). |
| `FX Rate To Base` | Multiplier for native → base currency conversion. |
| `Date/Time` | When the flow happened. Both `yyyyMMdd` and `yyyyMMdd;HHmmss` work. |
| `Amount` | Signed amount (positive = deposit, negative = withdrawal). |
| `Type` | The "Deposits/Withdrawals" filter label. |

Recommended-but-optional: `Description` (becomes the `notes` column —
useful for reconciliation), `TransactionID` (lets you stitch
multiple exports without duplicating rows).

!!! tip "If you're not sure, pick Path A"
    The parser is designed to ignore everything it doesn't use.
    The only practical difference between Path A and Path B is the
    file size. For most users, click-everything is the safer choice.

---

## Loading the CSV

```python
from fundcloud.accounts import IB

# Single file — the most common case
src = IB("export.csv")

# Inline content — useful in notebooks, tests, or piping from another tool
src = IB.from_string(csv_text)

# Multiple files — concatenated per-section. Use this for:
#  · year-by-year exports stitched into one continuous picture
#  · master-account books exported one sub-account at a time
src = IB(files=["2023.csv", "2024.csv", "2025.csv"])
```

Loading is **eager** — the CSV is parsed once at construction time,
so subsequent `nav()` / `capital_flows()` / `to_portfolio()` calls
are pure pandas slicing. Reload by re-instantiating.

---

## The headline workflow — tearsheet

For most managers, the daily question is "how did the account do?".
One call:

```python
import fundcloud as fc

src = fc.accounts.IB("export.csv")
pf  = src.to_portfolio()
```

`pf` is a regular `Portfolio` object — the same one a backtest
produces. Every `.fc` accessor method, every metrics function, and
every report renderer works on it:

=== "Quick metrics"

    ```python
    pf.summary()           # 11-metric snapshot
    pf.sharpe()            # single number
    pf.max_drawdown()      # peak-to-trough loss
    pf.cvar(alpha=0.95)    # tail-risk measure
    fc.metrics.cagr(pf.returns, periods_per_year=252)
    ```

=== "Full tearsheet"

    ```python
    spy = fc.data.YF("SPY", start=str(pf.returns.index[0].date())).read()["close"]
    fc.reports.Tearsheet(pf, benchmark=spy).render_html("ib_review.html")
    ```

    Produces a fully-formatted tearsheet with cumulative-return chart,
    drawdown table, monthly-return heatmap, period-return table, and
    benchmark comparison. Send to the LP. Pin to the wall.

=== "Drawdown table"

    ```python
    pf.worst_drawdowns(top=5)
    #     Started     Recovered  Drawdown   Days
    #   2025-10-23   2026-01-08    -0.18      78
    #   2025-12-04   2025-12-19    -0.07      16
    #   2026-02-14   2026-03-02    -0.05      17
    ```

    `Started → Recovered` is what risk committees actually ask
    about. `pf.drawdown_details()` returns the full DataFrame
    including `valley` and `days_to_recover`.

=== "Period returns"

    ```python
    pf.period_returns(benchmark=spy)
    #                       SPY    Strategy
    # MTD                0.0148      0.0289
    # 3M                 0.0521      0.1147
    # YTD                0.0964      0.1832
    # 1Y (ann.)          0.1212      0.2401
    ```

    Standard monthly-report layout. 3Y / 5Y / 10Y rows are
    annualised — directly comparable to 1Y.

!!! tip "FundCloud users can swap providers without changing analytics"
    `fc.accounts.IB("export.csv").to_portfolio()` and
    `fc.accounts.FundCloud().to_portfolio(fund_id="…")` return
    structurally identical `Portfolio` objects. Once you've got one,
    every downstream tool — Tearsheet, drawdown analysis, optimizer
    integration — is the same.

---

## Why IB's defaults differ from FundCloud's

When you call `to_portfolio()` on the FundCloud provider, the default
is `basis="nav_per_share"` + `method="total_return"` — the metric a
public mutual fund quotes when reporting investor return.

For IB, the defaults flip to `basis="aum"` + `method="modified_dietz"`.

The reason: a brokerage account doesn't issue shares. There's no
`shares_outstanding` for the per-share NAV concept to apply to. The
GIPS-standard alternative is **AUM-basis Modified Dietz** — a
time-weighted return formula that backs out deposits and withdrawals
mid-period:

$$
r_t = \frac{\mathrm{AUM}_t - \mathrm{AUM}_{t-1} - F_t}{\mathrm{AUM}_{t-1} + 0.5 \cdot F_t}
$$

where `F_t` is the signed net inflow on day `t`.

For daily-precision flow timing — assumes flows happen at period
start, more accurate when you have daily NAV and daily flow data —
pass `method="daily_twr"`:

```python
pf = src.to_portfolio(method="daily_twr")
```

!!! warning "Don't expect your IB statement's headline figure"
    IB's own statements often quote a money-weighted return
    (IRR-style) that includes the *timing* of your deposits as part
    of the return. Modified Dietz strips that out — it tells you
    "how would the underlying investments have performed without
    your deposit decisions". Useful for evaluating skill; not the
    same number as the IRR on your monthly statement. Document
    which one you're quoting in your reports.

---

## Daily P&L check

```python
import fundcloud as fc
import pandas as pd

src = fc.accounts.IB("export.csv")
pf  = src.to_portfolio()

today = pf.returns.iloc[-1]
mtd   = (1 + pf.returns[pf.returns.index.month == pd.Timestamp.now().month]).prod() - 1
ytd   = (1 + pf.returns[pf.returns.index.year  == pd.Timestamp.now().year]).prod()  - 1

print(f"Today:  {today:+.2%}")
print(f"MTD:    {mtd:+.2%}")
print(f"YTD:    {ytd:+.2%}")
```

For a one-call version, `pf.period_returns()` produces MTD / 3M /
YTD / 1Y / 3Y / 5Y / All-time in one go.

---

## Reconciliation — sanity-checking flows

Before quoting numbers in a report or to your accountant, eyeball the
deposits and withdrawals the parser saw:

```python
flows = src.capital_flows()
flows
#                       flow_type     amount  currency  account_id  notes
# flow_date
# 2024-09-15            INJECTION   5_000.00       USD  U_…         CASH RECEIPTS
# 2024-12-23            WITHDRAWAL  2_500.00       USD  U_…         ACH WITHDRAWAL
```

`amount` is always **positive** (in the account's base currency);
direction is encoded in `flow_type`. IB's raw signed `Amount` and
its native-currency value are gone by this layer — see "Bring your
own formula" below if you need them.

To prove the return calc isn't double-counting flows, compare
adjusted vs raw NAV around a deposit date:

```python
raw  = src.nav(adjust_for_flows=False)
adj  = src.nav(adjust_for_flows=True)

# On the day a deposit happens, raw shows a jump; adjusted does not.
(raw["aum"] - adj["aum"]).abs().describe()
```

Non-zero `(raw − adj)` outside flow days is rounding noise (small)
or a bug worth reporting (large).

!!! warning "Tax-season trap: dividends and interest are NOT capital flows"
    `capital_flows()` returns deposits and withdrawals only. Dividend
    income, broker interest paid / received, and withholding tax all
    appear in IB's Cash Transactions section but get filtered out
    here — they're investment income, not capital events. For a
    tax-time aggregation, parse the full CSV and read
    `src._tx_df` (advanced internal access — schema may move in a
    future release).

---

## Multi-account ("master account") usage

If your Flex Query covers multiple `ClientAccountID`s — e.g., a master
account with linked sub-accounts — every method takes a `fund_id` /
`account_id` kwarg:

```python
src = IB("master_export.csv")
src.list_accounts()
# →  account_id  account_name   fund_id  fund_name  currency  latest_aum
#    U_TEST_001  U_TEST_001   U_TEST_001 U_TEST_001     USD    500_000
#    U_TEST_002  U_TEST_002   U_TEST_002 U_TEST_002     USD    150_000

# Drill into one sub-account at a time
pf_main = src.to_portfolio(account_id="U_TEST_001")
pf_other = src.to_portfolio(account_id="U_TEST_002")

# Side-by-side panel for comparison
import pandas as pd
panel = pd.DataFrame({
    a["account_id"]: src.to_portfolio(account_id=a["account_id"]).returns
    for _, a in src.list_accounts().iterrows()
})
panel.fc.summary()              # 11-metric table, one column per account
panel.fc.plot_cumulative()      # one line per account
```

For IB, `fund_id` and `account_id` are the same identifier
(single-tier hierarchy) — passing either works. With a single
account visible in the export, pass nothing.

!!! tip "Single account ≠ single sleeve"
    Even a single IB account often holds multiple strategies (CTA
    sleeve + equity long/short + cash buffer, say). The Activity Flex
    CSV won't separate those — for sleeve-level attribution you need
    a separate Trades / Open Positions Flex section, or sleeve-tagged
    metadata maintained outside IB.

---

## Bring your own formula

Every method returns a plain pandas DataFrame, ready for whatever
custom analytic you need:

| Method | What it gives you |
|---|---|
| `src.nav()` | Daily `nav` (= AUM, since IB has no shares), `aum`, `shares` (synthesised `1.0`), `daily_return`, `fill_type`. Adjusted by default. |
| `src.capital_flows()` | One row per deposit / withdrawal: `flow_type` (INJECTION / WITHDRAWAL), `amount` (always positive, base currency), `currency`, `account_id`, `notes`. |
| `src.positions()` | Empty frame — the Activity Flex Query doesn't include per-symbol position rows. Configure a separate **Open Positions** Flex section in IB to populate it. |
| `src.trades()` | Empty frame — the Activity Flex Query doesn't include per-symbol trade rows. Configure a separate **Trades** Flex section in IB to populate it. |

All four take optional `fund_id`, `account_id`, `start`, `end`. NAV
also takes `adjust_for_flows`.

### Defaults you should know about

- **The full CSV period is shown by default.** Unlike the network
  providers (which apply a 1-year-back default to limit bandwidth),
  the IB provider doesn't trim — the CSV is already bounded by
  whatever export window you set in the Flex Query. Pass `start=` /
  `end=` to narrow.
- **`nav()` returns flow-adjusted values by default.** The synthetic
  curve replays returns as if no deposits / withdrawals had occurred
  — useful for charting and explaining returns to non-traders. Pass
  `adjust_for_flows=False` for the raw `Total` column straight from
  the CSV (useful for reconciliation).
- **`capital_flows()` is in base currency.** Native amounts and the
  FX rate to base are preserved on `src._tx_df` if you need them.

### Compute your own return convention

If you need a return formula we don't ship:

```python
nav   = src.nav(adjust_for_flows=False)            # raw Total column
flows = src.capital_flows()
# Provide signed flows in the convention `returns_from_nav` expects:
signed = flows["amount"].where(flows["flow_type"] == "INJECTION", -flows["amount"])

returns = fc.metrics.returns_from_nav(
    nav["aum"],
    capital_flows=signed.reindex(nav.index, fill_value=0),
    method="daily_twr",
)
pf = fc.Portfolio.from_nav(nav["aum"], capital_flows=signed, name="custom-IB")
```

See [`returns_from_nav`](../../reference/metrics.md) for the full
method menu.

---

## Currency / FX

Interactive Brokers' NAV section is denominated in your account's
**base currency**. The Cash Transactions section can carry flows in
*other* currencies (e.g., HKD deposits into a USD-base account); each
row has its own `Currency` plus an `FX Rate To Base` column.

The library multiplies `Amount × FXRateToBase` so the values returned
by `capital_flows()` are already in base currency, ready to
reconcile against base-currency NAV. The native amount and rate are
preserved in `src._tx_df` if you need them.

FX *conversion* of the NAV itself is not performed. If you have a
multi-base-currency export (which is unusual), the parser raises
`MalformedDataError` rather than silently picking one — re-run the
Flex Query in a single base currency.

!!! warning "Multi-currency reconciliation gotcha"
    A USD-base account with HKD deposits will show flow amounts in
    USD (correctly converted). But your monthly bank statement
    probably shows the HKD figure. When reconciling against your
    bank, read native amounts from `src._tx_df['amount_native']`
    instead of `flows['amount']`.

---

## Errors you might see

Every error from `fc.accounts.*` is rooted at
`fundcloud.errors.FundcloudError`, so a single `try`/`except` covers
the whole library:

```python
try:
    pf = src.to_portfolio()
except fc.errors.MalformedDataError:
    # The CSV's structure is broken — missing essential column,
    # unrecognised section, unparseable date. The exception message
    # tells you which.
    ...
except fc.errors.AmbiguousError:
    # Multiple accounts visible and no account_id passed.
    # Inspect src.list_accounts() and pick one.
    ...
except fc.errors.NotFoundError:
    # Asked-for fund_id / account_id isn't in the CSV.
    ...
```

See [Reference → Errors](../../reference/errors.md) for the full
hierarchy.

---

## What's next

- [`Returns analysis`](../portfolio/returns-analysis.md) — every
  metric `pf.summary()` exposes, explained for portfolio managers.
- [`Tearsheets`](../reports/tearsheets.md) — customising the
  rendered HTML / PDF / Excel output.
- [`fundcloud.accounts.FundCloud`](./fundcloud.md) — the matching
  guide for the FundCloud platform integration.
- [Reference → Accounts](../../reference/accounts.md) — protocol /
  class signatures.
