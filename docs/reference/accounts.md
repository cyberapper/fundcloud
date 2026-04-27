# Accounts

The `fundcloud.accounts` package wraps account-level data sources —
historical NAV, positions, trades, and capital flows from platforms
like FundCloud and Interactive Brokers. Every provider satisfies the
same [`AccountProvider`](#fundcloud.accounts._base.AccountProvider)
protocol, so the analysis surface is identical regardless of source.
For task-first walkthroughs, start with [Analysing a FundCloud
fund](../guides/accounts/fundcloud.md) or [Analysing an Interactive
Brokers account](../guides/accounts/ib.md).

## Provider protocol

::: fundcloud.accounts._base.AccountProvider
    options:
      filters: []

::: fundcloud.accounts._base.BaseAccountProvider
    options:
      filters: []

## FundCloud provider

::: fundcloud.accounts.fundcloud.FundCloud
    options:
      filters: []

## Interactive Brokers provider

::: fundcloud.accounts.ib.IB
    options:
      filters: []

## Errors

Provider errors are rooted at `fundcloud.errors.FundcloudError` — see the
[errors reference](./errors.md) for the full hierarchy.
