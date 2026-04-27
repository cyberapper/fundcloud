# Accounts

The `fundcloud.accounts` package wraps account-level data sources —
historical NAV, positions, trades, and capital flows from platforms
like FundCloud (today), IBKR and Plaid (planned). Every provider
satisfies the same [`AccountProvider`](#fundcloud.accounts._base.AccountProvider)
protocol, so the analysis surface is identical regardless of source.
For the task-first walkthrough, start with [Analysing a FundCloud
fund](../guides/accounts/fundcloud.md).

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

## Errors

Provider errors are rooted at `fundcloud.errors.FundcloudError` — see the
[errors reference](./errors.md) for the full hierarchy.
