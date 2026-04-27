"""Private per-provider HTTP clients (not public API).

Each module in this subpackage wraps the generic
:class:`fundcloud.data._http.HttpClient` with provider-specific auth,
pagination, and error-mapping. Consumers are always other modules in
:mod:`fundcloud` (e.g., ``fundcloud.data.fundcloud`` and
``fundcloud.accounts.fundcloud`` both use
:class:`fundcloud._clients.fundcloud.FundCloudClient`).
"""
