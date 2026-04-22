---
title: Install
description: Install Fundcloud with uv or pip. Core only, or with data, optimisation, TA, and reports extras.
---

# Install

Fundcloud ships as a single `abi3` wheel per platform, so one install covers Python 3.10 through 3.14 on Linux, macOS, and Windows. The core pulls only `pandas`, `numpy`, `scipy`, `scikit-learn`, and `plotly` — everything heavy is opt-in via extras.

=== "uv (recommended)"

    ```bash
    uv add fundcloud                  # core only
    uv add "fundcloud[data]"          # + every data provider (yf, fmp, av, binance)
    uv add "fundcloud[pf,ta,data]"    # + skfolio + TA-Lib + data sources
    uv add "fundcloud[all]"           # everything
    ```

=== "pip"

    ```bash
    pip install fundcloud
    pip install "fundcloud[data]"
    pip install "fundcloud[pf,ta,data]"
    pip install "fundcloud[all]"
    ```

!!! tip "Which extras do I actually need?"
    Start with core. Add `[data]` the moment you want to pull real prices, `[pf]` the moment you want optimisation, `[reports]` the moment you want a PDF or workbook. You can always add more later.

## Extras

| Extra | Adds | Notes |
|---|---|---|
| `pf` | [skfolio](https://skfolio.org) | Portfolio optimisation — MeanRisk, HRP, HERC, etc. |
| `ta` | [TA-Lib](https://github.com/TA-Lib/ta-lib-python) | 170+ technical indicators. Requires the C library (`brew install ta-lib` on macOS, `apt install libta-lib-dev` on Debian). |
| `data-yf` / `data-fmp` / `data-av` / `data-bn` | yfinance / httpx / httpx / ccxt | Individual data providers. |
| `data` | bundle of every provider above | |
| `viz` | matplotlib + kaleido | Static plot exports for PDF embedding. |
| `reports` | weasyprint + xlsxwriter | Adds the optional WeasyPrint PDF engine and XlsxWriter workbooks with native charts. |
| `all` | everything above | |

Exploratory data analysis (`fundcloud.explore.{profile, compare, quickview}`) ships in core — no extra needed.

!!! note "PDF engines"
    `Tearsheet.render_pdf(...)` defaults to a pure-Python matplotlib `PdfPages` backend and only needs the `[viz]` extra — no system libraries. An optional `engine="weasyprint"` backend (CSS-styled pages) activates when `[reports]` is installed; on macOS that additionally needs `brew install pango` and `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` so WeasyPrint can find the Pango / GLib dylibs.

## Python and platform support

| Platform | Python versions | Rust kernel |
|---|---|---|
| Linux x86_64 / aarch64 | 3.10 → 3.14 | prebuilt wheel |
| macOS x86_64 / arm64 | 3.10 → 3.14 | prebuilt wheel |
| Windows x86_64 | 3.10 → 3.14 | prebuilt wheel |
| Other architectures | 3.10 → 3.14 | pure-Python fallback (1e-10 parity) |

Wheels are built with PyO3's `abi3-py310` feature, so one wheel per platform covers every supported Python version. On architectures without a prebuilt wheel, `pip` falls back to the source distribution — which still runs; you just lose the Rust acceleration and get the verified NumPy fallback instead. See [Rust kernels → Fallback](guides/accelerators/rust-kernels.md#fallback) for the methodology.

## Verify the install

```python
import fundcloud
from fundcloud.kernels import HAS_RUST, kernel_version

print(fundcloud.__version__)      # "0.1.0"
print(HAS_RUST, kernel_version()) # True, "0.1.0"  — or  False, "python-fallback"
```

If `HAS_RUST` is `False`, you're on the pure-Python fallback — code still runs; numbers still agree to 1e-10; it's just slower on large panels.

## Next

[Run the 60-second quickstart →](quickstart.md){ .fc-btn .fc-btn--primary }
[Browse the API reference →](reference/data.md){ .fc-btn }
