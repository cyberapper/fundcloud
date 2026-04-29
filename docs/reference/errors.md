# Errors

The `fundcloud.errors` module defines the library-wide typed error
hierarchy. Every module that raises a library-specific error raises
from one of these classes, so user code can catch them uniformly:

```python
import fundcloud as fc

try:
    pf = fc.accounts.FundCloud().to_portfolio()
except fc.errors.AuthError:
    ...        # 401, bad / missing API key
except fc.errors.TransientError:
    ...        # 5xx / 429 after retries exhausted
```

::: fundcloud.errors
    options:
      filters: []
