# Contributing to Fundcloud

Thanks for your interest! Fundcloud is MIT-licensed; by contributing you agree your work lands under the same license.

## Development setup

```bash
git clone https://github.com/cyberapper/fundcloud
cd fundcloud
uv sync --group dev           # installs runtime + dev deps; builds Rust extension via maturin
uv run pre-commit install     # optional but recommended
```

## Common tasks

```bash
uv run pytest -q                        # unit + integration tests
uv run pytest -m "not network and not slow"
cargo test --workspace                  # Rust tests
uv run ruff check python/fundcloud tests
uv run ruff format python/fundcloud tests
uv run mypy python/fundcloud
uv run mkdocs serve                     # preview docs locally on :8000
```

## Full CI simulation

Pre-commit covers lint and format only. Run these before opening a PR:

```bash
uv run pytest -ra -m "not network and not slow"   # Python tests
cargo test --workspace --no-fail-fast             # Rust tests
uv run mypy python/fundcloud                      # Type checking
uv run mkdocs build --strict                      # Docs build
```

## PR scope

Keep each PR focused on a single concern. For changes larger than a bug fix or doc tweak, open an issue first to align on approach before writing code.

## Coding standards

- Python: Ruff (lint + format) + Mypy `strict`. Type annotations on all public APIs.
- Rust: `cargo fmt` + `cargo clippy -D warnings`. No `unsafe` without a `SAFETY:` comment block.
- Tests: every new public function ships with a test. Prefer property tests (Hypothesis/proptest) for numeric code.
- Docstrings: NumPy style.
- Commit messages: [Conventional Commits](https://www.conventionalcommits.org/).

## Design before code

For anything larger than a bug fix or a doc tweak, open an issue to align on approach before writing code.

## Known gotchas

**`cargo clippy -D warnings`** — every Rust warning is a build error. Run
`cargo clippy --workspace -- -D warnings` locally before committing Rust changes.

**PyO3 `useless_conversion`** — PyO3 0.22's `#[pyfunction]` macro can trigger
`clippy::useless_conversion` when a function returns `PyResult`. If you see this,
make the helper function infallible (return the value directly, use `.expect()`
instead of `?`).

**`mypy --strict`** — all public functions need full type annotations. Use
`from __future__ import annotations` for forward references. CI runs mypy over
`python/fundcloud`; add annotations before adding new public API.

**`mkdocs --strict`** — any `nav:` entry pointing to a missing file, or an
`--8<--` snippet referencing a non-existent path, is a build error. Run
`uv run mkdocs build --strict` after any docs changes.

**matplotlib version pinning** — if Python tests fail with `RecursionError`
inside `legend_handler.py`, a matplotlib release regression is likely. Check
`uv run pip show matplotlib` and compare against `uv.lock`.

## Reporting security issues

Please do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/cyberapper/fundcloud/security/advisories/new) — no email needed.
