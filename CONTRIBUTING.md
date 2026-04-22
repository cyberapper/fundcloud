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

## Reporting security issues

Please do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/cyberapper/fundcloud/security/advisories/new) — no email needed.
