"""Tests for ``fundcloud._config``."""

from __future__ import annotations

from fundcloud._config import config, get_config, set_config


def test_default_config_values() -> None:
    cfg = get_config()
    assert cfg.periods_per_year == 252
    assert cfg.risk_free_rate == 0.0
    assert cfg.tol == 1e-12


def test_set_config_overrides_fields_and_returns_new() -> None:
    before = get_config()
    updated = set_config(periods_per_year=365)
    try:
        assert updated.periods_per_year == 365
        assert get_config().periods_per_year == 365
    finally:
        set_config(periods_per_year=before.periods_per_year)


def test_config_context_manager_restores_previous() -> None:
    before = get_config().periods_per_year
    with config(periods_per_year=260) as scoped:
        assert scoped.periods_per_year == 260
        assert get_config().periods_per_year == 260
    assert get_config().periods_per_year == before


def test_config_context_manager_restores_on_exception() -> None:
    before = get_config().periods_per_year
    try:
        with config(periods_per_year=99):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert get_config().periods_per_year == before
