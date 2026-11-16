"""Levels of tests.

    pytest            fast numerical tests only (default; target under one minute)
    pytest --exact    adds the symbolic (sympy) tests marked `exact`
    pytest --all      adds also the tests marked `slow` (a few minutes)
    pytest -m core    the key checks of the audit map (supplement, section S2), exact
                      and slow ones included (about a minute and a half)
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--exact", action="store_true", help="also run tests marked exact")
    parser.addoption("--all", action="store_true", help="run every test (exact and slow)")


def _core_selected(config):
    """True when the marker expression asks for the core tests (pytest -m core)."""
    expr = config.getoption("markexpr", default="") or ""
    return "core" in expr.replace("not core", "")


def pytest_collection_modifyitems(config, items):
    run_all = config.getoption("--all") or _core_selected(config)
    run_exact = run_all or config.getoption("--exact")
    skip_exact = pytest.mark.skip(reason="symbolic check: use --exact or --all")
    skip_slow = pytest.mark.skip(reason="slow check: use --all")
    for item in items:
        if "exact" in item.keywords and not run_exact:
            item.add_marker(skip_exact)
        if "slow" in item.keywords and not run_all:
            item.add_marker(skip_slow)
