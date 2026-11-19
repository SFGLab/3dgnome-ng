"""Shared setup of the test suite.

    .venv/bin/pytest                      every test that needs no input data and no GPU
    .venv/bin/pytest tests/unit           the unit checks alone
    .venv/bin/pytest -m "not slow"        without the tests that take over a minute
    .venv/bin/python tests/unit/test_polymer.py     one file as a script

Most unit files record their checks through a module level `check(name, cond, detail)` that
appends to `PASS` or `FAIL` and prints one line. The hook below fails a test that recorded a
failed check, which keeps every file runnable as a script with its own summary.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Files that take over a minute on a laptop. The JAX kernels run on the CPU there.
SLOW = {"test_batch_seeding.py"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name in SLOW:
            item.add_marker(pytest.mark.slow)


def recorded_failures(module: Any) -> list[str] | None:
    """Returns the list that the module's `check` appends failed names to, or None without one."""
    check = getattr(module, "check", None)
    if check is None:
        return None
    failed = getattr(check, "__globals__", {}).get("FAIL")
    return failed if isinstance(failed, list) else None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, Any, Any]:
    failed = recorded_failures(getattr(item, "module", None))
    before = len(failed) if failed is not None else 0
    result = yield
    if failed is not None and len(failed) > before:
        pytest.fail("failed checks: " + "; ".join(failed[before:]), pytrace=False)
    return result
