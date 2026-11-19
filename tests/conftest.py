"""Shared setup of the test suite.

.venv/bin/pytest                      every test that needs no input data and no GPU
.venv/bin/pytest tests/unit           the unit tests alone
.venv/bin/pytest -m "not slow"        without the tests that take over a minute
.venv/bin/pytest tests/unit/test_polymer.py     one file
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Files that take over a minute on a laptop. The JAX kernels run on the CPU there.
SLOW = {"test_batch_seeding.py"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.name in SLOW:
            item.add_marker(pytest.mark.slow)
