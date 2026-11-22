"""Shared setup of the site's tests.

    pytest web/tests                      every test of the site
    pytest web/tests/test_api.py          one file

The tests of a file run in the order they are written. A file builds its site once, in a
fixture of module scope. Its tests read and change that one site.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1]
# The site, the packages it vendors and the engine of this checkout.
for extra in (WEB.parent, WEB / "vendor", WEB):
    sys.path.insert(0, str(extra))


@pytest.fixture(scope="module", autouse=True)
def own_environment() -> Generator[None, None, None]:
    """Gives each file the environment the run began with. A file sets the variables of its site."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)
