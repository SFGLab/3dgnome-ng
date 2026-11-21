"""The rows of a pandas table."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def rows(table: pd.DataFrame | pd.Series) -> Iterator[Any]:
    """Returns the rows of a table as named tuples, each column by its name."""
    return iter(table.itertuples(index=False))  # pyright: ignore[reportAttributeAccessIssue]
