"""The rows of a pandas table."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def rows(table: pd.DataFrame | pd.Series) -> Iterator[Any]:
    """The rows of a table as named tuples, each column by its name. pandas states a selection of
    rows as a table or a column, and a table is what the callers have."""
    return iter(table.itertuples(index=False))  # pyright: ignore[reportAttributeAccessIssue]
