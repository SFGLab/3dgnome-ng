"""The two refusals of a request before any work starts."""

from __future__ import annotations


class UnknownCell(Exception):
    """A request for a cell type the catalog lacks. The answer is 404."""

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown cell type {name!r}")


class Refused(ValueError):
    """A request the server refuses, with the form field it concerns. The answer is 400."""

    def __init__(self, message: str, field: str) -> None:
        super().__init__(message)
        self.field = field
