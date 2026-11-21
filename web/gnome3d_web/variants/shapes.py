"""The shape of the answer of the variants dialog."""

from __future__ import annotations

from typing_extensions import TypedDict


class VariantRow(TypedDict):
    type: str
    pos: int
    end: int
    length: int
    genotype: str
    change: str


class VariantCheck(TypedDict):
    summary: str
    applied: dict[str, int]
    skipped: dict[str, int]
    variants: list[VariantRow]
