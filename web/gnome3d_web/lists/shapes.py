"""The shape of a shared job list."""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class ListedJob(TypedDict):
    url: str
    label: str
    name: str
    at: float
    edited: NotRequired[float]


class SharedList(TypedDict):
    code: str
    private: bool
    jobs: list[ListedJob]
