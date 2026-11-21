"""The shapes of the record of an upload and of what the dialog is told about it."""

from __future__ import annotations

from typing_extensions import NotRequired, TypedDict


class Meta(TypedDict):
    """meta.json of an upload. A check adds `ok`. A contact map read by a link has the link, the
    name and the size of the remote file. Loops have their factor."""

    id: str
    kind: str
    name: str
    size: int
    created: float
    ok: NotRequired[bool]
    factor: NotRequired[str]
    link: NotRequired[str]
    link_name: NotRequired[str]
    link_size: NotRequired[str]


class ProblemFound(TypedDict):
    line: int
    found: str
    fix: str


class UploadResult(TypedDict):
    id: str
    kind: str
    name: str
    size: int
    ok: bool
    summary: str
    error: ProblemFound | None
    # The samples of an expression table and whether its values are whole numbers.
    samples: list[str]
    counts: bool


# A loop of an upload inside a region: its two anchor middles and its count.
LoopDrawn = tuple[int, int, int]
