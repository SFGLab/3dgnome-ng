"""The shapes of the files of a job directory and of what a job states of them."""

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, TypedDict

State = Literal["queued", "running", "done", "failed"]


class Stage(TypedDict):
    stage: str
    at: float


class Status(TypedDict):
    """status.json. A job has `state` from the moment it waits and gains the rest as it runs."""

    state: State
    stages: NotRequired[list[Stage]]
    stage: NotRequired[str]
    started: NotRequired[float]
    updated: NotRequired[float]
    finished: NotRequired[float]
    region: NotRequired[str]
    n_svs: NotRequired[int]
    skipped_svs: NotRequired[int]
    loops: NotRequired[dict[str, int]]
    error: NotRequired[str]
    log_tail: NotRequired[str]


class OwnStated(TypedDict):
    """The own data of a request as a job page states it, the factors of the
    loops and the kinds of the other files."""

    loops: list[str]
    anchors: bool
    contacts: bool
    enhancers: bool
    loops_from: NotRequired[str]
    enhancers_from: NotRequired[str]


class RequestStated(TypedDict):
    """The request of a job without the places of its uploads. The expression is the id of its
    source. A request of the first form of the site has no background, overrides or expression."""

    cell_type: str
    region: str
    factors: str
    sv_set: str
    sample: str
    haplotype: int
    n_models: int
    hicdiffusion: NotRequired[bool]
    background: NotRequired[str]
    chain: NotRequired[str]
    overrides: NotRequired[dict[str, str]]
    expression: NotRequired[str]
    own: NotRequired[OwnStated]


class FileSize(TypedDict):
    name: str
    size: int


class JobUrl(TypedDict):
    url: str
