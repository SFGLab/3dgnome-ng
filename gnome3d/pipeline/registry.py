"""The runners of each stage kind.

`serial` runs one problem. `batch` runs a group of problems of one kind and bucket in one
launch. DENSIFY has no batch runner. This module imports no kernel. The kernel modules call
`register`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gnome3d.pipeline.stage import Problem, Result, StageKind

SerialRunner = Callable[[Problem], Result]
BatchRunner = Callable[[Sequence[Problem]], list[Result]]


@dataclass
class KindRunners:
    serial: SerialRunner | None = None
    batch: BatchRunner | None = None


_REGISTRY: dict[StageKind, KindRunners] = {}


def register(
    kind: StageKind,
    *,
    serial: SerialRunner | None = None,
    batch: BatchRunner | None = None,
) -> None:
    runners = _REGISTRY.setdefault(kind, KindRunners())
    if serial is not None:
        runners.serial = serial
    if batch is not None:
        runners.batch = batch


def runners_for(kind: StageKind) -> KindRunners:
    return _REGISTRY[kind]
