"""
Runner registry: how a stage kind's work actually executes.

Each kind registers one or both:
  * ``serial`` - run ONE problem (numba, or JAX at K=1); used by SerialExecutor,
    one node at a time.
  * ``batch``  - run a GROUP of same-(kind,bucket) problems in one shot (the JAX
    ``mc_*_jax_batch`` entries); used by BatchExecutor.

``DENSIFY`` registers only ``serial`` (no GPU kernel); the batched executor just
maps it.  Kept free of kernel imports so `pipeline` stays dependency-light - the
`mc_*` / wiring modules call `register`.
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
