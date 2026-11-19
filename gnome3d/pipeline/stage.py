"""Stages turn states into kernel problems and kernel results into states. The kernel of a kind
comes from the registry.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from gnome3d.pipeline.state import State


class StageKind(StrEnum):
    """The kind of a stage. The executor groups ready nodes by `(kind, bucket)`."""

    COARSE = "coarse"  # one level of the cluster tree, positioned in place

    ARCS = "arcs"  # anchors by arc springs
    DENSIFY = "densify"  # inserts subanchors
    ESTIMATE_DIST = "estimate_dist"  # subanchor distance targets
    SMOOTH = "smooth"  # final smooth MC


# A problem is the dict a kernel takes and a result is what the kernel returns.
Problem = dict[str, Any]
Result = Any


@runtime_checkable
class Stage(Protocol):
    """The transform of one node, from the states of its dependencies to its output state."""

    kind: StageKind

    def bucket(self, inputs: tuple[State, ...]) -> int:
        """Returns the shape class for batch grouping, the anchor count for ARCS and the bead
        count for ESTIMATE_DIST and SMOOTH.
        """
        ...

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        """Builds the kernel input."""
        ...

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        """Returns the output state from the kernel result."""
        ...
