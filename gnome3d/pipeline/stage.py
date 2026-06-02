"""
Stages: pure state->state transforms that adapt to/from a kernel.

A stage never holds a kernel.  It declares its ``kind`` (what an executor groups
on to batch) and only knows how to turn its input state(s) into a kernel
*problem* and fold the *result* into its output state.  The actual numba/JAX
kernel comes from the per-kind runner registry (`registry.py`), so adding a
batched backend for a kind is a registration, not a code change here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from gnome3d.pipeline.state import State


class StageKind(StrEnum):
    """What a stage does.  The executor groups *ready* nodes by ``(kind, bucket)``
    to batch them and dispatches to the runner registered for the kind."""

    # --- coarse positioning (the cluster-tree spine, upstream of IBs) ---
    COARSE = "coarse"  # one cluster-graph level: build/position in place (serial)

    # --- per-IB reconstruction ---
    ARCS = "arcs"  # position anchors via arc-spring MC (anchor-shaped)
    DENSIFY = "densify"  # insert subanchor beads (CPU, reshapes)
    ESTIMATE_DIST = "estimate_dist"  # estimate subanchor distance target (bead-shaped)
    SMOOTH = "smooth"  # final smooth MC (bead-shaped)


# A stage's kernel input/output, opaque and kernel-shaped: a "problem" is the
# self-contained dict the `mc_*_jax_batch` entries consume; a "result" is
# whatever that kernel returns.  Stages own the interpretation.
Problem = dict[str, Any]
Result = Any


@runtime_checkable
class Stage(Protocol):
    """One node's transform.  Consumes its dependency states, turns them into a
    kernel ``Problem``, and folds the ``Result`` into its output ``State``.

    Linear IB stages read a single upstream state (``inputs[0]``); the tuple form
    keeps the door open for fan-in (merge) nodes without reshaping this contract.
    """

    kind: StageKind

    def bucket(self, inputs: tuple[State, ...]) -> int:
        """Shape class for batch grouping (anchor count for ARCS, bead count for
        ESTIMATE_DIST/SMOOTH) — computed from the input state(s)."""
        ...

    def to_problem(self, inputs: tuple[State, ...]) -> Problem:
        """Build this stage's self-contained kernel input."""
        ...

    def apply(self, inputs: tuple[State, ...], result: Result) -> State:
        """Produce the output state by folding in the kernel result."""
        ...
