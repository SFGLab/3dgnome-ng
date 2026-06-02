"""
Task-DAG abstraction for IB reconstruction.

`solver.py` fuses three unrelated jobs: building the coupled *skeleton*
(hierarchy + inter-chr heatmap + IB positioning), the *per-IB reconstruction*
(arcs -> densify -> heat-dist -> smooth), and *write-back* into shared mutable
cluster state.  The third tangles through the second, which is why the GPU path
had to re-implement the pipeline instead of just swapping a strategy.

The structural fact this package is built on: once the skeleton has placed the
IB seeds, **an IB reconstruction is a pure, isolated computation** —
``(anchors, arcs, singletons, orientations, seed_pos, settings, rng) -> beads``,
with no cross-IB shared state (which is why it already threads safely).  So we
model the work as a dataflow DAG of typed stages and let an executor decide
*how* to run it:

    skeleton  ->  Dag of Nodes  ->  Executor  ->  {node -> State}
    (coupled,      (pure stages,     (serial | batched | ensemble)
     sequential)    explicit deps)

Layout:
  * `state`    — sealed frozen state progression (Seeded..Smoothed) + Orientation
  * `stage`    — StageKind, the Stage protocol, Problem/Result
  * `dag`      — Node, Dag, NodeId (the dataflow graph + ready/inputs_for)
  * `registry` — per-kind serial/batch runner wiring (no kernel imports)
  * `executor` — the Executor protocol (+ Serial/Batch executors)
  * `stages/`  — concrete stage implementations (added with the port)

Authorized *structural* divergence from the reference's mutable ``LooperSolver``
shape — NOT a numerical one: every stage calls the same validated kernel, so the
math per stage is unchanged and the divergence is gated by fixed-seed exact bead
comparison (see `[[feedback_no_made_up_solutions]]`).
"""

from __future__ import annotations

from gnome3d.pipeline.dag import Dag, Node, NodeId
from gnome3d.pipeline.executor import Executor
from gnome3d.pipeline.registry import BatchRunner, KindRunners, SerialRunner, register, runners_for
from gnome3d.pipeline.stage import Problem, Result, Stage, StageKind
from gnome3d.pipeline.state import (
    AnchorMapEntry,
    Arced,
    CoarsePhase,
    Densified,
    DistEstimated,
    Orientation,
    Seeded,
    Smoothed,
    State,
)

__all__ = [
    "AnchorMapEntry",
    "Arced",
    "BatchRunner",
    "CoarsePhase",
    "Dag",
    "Densified",
    "Executor",
    "DistEstimated",
    "KindRunners",
    "Node",
    "NodeId",
    "Orientation",
    "Problem",
    "Result",
    "Seeded",
    "SerialRunner",
    "Smoothed",
    "Stage",
    "StageKind",
    "State",
    "register",
    "runners_for",
]
