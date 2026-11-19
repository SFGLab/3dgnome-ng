"""The task DAG of the reconstruction.

Once the coarse stages place the IB seeds, the reconstruction of an IB is a pure function of
its inputs. The work is a DAG of typed stages and an executor decides how to run it.

  state     the frozen states from Seeded to Smoothed
  stage     StageKind and the Stage protocol
  dag       Node and Dag
  registry  the serial and the batch runner of each kind
  executor  the executors
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
