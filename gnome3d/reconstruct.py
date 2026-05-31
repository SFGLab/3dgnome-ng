"""
Pipeline driver: skeleton -> Dag -> executor -> beads.

The top-level orchestration that ties the coupled skeleton (per-IB ``Seeded``
states) to the isolated stage pipeline.  `simulate` / `cli` call this; it returns
``dict[chr -> list[BeadOut]]`` (the former Solver reconstruction, now a pipeline).

It is a leaf consumer — it imports `skeleton` and `pipeline.stages` but nothing
imports it back, so there is no cycle (skeleton -> pipeline, stages -> mc_*).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from . import skeleton
from .pipeline import Dag, Node
from .pipeline.executor import BatchExecutor, Executor, SerialExecutor
from .pipeline.stage import StageKind
from .pipeline.stages import ArcsStage, DensifyStage, HeatDistStage, SmoothStage

if TYPE_CHECKING:
    from .data import ContactData
    from .settings import Settings
    from .types import BeadOut, BedRegion

_SMOOTH = StageKind.SMOOTH.value

# Well-separated RNG offset per ensemble member, so structure i's per-IB seeds
# (and coarse seed) are distinct from structure j's.  See skeleton.build_seeds.
MEMBER_SEED_STRIDE = 2_000_003


def pick_executor(settings: Settings) -> Executor:
    """Choose the executor for `settings`.

    JAX backend -> `BatchExecutor`, with the batched (GPU) kinds taken from the
    per-term ``mc_backend_apply_to_*`` flags so behavior matches the old paths:
    smooth/heat batched while arcs stays numba (default) reproduces JaxSolver;
    flipping ``apply_to_arcs`` opts arcs into the batched kernel too.  Anything
    else (numba backend, or the small-IB-boost case whose per-IB springs the
    batched kernels don't carry) -> `SerialExecutor` (numba, per-IB settings)."""
    backend = str(settings.mc_backend).strip().lower()
    if backend != "jax" or bool(settings.use_small_ib_boost):
        return SerialExecutor()

    batch_kinds: set[StageKind] = set()
    if bool(settings.mc_backend_apply_to_smooth):
        batch_kinds.add(StageKind.SMOOTH)
        batch_kinds.add(StageKind.HEAT_DIST)  # heat-dist is the smooth kernel (dry)
    if bool(settings.mc_backend_apply_to_arcs):
        batch_kinds.add(StageKind.ARCS)
    return BatchExecutor(batch_kinds)


def _node_id(ib_id: str, stage_name: str) -> str:
    return f"{ib_id} :: {stage_name}"


def build_dag(ibseeds: list[skeleton.IBSeed]) -> Dag:
    """Assemble one linear chain per IB: arcs -> densify -> [heat] -> smooth.
    HEAT_DIST is included only when the skeleton kept it (`IBSeed.wants_heat`,
    the sparse-signal early-out).  Chains are independent — the skeleton seed
    feeds each chain's root."""
    nodes: dict[str, Node] = {}
    seeds: dict[str, object] = {}
    for ibs in ibseeds:
        chain: list[tuple[StageKind, object]] = [
            (StageKind.ARCS, ArcsStage()),
            (StageKind.DENSIFY, DensifyStage()),
        ]
        if ibs.wants_heat:
            chain.append((StageKind.HEAT_DIST, HeatDistStage()))
        chain.append((StageKind.SMOOTH, SmoothStage()))

        prev: str | None = None
        for kind, stage in chain:
            nid = _node_id(ibs.ib_id, kind.value)
            nodes[nid] = Node(nid, stage, () if prev is None else (prev,))  # type: ignore[arg-type]
            if prev is None:
                seeds[nid] = ibs.seed
            prev = nid
    return Dag(nodes=nodes, seeds=seeds)  # type: ignore[arg-type]


def reconstruct(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    executor: Executor | None = None,
    seed_offset: int = 0,
) -> dict[str, list[BeadOut]]:
    """Reconstruct via the task-DAG pipeline.  Returns one bead list per chr,
    sorted by genomic start (matching `Solver.get_leaf_positions`).

    `seed_offset` selects the ensemble member (distinct structures from the same
    inputs); 0 is the canonical single structure."""
    ibseeds = skeleton.build_seeds(settings, data, chrs, region, seed_offset=seed_offset)
    dag = build_dag(ibseeds)
    outputs = (executor or SerialExecutor()).run(dag)

    per_chr: dict[str, list[BeadOut]] = defaultdict(list)
    for ibs in ibseeds:
        per_chr[ibs.chr_].extend(outputs[_node_id(ibs.ib_id, _SMOOTH)].beads)
    return {chr_: sorted(beads, key=lambda b: b.start) for chr_, beads in per_chr.items()}
