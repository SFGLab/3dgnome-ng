"""The inputs of each interaction block, read from the positioned cluster graph.

After the coarse stages place the cluster tree, each interaction block (IB) is reconstructed
alone. This module copies everything one IB needs into a `Seeded` of plain arrays: the expected
distances, the contact heatmaps, the CTCF orientations and the anchor seed positions.
"""

from __future__ import annotations

import copy
import math
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from gnome3d import log
from gnome3d.hierarchy import Level, set_level
from gnome3d.pipeline import Orientation, Seeded
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse import CoarseState
from gnome3d.pipeline.ib.arcs import hilbert_start, run_arcs_problem, walk_start
from gnome3d.pipeline.ib.maxent import anchor_draw
from gnome3d.pipeline.ib.spans import anchor_spans, grid_edges, merge_nodes, uniform_heatmap
from gnome3d.settings import Settings
from gnome3d.tracks import slice_intervals
from gnome3d.types import (
    CompartmentInterval,
    F32Array,
    F64Array,
    I8Array,
    I64Array,
)
from gnome3d.util import seed_rng

LOG = log.get("skeleton")

# The noise step of the arcs stage is fixed.
_ARCS_NOISE: float = 0.005

_ORN_CODE: dict[str, int] = {"L": Orientation.LEFT, "R": Orientation.RIGHT}


def _heat_signal_negligible(
    settings: Settings, subanchor_heat_raw: F32Array, n: int, ib_label: str = ""
) -> bool:
    """Returns True when the subanchor heat of an IB is too sparse to change the structure.

    The share of active pairs is an upper bound on the mean reduction of the target distances by
    the heat term. Under `subanchor_heat_min_reduction` the IB runs without the ESTIMATE_DIST
    stage. A threshold of 0.0 turns the check off.
    """
    thresh = float(settings.subanchor_heat_min_reduction)
    if thresh <= 0.0:
        return False
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return True
    iu = np.triu_indices(n, k=1)
    n_active = int(np.count_nonzero(subanchor_heat_raw[iu] > 0.0))
    frac = n_active / n_pairs
    if frac < thresh:
        LOG.info(
            "%sheat-dist skipped: %d/%d pairs active (%.3g < %.3g min reduction)",
            f"{ib_label}  " if ib_label else "",
            n_active,
            n_pairs,
            frac,
            thresh,
        )
        return True
    return False


@dataclass(frozen=True)
class IBSeed:
    """The seed of one IB: its id, its chromosome, the `Seeded` inputs and whether its chain has
    an ESTIMATE_DIST stage.
    """

    ib_id: str
    chr_: str
    seed: Seeded
    wants_heat: bool


def _collect_ib_work(
    state: CoarseState, chr_: str, seg_level: dict[str, list[int]]
) -> list[tuple[str, int, list[int]]]:
    """Lists the IBs of one chromosome that have more than one anchor and seeds each anchor at
    its IB centroid, unless a segment scope arc pass placed them.

    Returns `(ib_id, ib_idx, active_region)` tuples. Every write to the shared graph happens
    here, before the build threads read it.
    """
    clusters = state.clusters
    segs = seg_level.get(chr_, [])
    ibs: list[int] = []
    for seg_idx in segs:
        ibs.extend(clusters[seg_idx].children)
    n_ibs = len(ibs)

    work: list[tuple[str, int, list[int]]] = []
    for ib_i, ib_idx in enumerate(ibs):
        ib = clusters[ib_idx]
        active_region = list(ib.children)
        ib_id = f"{chr_} IB {ib_i + 1}/{n_ibs}"
        if len(active_region) <= 1:
            LOG.info("%s  (%d anchors - skip)", ib_id, len(active_region))
            continue
        # IBs partition the anchors. These writes do not overlap.
        for a_idx in active_region:
            clusters[a_idx].pos = ib.pos.copy()
        work.append((ib_id, ib_idx, active_region))
    return work


def _anchor_nodes(state: CoarseState, work: list[tuple[str, int, list[int]]]) -> I64Array:
    """Returns the arcs node of each anchor of a chromosome under `uniform_chain`, in the order
    of `work`. The anchors of a block with their midpoints in one bead of its chain share a
    node. Nodes are numbered along the chromosome.
    """
    clusters = state.clusters
    target = int(state.s.target_bp_per_subanchor)
    out: list[I64Array] = []
    base = 0
    for _, _, ar in work:
        genomic = [(clusters[a].start, clusters[a].end, clusters[a].genomic_pos) for a in ar]
        _lo, _hi, knot = anchor_spans(genomic, grid_edges(genomic, target))
        _beads, rank = np.unique(knot, return_inverse=True)
        out.append(rank.astype(np.int64) + base)
        base += int(rank.max()) + 1
    return np.concatenate(out)


def joint_arcs_solve(
    state: CoarseState, chr_: str, work: list[tuple[str, int, list[int]]], seed_offset: int
) -> None:
    """Solves every anchor of one chromosome as one arcs problem and writes the result into the
    cluster graph.

    The anchors of a block start at the block's centroid, on a walk around it under
    `arcs_start = walk`, or from one maxent draw under `arcs_start = maxent`. With `solver_iters`
    at zero they stay at the start. The target matrix is built over the whole chromosome. The
    loops and the contact background act across blocks.
    """
    s = state.s
    clusters = state.clusters
    active_all = [a for _, _, ar in work for a in ar]
    if len(active_all) < 2:
        return
    seed = (zlib.crc32(chr_.encode()) * 2_654_435_761 + 40_503 + seed_offset) & 0x7FFFFFFF
    seed_rng(seed)
    law = s.polymer_law()
    starts: list[F32Array] = []
    for _, ib_idx, ar in work:
        cen = np.tile(clusters[ib_idx].pos.astype(np.float32), (len(ar), 1))
        if s.arcs_start == "walk":
            mids = np.array([clusters[a].genomic_pos for a in ar], dtype=np.int64)
            cen = walk_start(cen, mids, law)
        starts.append(cen)
    pos0 = np.ascontiguousarray(np.concatenate(starts), dtype=np.float32)
    if s.arcs_start == "hilbert":
        # One curve over the anchors of the chromosome, centred on the centroid of the block layout.
        mids_all = np.array([clusters[a].genomic_pos for a in active_all], dtype=np.int64)
        pos0 = hilbert_start(pos0, mids_all, law)
    anchor_heat: F64Array | None = None
    if s.use_anchor_heatmap and (state.singletons or state.anchor_contacts):
        anchor_heat, _ = cb.build_contact_heatmaps(state, active_all, chr_, with_subanchor=False)
    exp_dist = cb.calc_anchor_expected_distances(state, active_all, chr_, anchor_heat, seed)
    node = _anchor_nodes(state, work) if s.uniform_chain else None
    if s.arcs_start == "maxent" and node is None:
        # One draw per conformation, placed at the centroid of the block layout.
        mids_all = np.array([clusters[a].genomic_pos for a in active_all], dtype=np.float64)
        draw = anchor_draw(exp_dist, mids_all, law, np.random.default_rng([seed, 7]))
        pos0 = np.ascontiguousarray(draw + pos0.mean(axis=0), dtype=np.float32)
    arc_w = cb.calc_anchor_arc_weights(state, active_all, chr_, seed)
    activity = cb.calc_anchor_activity(state, active_all, chr_)
    anchor_genomic = [
        (clusters[a].start, clusters[a].end, clusters[a].genomic_pos) for a in active_all
    ]
    if node is not None:
        # Under `uniform_chain` the anchors with their midpoints in one bead are one node.
        n_nodes = int(node.max()) + 1
        size = np.bincount(node, minlength=n_nodes).astype(np.float64)
        exp_dist, arc_w = merge_nodes(exp_dist, arc_w, node)
        if activity is not None:
            activity = (np.bincount(node, weights=activity, minlength=n_nodes) / size).astype(
                np.float32
            )
        g = np.array(anchor_genomic, dtype=np.int64)
        lo = np.full(n_nodes, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.zeros(n_nodes, dtype=np.int64)
        np.minimum.at(lo, node, g[:, 0])
        np.maximum.at(hi, node, g[:, 1])
        mid = np.bincount(node, weights=g[:, 2].astype(np.float64), minlength=n_nodes) / size
        anchor_genomic = [(int(a), int(b), int(m)) for a, b, m in zip(lo, hi, mid, strict=True)]
        start = np.stack(
            [np.bincount(node, weights=pos0[:, c], minlength=n_nodes) / size for c in range(3)],
            axis=1,
        )
        if s.arcs_start == "maxent":
            draw = anchor_draw(exp_dist, mid, law, np.random.default_rng([seed, 7]))
            start = draw + start.mean(axis=0)
        pos0 = np.ascontiguousarray(start, dtype=np.float32)
    # The per block stage passes anchors through at chromosome scope. The joint call runs at
    # block scope on a copy.
    s_joint = copy.copy(s)
    s_joint.arcs_scope = "block"
    s_joint.arcs_start = "centroid"
    score, pos = run_arcs_problem(
        {
            "anchor_pos": pos0,
            "exp_dist": exp_dist,
            "arc_w": arc_w,
            "activity": activity,
            "step_size": _ARCS_NOISE,
            "settings": s_joint,
            "seed": seed,
            "anchor_genomic": anchor_genomic,
        }  # type: ignore[arg-type]
    )
    for k, a in enumerate(active_all):
        clusters[a].pos = np.asarray(pos[k if node is None else node[k]], dtype=np.float64)
    log.status(
        LOG,
        "joint arcs solve %s: %d anchors in %d blocks, energy %.4g",
        chr_,
        len(active_all),
        len(work),
        score,
    )
    if node is not None:
        log.status(LOG, "joint arcs solve %s: %d nodes of the uniform chain", chr_, len(pos))


def _build_ib_seed(
    state: CoarseState, chr_: str, item: tuple[str, int, list[int]], seed_offset: int
) -> IBSeed:
    """Builds the `Seeded` of one IB. Reads the shared graph and does not write it. Threads
    may run it.
    """
    ib_id, ib_idx, active_region = item
    seed, wants_heat = seed_for_ib(state, chr_, ib_idx, active_region, seed_offset, ib_id)
    return IBSeed(ib_id=ib_id, chr_=chr_, seed=seed, wants_heat=wants_heat)


def gather_all_ib_seeds(state: CoarseState, seed_offset: int = 0) -> list[IBSeed]:
    """Gathers the `Seeded` of every IB of the positioned graph, in a fixed order.

    The heatmap builds run on `settings.heatmap_workers` threads.
    """
    seg_level = set_level(
        Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
    )
    items: list[tuple[str, tuple[str, int, list[int]]]] = []
    for chr_ in state.chrs:
        work = _collect_ib_work(state, chr_, seg_level)
        if state.s.arcs_scope == "chromosome":
            joint_arcs_solve(state, chr_, work, seed_offset)
        items.extend((chr_, w) for w in work)
    workers = max(1, int(getattr(state.s, "heatmap_workers", 1)))
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="heatbuild") as ex:
            return list(ex.map(lambda it: _build_ib_seed(state, it[0], it[1], seed_offset), items))
    return [_build_ib_seed(state, chr_, w, seed_offset) for chr_, w in items]


def gather_ib_seeds(
    state: CoarseState, chr_: str, seg_level: dict[str, list[int]], seed_offset: int = 0
) -> list[IBSeed]:
    """Gathers the `Seeded` of every IB of one chromosome from the positioned graph. Draws no
    random number.
    """
    work = _collect_ib_work(state, chr_, seg_level)
    if state.s.arcs_scope == "chromosome":
        joint_arcs_solve(state, chr_, work, seed_offset)
    return [_build_ib_seed(state, chr_, w, seed_offset) for w in work]


def seed_for_ib(
    state: CoarseState,
    chr_: str,
    ib_idx: int,
    active_region: list[int],
    seed_offset: int = 0,
    ib_label: str = "",
) -> tuple[Seeded, bool]:
    """Gathers the `Seeded` inputs of one IB as plain arrays that hold no cluster reference."""
    s = state.s
    clusters = state.clusters
    a = len(active_region)

    anchor_heat: F64Array | None = None
    subanchor_heat_raw: F32Array | None = None
    # Under `uniform_chain` the subanchor heatmap is binned to the beads of the uniform chain.
    default_bins = bool(state.s.use_subanchor_heatmap) and not s.uniform_chain
    if (state.s.use_anchor_heatmap or state.s.use_subanchor_heatmap) and (
        state.singletons or state.anchor_contacts
    ):
        anchor_heat, subanchor_heat_raw = cb.build_contact_heatmaps(
            state, active_region, chr_, with_subanchor=default_bins
        )
        if s.uniform_chain and s.use_subanchor_heatmap and state.singletons:
            edges = grid_edges(
                [(clusters[ci].start, clusters[ci].end, 0) for ci in active_region],
                int(s.target_bp_per_subanchor),
            )
            subanchor_heat_raw = uniform_heatmap(state.singletons, chr_, edges)

    # The seed of the block also selects the conformation for the loop dropout.
    block_seed = (ib_idx * 2_654_435_761 + 40_503 + seed_offset) & 0x7FFFFFFF
    exp_dist = cb.calc_anchor_expected_distances(
        state, active_region, chr_, anchor_heat, block_seed
    )
    arc_w = cb.calc_anchor_arc_weights(state, active_region, chr_, block_seed)
    activity = cb.calc_anchor_activity(state, active_region, chr_)

    anchor_seed_pos = np.array([clusters[ci].pos for ci in active_region], dtype=np.float32)
    anchor_genomic = [
        (clusters[ci].start, clusters[ci].end, clusters[ci].genomic_pos) for ci in active_region
    ]

    orientations: I8Array | None = None
    anchor_neighbors: dict[int, list[int]] | None = None
    anchor_neighbor_weights: dict[int, list[float]] | None = None
    if state.s.use_ctcf_motif and chr_:
        orientations = np.array(
            [
                _ORN_CODE.get(clusters[ci].orientation or "N", Orientation.NONE)
                for ci in active_region
            ],
            dtype=np.int8,
        )
        cluster_to_k = {ci: k for k, ci in enumerate(active_region)}
        chr_arcs = state.arcs.get(chr_, [])
        anchor_neighbors = {k: [] for k in range(a)}
        anchor_neighbor_weights = {k: [] for k in range(a)}
        for k, ci in enumerate(active_region):
            for arc_local in clusters[ci].arcs:
                if arc_local >= len(chr_arcs):
                    continue
                arc = chr_arcs[arc_local]
                if arc.factor != 0:
                    # The motif term is CTCF's. A loop of another factor has no orientation.
                    continue
                other_ci = arc.end if arc.start == ci else arc.start
                if other_ci in cluster_to_k:
                    anchor_neighbors[k].append(cluster_to_k[other_ci])
                    anchor_neighbor_weights[k].append(math.sqrt(max(arc.score, 0)))

    track_compartments: list[CompartmentInterval] | None = None
    if chr_ and anchor_genomic and s.use_compartments:
        ib_lo = anchor_genomic[0][0]
        ib_hi = anchor_genomic[-1][1]
        track_compartments = slice_intervals(state.compartments.get(chr_, []), ib_lo, ib_hi)

    wants_heat = (
        subanchor_heat_raw is not None
        and bool(state.s.use_subanchor_heatmap)
        and float(subanchor_heat_raw.mean(dtype=np.float64)) >= 1e-6  # summed in float64
        and not _heat_signal_negligible(
            state.s,
            subanchor_heat_raw,
            int(subanchor_heat_raw.shape[0]) if s.uniform_chain else a,
            ib_label,
        )
    )

    seed = Seeded(
        settings=s,
        # Not from `hash()`, which is salted per process.
        seed=block_seed,
        anchor_seed_pos=anchor_seed_pos,
        exp_dist=exp_dist,
        arc_w=arc_w,
        activity=activity,
        orientations=orientations,
        anchor_neighbors=anchor_neighbors,
        anchor_neighbor_weights=anchor_neighbor_weights,
        subanchor_heat_raw=subanchor_heat_raw,
        anchor_genomic=anchor_genomic,
        step_size_arcs=_ARCS_NOISE,
        track_compartments=track_compartments,
        chrom=chr_,
    )
    return seed, wants_heat
