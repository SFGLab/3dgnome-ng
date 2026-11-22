"""The inputs of each interaction block, read from the positioned cluster graph.

After the coarse stages place the cluster tree, each interaction block (IB) is reconstructed
alone. This module copies everything one IB needs into a `Seeded` of plain arrays: the expected
distances, the contact heatmaps, the CTCF orientations and the anchor seed positions.
"""

from __future__ import annotations

import copy
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from gnome3d import log
from gnome3d.hierarchy import Level, set_level
from gnome3d.pipeline import Orientation, Seeded
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse import CoarseState
from gnome3d.pipeline.ib.arcs import hilbert_start, run_arcs_problem, walk_start
from gnome3d.pipeline.ib.cells import available_resolution, hold_by_cells, read_cells
from gnome3d.pipeline.ib.components import (
    LOCUS_BP,
    ArmComponents,
    hold_loci,
    read_components,
)
from gnome3d.pipeline.ib.maxent import anchor_draw
from gnome3d.pipeline.ib.nodes import (
    ContactNodes,
    chain_share,
    contact_nodes,
    draw_ends,
    locus_beads,
    mean_loop_target,
    node_targets,
    node_weights,
    reach_pairs,
)
from gnome3d.pipeline.ib.spans import anchor_spans, grid_edges, uniform_heatmap
from gnome3d.settings import Settings
from gnome3d.tracks import slice_intervals
from gnome3d.types import (
    CompartmentInterval,
    F32Array,
    F64Array,
    I8Array,
    I32Array,
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


class ChromNodes(NamedTuple):
    """The nodes of one conformation of a chromosome on the uniform chain, solved.

    `orient` is the `Orientation` code of each node, that of the anchor of its CTCF loop with
    the most PETs. `nbrs` and `nbr_w` list the nodes that a CTCF loop joins to each node and
    the root of the PETs of the pair.
    """

    nodes: ContactNodes
    pos: F64Array
    orient: I8Array
    nbrs: list[list[int]]
    nbr_w: list[list[float]]


def locus_extra(
    edges: list[I64Array],
    anchor_block: I32Array,
    ends: I32Array,
    beads: I64Array,
    s: Settings,
) -> I64Array | None:
    """Returns the locus beads that `[springs] locus_bead` makes nodes, counted along the
    chromosome, or None for `node`.

    `middle` gives every locus of the chains its locus bead. `fill` gives one to the loci where
    no loop end of the conformation is.
    """
    if s.locus_bead == "node":
        return None
    extra, loci = locus_beads(edges, LOCUS_BP)
    if s.locus_bead == "middle":
        return extra
    no_weight: F64Array = np.ones(int(ends.shape[0]), dtype=np.float64)
    with_node: I64Array = contact_nodes(edges, anchor_block, ends, beads, no_weight).mid // LOCUS_BP
    keep: I64Array = extra[~np.isin(loci, with_node)]
    return keep


def solve_nodes(
    state: CoarseState, chr_: str, work: list[tuple[str, int, list[int]]], seed: int
) -> ChromNodes:
    """Draws the contact beads of one conformation of a chromosome and solves them as one arcs
    problem, `design/uniform-chain.md`.

    A node starts at the centroid of its block, on a walk around it under `arcs_start = walk`,
    or from one maxent draw under `arcs_start = maxent`. Consecutive nodes of a block are held
    within the chain between them. The excluded volume acts at the radius of the arcs stage.
    """
    s = state.s
    clusters = state.clusters
    law = s.polymer_law()
    target = int(s.target_bp_per_subanchor)
    if float(s.factory_weight) > 0.0:
        raise ValueError(
            "[main] uniform_chain is on with [factories] weight above zero. The factory term"
            " is not stated on the nodes of the uniform chain. Set the weight to 0."
        )
    active_all = [a for _, _, ar in work for a in ar]
    edges: list[I64Array] = []
    lo_parts: list[I32Array] = []
    hi_parts: list[I32Array] = []
    block_parts: list[I32Array] = []
    for b, (_, _, ar) in enumerate(work):
        genomic = [(clusters[a].start, clusters[a].end, clusters[a].genomic_pos) for a in ar]
        e = grid_edges(genomic, target)
        lo_b, hi_b, _knot = anchor_spans(genomic, e)
        edges.append(e)
        lo_parts.append(lo_b)
        hi_parts.append(hi_b)
        block_parts.append(np.full(len(ar), b, dtype=np.int32))
    anchor_block: I32Array = np.concatenate(block_parts)

    mids, arcs = cb.active_arcs(state, active_all, chr_, seed)
    ends: I32Array = np.array([(a[0], a[1]) for a in arcs], dtype=np.int32).reshape(-1, 2)
    score = np.array([a[2] for a in arcs], dtype=np.float64)
    factor = np.array([a[3] if len(a) > 3 else 0 for a in arcs], dtype=np.int64)
    strength: F64Array = np.array(
        [
            law.arc_strength(int(sc), abs(mids[int(i)] - mids[int(j)]), int(f))
            for (i, j), sc, f in zip(ends, score, factor, strict=True)
        ],
        dtype=np.float64,
    )
    beads = draw_ends(
        np.concatenate(lo_parts),
        np.concatenate(hi_parts),
        ends,
        np.random.default_rng([seed, 11]),
    )
    nodes = contact_nodes(
        edges, anchor_block, ends, beads, strength, locus_extra(edges, anchor_block, ends, beads, s)
    )
    m = int(nodes.mid.shape[0])

    # The orientation of a node and the nodes its CTCF loops join it to.
    orient: I8Array = np.full(m, int(Orientation.NONE), dtype=np.int8)
    heaviest = np.zeros(m, dtype=np.float64)
    pets: dict[tuple[int, int], float] = {}
    for k in np.flatnonzero((factor == 0) & (nodes.loop_node[:, 0] >= 0)):
        na, nb = int(nodes.loop_node[k, 0]), int(nodes.loop_node[k, 1])
        for node, anchor in ((na, int(ends[k, 0])), (nb, int(ends[k, 1]))):
            if score[k] > heaviest[node]:
                heaviest[node] = score[k]
                code = _ORN_CODE.get(clusters[active_all[anchor]].orientation or "N")
                orient[node] = int(Orientation.NONE if code is None else code)
        key = (min(na, nb), max(na, nb))
        pets[key] = pets.get(key, 0.0) + float(score[k])
    nbrs: list[list[int]] = [[] for _ in range(m)]
    nbr_w: list[list[float]] = [[] for _ in range(m)]
    for (na, nb), total in sorted(pets.items()):
        w = math.sqrt(max(total, 0.0))
        nbrs[na].append(nb)
        nbr_w[na].append(w)
        nbrs[nb].append(na)
        nbr_w[nb].append(w)

    centre: F64Array = np.array(
        [clusters[work[int(b)][1]].pos for b in nodes.block], dtype=np.float64
    ).reshape(m, 3)
    if m < 2:
        return ChromNodes(nodes, centre, orient, nbrs, nbr_w)

    exp_dist = node_targets(nodes, law, s)
    if s.use_contact_background and float(s.background_weight) > 0.0:
        if not s.data_contact_map:
            log.status(
                LOG,
                "joint arcs solve %s: no [data] contact_map, the nodes read no contact data",
                chr_,
            )
        elif float(s.contact_background_z) <= 0.0:
            raise ValueError(
                "[main] uniform_chain reads the cells of [data] contact_map under the Poisson"
                " test and [springs] contact_background_z is 0. Set contact_background_z above"
                " zero, or use_contact_background to no."
            )
        else:
            finest = available_resolution(str(s.data_contact_map), int(s.contact_resolution))
            if finest != int(s.contact_resolution):
                log.status(
                    LOG,
                    "joint arcs solve %s: [data] contact_map holds no map at %d bp, the cells are"
                    " read at %d bp",
                    chr_,
                    int(s.contact_resolution),
                    finest,
                )
            count, offset, expected, mapped, node_bias = read_cells(
                str(s.data_contact_map), chr_, nodes.mid, finest
            )
            exp_dist = hold_by_cells(
                exp_dist, nodes.mid, count, offset, expected, mapped, law, s, finest, node_bias
            )
    if float(s.background_weight) > 0.0:
        arms: list[ArmComponents] = []
        if s.data_components:
            track = str(s.data_components)
            if not os.path.isabs(track) and not os.path.isfile(track):
                track = os.path.join(str(s.data_dir), track)
            arms = read_components(track).get(chr_, [])
        exp_dist = hold_loci(exp_dist, nodes.mid, arms, law, s)
    exp_dist = cb.repulsion_range(exp_dist, [int(x) for x in nodes.mid], s)
    pos0: F32Array = np.ascontiguousarray(centre, dtype=np.float32)
    if s.arcs_start == "walk":
        for b in np.unique(nodes.block):
            sel = np.flatnonzero(nodes.block == b)
            pos0[sel] = walk_start(pos0[sel], nodes.mid[sel], law)
    elif s.arcs_start == "hilbert":
        pos0 = hilbert_start(pos0, nodes.mid, law)
    elif s.arcs_start == "maxent":
        draw = anchor_draw(
            exp_dist, nodes.mid.astype(np.float64), law, np.random.default_rng([seed, 7])
        )
        pos0 = np.ascontiguousarray(draw + pos0.mean(axis=0), dtype=np.float32)

    s_joint = copy.copy(s)
    s_joint.arcs_scope = "block"
    s_joint.arcs_start = "centroid"
    bond = float(law.background(target))
    node_genomic = [
        (
            int(edges[int(b)][int(k)]),
            int(edges[int(b)][int(k) + 1]),
            int(mid),
        )
        for b, k, mid in zip(nodes.block, nodes.bead, nodes.mid, strict=True)
    ]
    # Under the dropout a constant of the conformation is read from every loop of the chromosome.
    mids_all: list[int] = []
    arcs_all: list[tuple[int, int, int] | tuple[int, int, int, int]] = []
    if s.loop_dropout:
        mids_all, arcs_all = cb.active_arcs(state, active_all, chr_, None)
    derived = bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_arcs)
    if s.loop_dropout and derived and float(s.exclusion_radius_arcs) <= 0.0:
        s_joint.exclusion_radius_arcs = float(s.exclusion_auto_factor_arcs) * mean_loop_target(
            mids_all, arcs_all, law, s
        )
        log.status(
            LOG,
            "joint arcs solve %s: radius of the excluded volume %.2f, from the %d loops before"
            " the dropout",
            chr_,
            float(s_joint.exclusion_radius_arcs),
            len(arcs_all),
        )
    share: F64Array | None = None
    if s.exclusion_arcs_chain_radius != "none":
        every = m
        if s.loop_dropout:
            # The count of nodes of a conformation that keeps every loop, from a draw of its own.
            ends_all: I32Array = np.array([(a[0], a[1]) for a in arcs_all], dtype=np.int32).reshape(
                -1, 2
            )
            beads_all = draw_ends(
                np.concatenate(lo_parts),
                np.concatenate(hi_parts),
                ends_all,
                np.random.default_rng([seed, 12]),
            )
            every = int(
                contact_nodes(
                    edges,
                    anchor_block,
                    ends_all,
                    beads_all,
                    np.ones(len(arcs_all)),
                    locus_extra(edges, anchor_block, ends_all, beads_all, s),
                ).mid.shape[0]
            )
        share = chain_share(
            nodes.mid, min(int(e[0]) for e in edges), max(int(e[-1]) for e in edges), every
        )
        f = np.cbrt(share)
        log.status(
            LOG,
            "joint arcs solve %s: radius from the chain under %s, %d nodes of %d with every loop,"
            " factor of a node %.2f to %.2f, median %.2f",
            chr_,
            s.exclusion_arcs_chain_radius,
            m,
            every,
            float(f.min()),
            float(f.max()),
            float(np.median(f)),
        )
    energy, pos = run_arcs_problem(
        {
            "anchor_pos": pos0,
            "exp_dist": exp_dist,
            "arc_w": node_weights(nodes, s),
            "activity": None,
            "step_size": _ARCS_NOISE,
            "settings": s_joint,
            "seed": seed,
            "anchor_genomic": node_genomic,
            "reach": reach_pairs(nodes, bond),
            "share": share,
        }  # type: ignore[arg-type]
    )
    log.status(
        LOG,
        "joint arcs solve %s: %d nodes from %d loops of %d anchors in %d blocks, %d loops with"
        " both ends on one bead, energy %.4g",
        chr_,
        m,
        len(arcs),
        len(active_all),
        len(work),
        int((nodes.loop_node[:, 0] < 0).sum()),
        energy,
    )
    return ChromNodes(nodes, np.asarray(pos, dtype=np.float64), orient, nbrs, nbr_w)


def joint_arcs_solve(
    state: CoarseState, chr_: str, work: list[tuple[str, int, list[int]]], seed_offset: int
) -> ChromNodes | None:
    """Solves every anchor of one chromosome as one arcs problem and writes the result into the
    cluster graph.

    The anchors of a block start at the block's centroid, on a walk around it under
    `arcs_start = walk`, or from one maxent draw under `arcs_start = maxent`. With `solver_iters`
    at zero they stay at the start. The target matrix is built over the whole chromosome. The
    loops and the contact background act across blocks.

    Under `uniform_chain` the nodes of the conformation are solved in place of the anchors and
    returned. The cluster graph is not written.
    """
    s = state.s
    clusters = state.clusters
    active_all = [a for _, _, ar in work for a in ar]
    if len(active_all) < 2:
        return None
    seed = (zlib.crc32(chr_.encode()) * 2_654_435_761 + 40_503 + seed_offset) & 0x7FFFFFFF
    seed_rng(seed)
    if s.uniform_chain:
        return solve_nodes(state, chr_, work, seed)
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
    if s.arcs_start == "maxent":
        # One draw per conformation, placed at the centroid of the block layout.
        mids_all = np.array([clusters[a].genomic_pos for a in active_all], dtype=np.float64)
        draw = anchor_draw(exp_dist, mids_all, law, np.random.default_rng([seed, 7]))
        pos0 = np.ascontiguousarray(draw + pos0.mean(axis=0), dtype=np.float32)
    arc_w = cb.calc_anchor_arc_weights(state, active_all, chr_, seed)
    activity = cb.calc_anchor_activity(state, active_all, chr_)
    anchor_genomic = [
        (clusters[a].start, clusters[a].end, clusters[a].genomic_pos) for a in active_all
    ]
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
        clusters[a].pos = np.asarray(pos[k], dtype=np.float64)
    log.status(
        LOG,
        "joint arcs solve %s: %d anchors in %d blocks, energy %.4g",
        chr_,
        len(active_all),
        len(work),
        score,
    )
    return None


def _solve_chromosome(
    state: CoarseState, chr_: str, work: list[tuple[str, int, list[int]]], seed_offset: int
) -> ChromNodes | None:
    """Runs the joint arcs solve of a chromosome at chromosome scope. Returns the nodes of a
    uniform chain, None otherwise.
    """
    if state.s.arcs_scope == "chromosome":
        return joint_arcs_solve(state, chr_, work, seed_offset)
    if state.s.uniform_chain:
        raise ValueError(
            "[main] uniform_chain is on with [simulation_arcs] scope = block. The nodes of a"
            " uniform chain are solved over the chromosome. Set scope to chromosome."
        )
    return None


def _build_ib_seed(
    state: CoarseState,
    chr_: str,
    item: tuple[str, int, list[int]],
    seed_offset: int,
    nodes: ChromNodes | None = None,
    block: int = -1,
) -> IBSeed:
    """Builds the `Seeded` of one IB. Reads the shared graph and does not write it. Threads
    may run it. `nodes` and `block` are the nodes of the chromosome and the place of the IB among
    its blocks, under `uniform_chain`.
    """
    ib_id, ib_idx, active_region = item
    if state.s.uniform_chain:
        seed, wants_heat = seed_for_uniform(
            state, chr_, ib_idx, active_region, nodes, block, seed_offset, ib_id
        )
    else:
        seed, wants_heat = seed_for_ib(state, chr_, ib_idx, active_region, seed_offset, ib_id)
    return IBSeed(ib_id=ib_id, chr_=chr_, seed=seed, wants_heat=wants_heat)


def gather_all_ib_seeds(state: CoarseState, seed_offset: int = 0) -> list[IBSeed]:
    """Gathers the `Seeded` of every IB of the positioned graph, in a fixed order.

    The heatmap builds run on `settings.heatmap_workers` threads.
    """
    seg_level = set_level(
        Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
    )
    items: list[tuple[str, tuple[str, int, list[int]], ChromNodes | None, int]] = []
    for chr_ in state.chrs:
        work = _collect_ib_work(state, chr_, seg_level)
        nodes = _solve_chromosome(state, chr_, work, seed_offset)
        items.extend((chr_, w, nodes, b) for b, w in enumerate(work))
    workers = max(1, int(getattr(state.s, "heatmap_workers", 1)))

    def build(it: tuple[str, tuple[str, int, list[int]], ChromNodes | None, int]) -> IBSeed:
        return _build_ib_seed(state, it[0], it[1], seed_offset, it[2], it[3])

    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="heatbuild") as ex:
            return list(ex.map(build, items))
    return [build(it) for it in items]


def gather_ib_seeds(
    state: CoarseState, chr_: str, seg_level: dict[str, list[int]], seed_offset: int = 0
) -> list[IBSeed]:
    """Gathers the `Seeded` of every IB of one chromosome from the positioned graph. Draws no
    random number.
    """
    work = _collect_ib_work(state, chr_, seg_level)
    nodes = _solve_chromosome(state, chr_, work, seed_offset)
    return [_build_ib_seed(state, chr_, w, seed_offset, nodes, b) for b, w in enumerate(work)]


def seed_for_uniform(
    state: CoarseState,
    chr_: str,
    ib_idx: int,
    active_region: list[int],
    nodes: ChromNodes | None,
    block: int,
    seed_offset: int = 0,
    ib_label: str = "",
) -> tuple[Seeded, bool]:
    """Gathers the `Seeded` inputs of one IB of a uniform chain.

    The nodes of the block and their positions come from the joint solve. The orientation codes
    and the neighbour lists are per node of the block. A CTCF loop to a node of another block is
    left out of the lists.
    """
    s = state.s
    clusters = state.clusters
    a = len(active_region)
    anchor_genomic = [
        (clusters[ci].start, clusters[ci].end, clusters[ci].genomic_pos) for ci in active_region
    ]
    node_bead: I32Array = np.zeros(0, dtype=np.int32)
    node_pos: F32Array = np.zeros((0, 3), dtype=np.float32)
    orientations: I8Array | None = None
    neighbors: dict[int, list[int]] | None = None
    weights: dict[int, list[float]] | None = None
    if nodes is not None:
        sel = np.flatnonzero(nodes.nodes.block == block)
        local = {int(g): k for k, g in enumerate(sel)}
        node_bead = nodes.nodes.bead[sel].astype(np.int32)
        node_pos = nodes.pos[sel].astype(np.float32)
        if s.use_ctcf_motif and chr_:
            orientations = nodes.orient[sel].astype(np.int8)
            neighbors = {k: [] for k in range(len(sel))}
            weights = {k: [] for k in range(len(sel))}
            for k, g in enumerate(sel):
                for other, w in zip(nodes.nbrs[int(g)], nodes.nbr_w[int(g)], strict=True):
                    if other in local:
                        neighbors[k].append(local[other])
                        weights[k].append(w)

    subanchor_heat_raw: F32Array | None = None
    if s.use_subanchor_heatmap and state.singletons:
        edges = grid_edges(anchor_genomic, int(s.target_bp_per_subanchor))
        subanchor_heat_raw = uniform_heatmap(state.singletons, chr_, edges)
    wants_heat = (
        subanchor_heat_raw is not None
        and float(subanchor_heat_raw.mean(dtype=np.float64)) >= 1e-6
        and not _heat_signal_negligible(
            s, subanchor_heat_raw, int(subanchor_heat_raw.shape[0]), ib_label
        )
    )

    track_compartments: list[CompartmentInterval] | None = None
    if chr_ and anchor_genomic and s.use_compartments:
        track_compartments = slice_intervals(
            state.compartments.get(chr_, []),
            min(g[0] for g in anchor_genomic),
            max(g[1] for g in anchor_genomic),
        )

    seed = Seeded(
        settings=s,
        seed=(ib_idx * 2_654_435_761 + 40_503 + seed_offset) & 0x7FFFFFFF,
        # The centroid of the block, where a chain without a node is placed.
        anchor_seed_pos=np.tile(clusters[ib_idx].pos.astype(np.float32), (a, 1)),
        exp_dist=np.zeros((a, a), dtype=np.float64),
        orientations=orientations,
        anchor_neighbors=neighbors,
        anchor_neighbor_weights=weights,
        subanchor_heat_raw=subanchor_heat_raw,
        anchor_genomic=anchor_genomic,
        step_size_arcs=_ARCS_NOISE,
        track_compartments=track_compartments,
        chrom=chr_,
        node_bead=node_bead,
        node_pos=node_pos,
    )
    return seed, wants_heat


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
    if (state.s.use_anchor_heatmap or state.s.use_subanchor_heatmap) and (
        state.singletons or state.anchor_contacts
    ):
        anchor_heat, subanchor_heat_raw = cb.build_contact_heatmaps(
            state, active_region, chr_, with_subanchor=bool(state.s.use_subanchor_heatmap)
        )

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
        and not _heat_signal_negligible(state.s, subanchor_heat_raw, a, ib_label)
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
