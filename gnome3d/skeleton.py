"""
Skeleton: the coupled preamble that produces per-IB ``Seeded`` states.

This is the sequential, cross-IB-coupled half of reconstruction — hierarchy,
the inter-chromosomal + segment heatmaps, and IB-centroid positioning — plus the
per-IB *input* gathering (expected-distance matrix, contact heatmaps, CTCF
orientation/motif graph, anchor genomic spans).  Its output is a flat list of
``Seeded`` states: everything an isolated IB needs to reconstruct itself, copied
out as plain arrays so nothing downstream references the cluster graph.

The expensive, *isolated* half — arcs -> densify -> heat -> smooth — lives in
`pipeline.stages` and is driven by an executor over these seeds.

The coarse engine itself is the free functions in `gnome3d.pipeline.coarse` over a
`CoarseState` (hierarchy + inter-chr/segment heatmap MC + IB positioning); this
module drives them and reads the positioned cluster graph to emit each IB's
`Seeded` inputs.  The per-IB *gather* (`gather_ib_seeds` / `seed_for_ib`) is
reused by the coarse pipeline's fan-out `expand` (`pipeline.coarse.stages`), so
the legacy and unified-DAG paths build identical seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.hierarchy import Level, set_level
from gnome3d.pipeline import Orientation, Seeded
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse import CoarseState
from gnome3d.settings import Settings
from gnome3d.types import BedRegion, F64Array, I8Array

LOG = log.get("skeleton")

# Arc-level MC uses a hardcoded noise size (Reference LooperSolver.cpp:2136 passes
# noise_size_small=0.005, ignoring noiseCoefficientLevelAnchor).
_ARCS_NOISE: float = 0.005

_ORN_CODE: dict[str, int] = {"L": Orientation.LEFT, "R": Orientation.RIGHT}


def _heat_signal_negligible(settings: Settings, subanchor_heat_raw: F64Array, n: int) -> bool:
    """True when an IB's subanchor heat is too sparse to affect the structure, so
    the (expensive) HEAT_DIST stage can be dropped from its chain.

    The active-pair fraction ``n_active / n_pairs`` is a provable upper bound on
    the mean target-distance reduction the heat term can produce (each active
    pair's target is ``avg_dist * (1 - s_val)`` with ``s_val`` capped at 1), and
    it's known from the raw heatmap alone — no dry-smooth trials needed.  Below
    ``subanchor_heat_min_reduction`` the IB smooths without heat.  Threshold 0.0
    (default) disables this — full parity; opt-in divergence for sparse data.

    (Lives here, not in coarse: it's the skeleton's chain-shape decision.)
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
            "heat-dist skipped: %d/%d pairs active (%.3g < %.3g min reduction)",
            n_active,
            n_pairs,
            frac,
            thresh,
        )
        return True
    return False


@dataclass(frozen=True)
class IBSeed:
    """One IB's seed: its display id, chromosome, the ``Seeded`` inputs, and
    whether its chain should include a HEAT_DIST stage (decided here, from the
    raw heatmap, via the same sparse-signal early-out the smooth path uses)."""

    ib_id: str
    chr_: str
    seed: Seeded
    wants_heat: bool


def build_seeds(
    settings: Settings,
    data: ContactData,
    chrs: list[str],
    region: BedRegion | None = None,
    seed_offset: int = 0,
) -> list[IBSeed]:
    """Build the hierarchy + coarse positioning, then emit one ``Seeded`` per IB.

    Runs the coupled preamble (`coarse.build_state` +
    `coarse.reconstruct_heatmap`), then walks each chromosome's positioned
    IBs (positioning each segment's IB centroids and gathering its seeds).

    `seed_offset` shifts every RNG seed (coarse + per-IB) so that distinct
    ensemble members produce distinct structures from the same inputs — pass a
    well-separated offset per member.  0 is the canonical single-structure run.
    """
    # Deterministic coarse engine: seed global RNG before any coarse MC runs, so
    # anchor seed positions are reproducible run-to-run.  (The unified-DAG path
    # seeds the same way, but from the root stage — see `coarse.seed_global_rng`.)
    cb.seed_global_rng(seed_offset)

    state = cb.build_state(settings, data, chrs, region)
    cb.reconstruct_heatmap(state)

    seeds: list[IBSeed] = []
    for chr_ in state.chrs:
        seeds.extend(_seeds_for_chr(state, chr_, seed_offset))
    return seeds


def _seeds_for_chr(state: CoarseState, chr_: str, seed_offset: int = 0) -> list[IBSeed]:
    """Position this chromosome's IB centroids, then gather one ``Seeded`` per IB."""
    seg_level = set_level(Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs)
    segs = seg_level.get(chr_, [])
    if not segs:
        return []

    LOG.info("anchor level: %s", chr_)
    cb.position_interaction_blocks(state, segs)
    return gather_ib_seeds(state, chr_, seg_level, seed_offset)


def gather_ib_seeds(
    state: CoarseState, chr_: str, seg_level: dict[str, list[int]], seed_offset: int = 0
) -> list[IBSeed]:
    """Gather one ``Seeded`` per IB of `chr_` from the *already positioned*
    cluster graph.  Consumes no RNG — pure read-out — so it can run after all
    coarse positioning (the unified-DAG fan-out) or interleaved per chr (the
    legacy path) with identical results."""
    clusters = state.clusters
    segs = seg_level.get(chr_, [])
    ibs: list[int] = []
    for seg_idx in segs:
        ibs.extend(clusters[seg_idx].children)
    n_ibs = len(ibs)

    out: list[IBSeed] = []
    for ib_i, ib_idx in enumerate(ibs):
        ib = clusters[ib_idx]
        active_region = list(ib.children)
        ib_id = f"{chr_} IB {ib_i + 1}/{n_ibs}"
        if len(active_region) <= 1:
            LOG.info("%s  (%d anchors - skip)", ib_id, len(active_region))
            continue
        # Each anchor seeds at the IB centroid (matches the arc stage's
        # initial_pos before its per-anchor noise).
        for a_idx in active_region:
            clusters[a_idx].pos = ib.pos.copy()

        seed, wants_heat = seed_for_ib(state, chr_, ib_idx, active_region, seed_offset)
        out.append(IBSeed(ib_id=ib_id, chr_=chr_, seed=seed, wants_heat=wants_heat))
    return out


def seed_for_ib(
    state: CoarseState, chr_: str, ib_idx: int, active_region: list[int], seed_offset: int = 0
) -> tuple[Seeded, bool]:
    """Gather one IB's ``Seeded`` inputs from the positioned cluster graph,
    copying everything out as plain arrays (no cluster references retained)."""
    s = cb.settings_for_ib(state, active_region)
    clusters = state.clusters
    a = len(active_region)

    # Contact heatmaps (anchor-level scales exp_dist; subanchor feeds HEAT_DIST).
    anchor_heat: F64Array | None = None
    subanchor_heat_raw: F64Array | None = None
    if (state.s.use_anchor_heatmap or state.s.use_subanchor_heatmap) and state.singletons:
        anchor_heat, subanchor_heat_raw = cb.build_contact_heatmaps(state, active_region, chr_)

    exp_dist = cb.calc_anchor_expected_distances(state, active_region, chr_, anchor_heat)

    # Anchor seed positions (all at the IB centroid right now) + genomic spans.
    anchor_seed_pos = np.array([clusters[ci].pos for ci in active_region], dtype=np.float32)
    anchor_genomic = [
        (clusters[ci].start, clusters[ci].end, clusters[ci].genomic_pos) for ci in active_region
    ]

    # CTCF orientation (int8) + motif neighbour graph, anchor-indexed.  Built
    # only when the motif term is on (matches the old smooth-problem build).
    orientations: I8Array | None = None
    anchor_neighbors: dict[int, list[int]] | None = None
    anchor_neighbor_weights: dict[int, list[float]] | None = None
    if state.s.use_ctcf_motif and chr_:
        orientations = np.array(
            [_ORN_CODE.get(clusters[ci].orientation or "N", Orientation.NONE) for ci in active_region],
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
                other_ci = arc.end if arc.start == ci else arc.start
                if other_ci in cluster_to_k:
                    anchor_neighbors[k].append(cluster_to_k[other_ci])
                    anchor_neighbor_weights[k].append(math.sqrt(max(arc.score, 0)))

    # HEAT_DIST inclusion: same sparse-signal early-out as the engine — empty
    # heatmap or active-fraction below threshold => no heat stage.
    wants_heat = (
        subanchor_heat_raw is not None
        and bool(state.s.use_subanchor_heatmap)
        and float(subanchor_heat_raw.mean()) >= 1e-6
        and not _heat_signal_negligible(state.s, subanchor_heat_raw, a)
    )

    seed = Seeded(
        settings=s,
        # Per-IB RNG seed: deterministic + process-stable (avoid hash(), which is
        # salted per-process).  ib_idx is a globally-unique cluster index; spread
        # it so adjacent IBs get well-separated seeds.  seed_offset varies it per
        # ensemble member.
        seed=(ib_idx * 2_654_435_761 + 40_503 + seed_offset) & 0x7FFFFFFF,
        anchor_seed_pos=anchor_seed_pos,
        exp_dist=exp_dist,
        orientations=orientations,
        anchor_neighbors=anchor_neighbors,
        anchor_neighbor_weights=anchor_neighbor_weights,
        subanchor_heat_raw=subanchor_heat_raw,
        anchor_genomic=anchor_genomic,
        step_size_arcs=_ARCS_NOISE,
    )
    return seed, wants_heat
