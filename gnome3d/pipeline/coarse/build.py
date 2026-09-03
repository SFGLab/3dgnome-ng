"""
Coarse substrate: the cluster-graph state + its pure build helpers.

`CoarseState` is the immutable carrier for the coarse half of reconstruction -
the cluster hierarchy plus the contact data needed to position it.  The free
functions here are the *build* helpers that read that state to produce heatmaps,
expected-distance matrices, bin layouts and child interpolation - everything the
coarse positioning stages consume but that holds no MC orchestration of its own.

Splitting these out of the old `CoarseModel` god-object lets the coarse
positioning levels become small pipeline stages (in `pipeline.coarse.stages`)
that call these helpers, instead of methods on one big class.  Two of them touch
RNG and so must be called in a fixed order to keep the layout byte-exact:

* `interpolate_children_linear` - its single-parent branch adds
  `random_vector_np(100.0)` noise per child (and recurses), so it consumes the
  global stream; the coarse stages call it in the same order the old engine did.

Everything else here is pure (no RNG): bin layout, heatmap binning/normalization,
expected-distance and contact-heatmap construction, and the small-IB settings
boost.  `clusters` is shared and its `.pos` mutates in place across the coarse
stages - natural for a graph algorithm; the immutability is of the *state's*
field set, not of the positioned graph.
"""

from __future__ import annotations

from dataclasses import field

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.hierarchy import Cluster, Level, build_cluster_tree, set_level
from gnome3d.io import create_singleton_heatmap
from gnome3d.pipeline.coarse.heatmap import (
    create_distance_heatmap,
    get_diagonal_size,
    normalize_heatmap,
    normalize_heatmap_diagonal_total,
    normalize_heatmap_inter,
)
from gnome3d.settings import Settings
from gnome3d.tracks import bin_compartments, bin_signal, normalize_accessibility
from gnome3d.types import *
from gnome3d.util import random_vector_np, seed_rng

LOG = log.get("coarse")

# Fixed base seed for the coarse engine.  The coarse spine consumes a single
# global RNG stream in dependency order, so seeding once makes the whole layout
# deterministic; `seed_offset` shifts it per ensemble member.
COARSE_SEED: int = 0xC0FFEE


def seed_global_rng(seed_offset: int = 0) -> int:
    """Seed the global RNG (Python `random` for positioning noise, numba for the
    heatmap/IB kernels) for the coarse spine, and return the resolved seed.

    The coarse spine seeds *once* here and then flows the stream in dependency
    order - it does NOT re-seed per stage.  That's the opposite of the per-IB
    stages (which re-seed from `Seeded.seed` per node to be order-independent for
    batching): the coarse levels are a coupled linear sequence that is never
    reordered, so one seed is both deterministic and matches the reference
    engine's single-stream order (keeping the layout byte-exact).
    """
    from gnome3d.mc import numba as mc_numba

    coarse_seed = (COARSE_SEED + seed_offset) & 0x7FFFFFFF
    seed_rng(coarse_seed)
    mc_numba.seed_numba(coarse_seed)
    return coarse_seed


@dataclass(frozen=True)
class CoarseState:
    """The cluster hierarchy + the contact data needed to position it.

    Immutable field set; the cluster objects it points at mutate in place (their
    `.pos` is written by the coarse positioning stages - natural for a graph
    algorithm).  Built once by `build_state`; threaded through the coarse stages.
    """

    s: Settings
    clusters: list[Cluster]
    chr_root: ChrRootMap
    chr_first_cluster: ChrFirstClusterMap
    chrs: list[str]
    anchors: AnchorMap
    arcs: ArcMap
    singletons: list[SingletonContact]
    long_arcs: RawArcMap
    selected_region: BedRegion | None = None
    # Epigenomic tracks for the opt-in compartment and accessibility terms.
    # Empty when no track is configured, which leaves those terms inert.
    compartments: CompartmentMap = field(default_factory=empty_compartment_map)
    accessibility: SignalMap = field(default_factory=empty_signal_map)


def build_state(
    settings: Settings,
    data: ContactData,
    chrs_list: list[str],
    region: BedRegion | None = None,
) -> CoarseState:
    """Build the cluster hierarchy from pre-loaded contact data.

    Mirrors the old `CoarseModel.load` / Reference LooperSolver::setContactData().
    """
    with log.step(LOG, "build cluster hierarchy"):
        clusters, chr_root, chr_first_cluster = build_cluster_tree(
            data.anchors,
            data.arcs,
            data.breakpoints,
            chrs_list,
        )
        LOG.info("total clusters: %d", len(clusters))

    return CoarseState(
        s=settings,
        clusters=clusters,
        chr_root=chr_root,
        chr_first_cluster=chr_first_cluster,
        chrs=chrs_list,
        anchors=data.anchors,
        arcs=data.arcs,
        singletons=data.singletons,
        long_arcs=data.long_arcs,
        selected_region=region,
        compartments=data.compartments,
        accessibility=data.accessibility,
    )


# --- epigenomic track lookup ------------------------------------------------


def compartment_for_clusters(
    state: CoarseState, indices: list[ClusterIndex], chr_: str
) -> I8Array | None:
    """
    Compartment call per cluster, or None when no track covers this chromosome.

    Clusters carry inclusive genomic ranges, which is what `bin_compartments`
    expects.
    """
    ivs = state.compartments.get(chr_, [])
    if not ivs or not indices:
        return None
    clusters = state.clusters
    starts = [clusters[ci].start for ci in indices]
    ends = [clusters[ci].end for ci in indices]
    cls, _score = bin_compartments(ivs, starts, ends)
    return cls


def coarse_track_arrays(
    state: CoarseState, active_region: list[ClusterIndex], chr_of: list[str]
) -> tuple[I8Array | None, F32Array | None, I32Array | None, F64Array | None]:
    """Per-cluster track arrays for a multi-chromosome active region.

    `active_region` and `chr_of` are parallel, so each cluster is binned against
    its own chromosome's track.  Returns
    (compartment, accessibility, chromosome id, chromosome weight), each None
    when the term that reads it is off or its data is missing.

    The chromosome weight drives the nucleolar pull.  It is the mean chromosome
    span over this chromosome's span, so a smaller chromosome gets a larger
    weight and sits nearer the centre, which is the bias MultiMM's central force
    encodes.
    """
    s = state.s
    if not active_region:
        return None, None, None, None

    want_comp = s.use_compartments or s.use_lamina
    want_acc = s.use_bridging
    want_chrom = s.use_central_force or s.use_chromosomal_blocks

    n = len(active_region)
    clusters = state.clusters
    by_chr: dict[str, list[int]] = {}
    for i, c in enumerate(chr_of):
        by_chr.setdefault(c, []).append(i)

    comp: I8Array | None = np.zeros(n, dtype=np.int8) if want_comp else None
    acc: F32Array | None = np.zeros(n, dtype=np.float32) if want_acc else None
    got_comp = False
    got_acc = False

    for chr_, rows in by_chr.items():
        idx = [active_region[i] for i in rows]
        starts = [clusters[ci].start for ci in idx]
        ends = [clusters[ci].end for ci in idx]
        if comp is not None:
            ivs = state.compartments.get(chr_, [])
            if ivs:
                cls, _score = bin_compartments(ivs, starts, ends)
                comp[rows] = cls
                got_comp = True
        if acc is not None:
            sig = state.accessibility.get(chr_, [])
            if sig:
                acc[rows] = bin_signal(sig, starts, ends)
                got_acc = True

    chrom_id: I32Array | None = None
    chrom_w: F64Array | None = None
    if want_chrom:
        order = {c: k for k, c in enumerate(sorted(by_chr))}
        chrom_id = np.array([order[c] for c in chr_of], dtype=np.int32)
        spans = {
            c: max(
                max(clusters[active_region[i]].end for i in rows)
                - min(clusters[active_region[i]].start for i in rows),
                1,
            )
            for c, rows in by_chr.items()
        }
        mean_span = float(sum(spans.values())) / len(spans)
        chrom_w = np.array([mean_span / spans[c] for c in chr_of], dtype=np.float64)

    return (
        comp if got_comp else None,
        acc if got_acc else None,
        chrom_id,
        chrom_w,
    )


def accessibility_for_clusters(
    state: CoarseState, indices: list[ClusterIndex], chr_: str
) -> F32Array | None:
    """
    Normalised accessibility per cluster, or None when no track covers this
    chromosome.

    The normalisation is done over the clusters handed in rather than genome
    wide, so `a` spans [0, 1] within whatever region is being scored.  That keeps
    the bridging strength comparable across regions of very different signal
    depth.
    """
    ivs = state.accessibility.get(chr_, [])
    if not ivs or not indices:
        return None
    clusters = state.clusters
    starts = [clusters[ci].start for ci in indices]
    ends = [clusters[ci].end for ci in indices]
    return normalize_accessibility(bin_signal(ivs, starts, ends))


# --- RNG-ordered shared subroutine ------------------------------------------


def interpolate_children_linear(state: CoarseState, parent_indices: list[int]) -> None:
    """
    Set child cluster positions by linear interpolation between parents.
    Used for IBs between segments, and anchors within IBs.
    Simplified version of Reference interpolateChildrenPositionSpline().

    RNG: the single-parent branch adds `random_vector_np(100.0)` per child and
    recurses, so this consumes the global stream - call order matters for the
    byte-exact layout.
    """
    clusters = state.clusters
    n = len(parent_indices)
    if n == 0:
        return

    if n == 1:
        # All children at parent position with small noise
        par = clusters[parent_indices[0]]
        for child_idx in par.children:
            clusters[child_idx].pos = par.pos + random_vector_np(100.0)
            # Recurse into grandchildren
            if clusters[child_idx].children:
                interpolate_children_linear(state, [child_idx])
        return

    for i, par_idx in enumerate(parent_indices):
        par = clusters[par_idx]
        n_children = len(par.children)
        if n_children == 0:
            continue

        # Interpolation endpoints in 3D
        p_start = par.pos
        p_end = clusters[parent_indices[min(i + 1, n - 1)]].pos

        for j, child_idx in enumerate(par.children):
            t = (j + 0.5) / n_children
            clusters[child_idx].pos = ((1 - t) * p_start + t * p_end).astype(np.float32)
            # Recurse
            if clusters[child_idx].children:
                interpolate_children_linear(state, [child_idx])


# --- pure (no-RNG) build helpers --------------------------------------------


def compute_segment_bins(
    state: CoarseState, current_level: ChrLevel
) -> tuple[dict[str, list[int]], dict[str, int], int, list[float]]:
    """
    Compute heatmap bin boundaries for segment-level clusters.
    Mirrors bin calculation in createSingletonHeatmap().
    Returns (bins, start_ind, total_size, bin_lengths_mb).

    bin_lengths_mb is a flat list aligned to global bin indices giving
    the genomic span of each bin in Mb.  The first and last bins of each
    chromosome use the actual cluster start/end (not the 0/1e9 sentinels)
    so their lengths are not artificially inflated - mirrors the Reference min/max
    position update done after reading contacts.
    """
    clusters = state.clusters
    bins: dict[str, list[int]] = {}
    start_ind: dict[str, int] = {}
    curr_idx = 0
    bin_lengths_mb: list[float] = []

    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        breaks = [0]
        for i in range(len(segs) - 1):
            pos = (clusters[segs[i]].end + clusters[segs[i + 1]].start) // 2
            breaks.append(pos)
        breaks.append(int(1e9))
        bins[chr_] = breaks
        start_ind[chr_] = curr_idx

        n = len(segs)
        for i in range(n):
            if n == 1:
                bp = clusters[segs[0]].end - clusters[segs[0]].start
            elif i == 0:
                bp = breaks[1] - clusters[segs[0]].start
            elif i == n - 1:
                bp = clusters[segs[-1]].end - breaks[-2]
            else:
                bp = breaks[i + 1] - breaks[i]
            bin_lengths_mb.append(max(bp, 1) / 1e6)

        curr_idx += len(breaks) - 1

    return bins, start_ind, curr_idx, bin_lengths_mb


def add_long_pet_to_segment_heatmap(
    state: CoarseState,
    h: list[list[float]],
    bins: dict[str, list[int]],
    start_ind: dict[str, int],
    total_size: int,
) -> None:
    """
    Bin long-range PET arcs (gap > max_pet_length) into the segment heatmap.
    Each arc contributes long_pet_scale * arc.score ** long_pet_power to the
    (st, end) bin pair. Mirrors Reference LooperSolver.cpp:1069-1104.

    Note: the Reference reference does h[st][end] += val twice (to the same cell),
    not symmetrically.  The downstream normalize_heatmap symmetrize step
    averages [st][end] and [end][st] so the net effect is val on each side
    after normalization. We preserve the same pattern for parity.
    """
    if not state.long_arcs:
        return
    import bisect

    scale = state.s.long_pet_scale
    power = state.s.long_pet_power
    n_added = 0
    for chr_ in state.chrs:
        chrlong_arcs = state.long_arcs.get(chr_, [])
        chr_bins = bins.get(chr_)
        if not chrlong_arcs or chr_bins is None:
            continue
        si_base = start_ind.get(chr_, 0)
        for arc in chrlong_arcs:
            st = si_base + bisect.bisect_right(chr_bins, arc.start) - 1
            end = si_base + bisect.bisect_right(chr_bins, arc.end) - 1
            if st == end:
                continue
            if st < 0 or end < 0 or st >= total_size or end >= total_size:
                continue
            val = scale * (arc.score**power)
            h[st][end] += val
            h[st][end] += val
            n_added += 1
    if n_added > 0:
        LOG.info("long-PET folded into segment heatmap: %d arcs", n_added)


def arc_expected_matrix(s: Settings, mids: list[int], arcs: list[tuple[int, int, int]]) -> F64Array:
    """The arc target matrix for one active region. -1 marks an arcless pair, 0 the diagonal,
    and an arc pair carries `Settings.arc_expected_distance` of its PET count and span.

    Parameters
    ----------
    mids
        Genomic midpoint in bp per anchor, in active region order.
    arcs
        (i, j, score) per arc in active region indices.
    """
    n = len(mids)
    mat: F64Array = np.full((n, n), -1.0, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)
    for i, j, score in arcs:
        exp_d = s.arc_expected_distance(score, mids[j] - mids[i])
        mat[i, j] = exp_d
        mat[j, i] = exp_d
    return mat


def add_chain_bonds(mat: F64Array, mids: list[int], s: Settings) -> F64Array:
    """Give every consecutive anchor pair with no arc a spring at the chain law distance of its
    gap. Returns a new matrix; the input is left alone. With `use_arcs_chain_bonds` off the
    input is returned as is.

    The arcs MC otherwise has no term between genomic neighbours, so a group of anchors joined
    only among themselves by arcs floats out to the confinement leash. The bond ties it to its
    neighbours the way the smooth stage ties consecutive beads. A pair that already has an arc
    keeps the arc. The bond carries the arcs spring constants and its target is
    `arcs_chain_bond_scale` times the chain law.
    """
    if not s.use_arcs_chain_bonds:
        return mat
    out = np.array(mat, dtype=np.float64, copy=True)
    order = sorted(range(len(mids)), key=lambda i: mids[i])
    for a, b in zip(order[:-1], order[1:], strict=True):
        if out[a, b] > 0.0:
            continue
        d = float(s.arcs_chain_bond_scale) * float(
            s.genomic_length_to_distance(abs(int(mids[b]) - int(mids[a])))
        )
        out[a, b] = d
        out[b, a] = d
    return out


def calc_anchor_expected_distances(
    state: CoarseState,
    active_region: list[int],
    chr_: str,
    anchor_heatmap: F64Array | None = None,
) -> F64Array:
    """
    Build expected distance matrix for anchor-level active region.
    Mirrors Reference calcAnchorExpectedDistancesHeatmap().

    If anchor_heatmap (n x n) is provided and use_anchor_heatmap is True,
    scales down expected distances for high-contact anchor pairs, mirroring
    Reference calcAnchorExpectedDistancesHeatmap() post-processing.

    Returns mat where:
      mat[i,j] = -1  -> repulsion (no arc)
      mat[i,j] =  0  -> diagonal (self)
      mat[i,j] > 0   -> expected distance from freqToDistance(score)
    """
    s = state.s
    clusters = state.clusters
    n = len(active_region)
    cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}
    chr_arcs = state.arcs.get(chr_, [])

    arcs: list[tuple[int, int, int]] = []
    for ai, ci in enumerate(active_region):
        for arc_local in clusters[ci].arcs:
            if arc_local >= len(chr_arcs):
                continue
            arc = chr_arcs[arc_local]
            other = arc.end if arc.start == ci else arc.start

            if other < ci or other not in cluster_to_active:
                continue

            arcs.append((ai, cluster_to_active[other], int(arc.score)))
    mids = [int(clusters[ci].genomic_pos) for ci in active_region]
    mat = arc_expected_matrix(s, mids, arcs)

    # Apply anchor heatmap: scale down expected distances for high-contact pairs.
    # Mirrors Reference post-processing in calcAnchorExpectedDistancesHeatmap().
    if anchor_heatmap is not None and s.use_anchor_heatmap:
        max_val = float(anchor_heatmap.max())
        influence = float(s.anchor_heatmap_influence)
        if max_val > 1e-6:
            for i in range(n):
                for j in range(i + 1, n):
                    if mat[i, j] <= 0.0:
                        continue
                    s_val = (anchor_heatmap[i, j] / max_val) * influence
                    if s_val > 1.0:
                        s_val = 1.0
                    mat[i, j] *= 1.0 - s_val
                    mat[j, i] = mat[i, j]

    # After the heatmap scaling, so Hi-C contact between neighbours does not shrink the bond.
    return add_chain_bonds(mat, mids, s)


def subanchor_counts_per_arc(state: CoarseState, active_region: list[int]) -> list[int]:
    """
    Number of subanchors to insert between each consecutive anchor pair.

    Default mode (`use_dynamic_loop_density = False`): uniform
    `s.loop_density` for every arc - matches the historical behavior
    and the reference.

    Dynamic mode (`use_dynamic_loop_density = True`): for each arc with
    in-between span `span_bp`, pick the subanchor count so that the chain
    segments connecting beads are roughly `target_bp_per_subanchor` long.
    With `n` subanchors there are `n + 1` segments in the gap, so
    `n = round(span_bp / target) - 1` (clamped to `[min, max]`).

    Examples (target = 5kb): a 3 kb arc rounds to 1 segment → 0 subanchors;
    a 5 kb arc → 1 segment → 0 subanchors (consecutive anchors are linked
    directly); a 10 kb arc → 2 segments → 1 subanchor; a 50 kb arc → 10
    segments → 9 subanchors.  Set `min_subanchors_per_arc = 1` to force
    at least one subanchor between every pair regardless of span.

    Both densification and `build_contact_heatmaps` consume this so the
    densified bead chain and the subanchor heatmap binning stay in sync.
    """
    s = state.s
    clusters = state.clusters
    n_arcs = max(len(active_region) - 1, 0)
    if not s.use_dynamic_loop_density:
        return [s.loop_density] * n_arcs

    target = max(s.target_bp_per_subanchor, 1)
    mn = max(s.min_subanchors_per_arc, 0)
    mx = max(s.max_subanchors_per_arc, mn)
    counts: list[int] = []
    for i in range(n_arcs):
        ca = clusters[active_region[i]]
        cb = clusters[active_region[i + 1]]
        span = abs(cb.start - ca.end)
        # n_segments = round(span / target); subanchors = segments - 1.
        n = round(span / target) - 1
        counts.append(max(mn, min(mx, n)))
    return counts


def build_contact_heatmaps(
    state: CoarseState,
    active_region: list[int],
    chr_: str,
) -> tuple[F64Array, F32Array]:
    """
    Build anchor-level and subanchor-level singleton contact heatmaps.
    Mirrors Reference createSingletonSubanchorHeatmap().

    Returns (anchor_heatmap, subanchor_heatmap_raw) where:
      anchor_heatmap:      (n_anchors, n_anchors) float64 - normalized contact
                           density between anchor pairs; used for expected-distance
                           scaling in arc MC.
      subanchor_heatmap_raw: (N, N) float32 where N = n_anchors + (n_anchors-1)*ld
                           - normalized contact density at densified-bead resolution;
                           used for heat energy in smooth MC.  Built in float64 (the
                           binning/normalization stay byte-exact) then stored float32:
                           it's a dense (N,N) matrix held per-IB, ~8.6 GB at N=32768
                           in f64, and the reference Heatmap is `float` anyway.
    """
    clusters = state.clusters
    n_anchors = len(active_region)
    counts = subanchor_counts_per_arc(state, active_region)  # length n_anchors-1
    # Total bins = anchors + subanchors. Each arc i contributes counts[i] bins.
    N = n_anchors + sum(counts)

    # anchor_offsets[k] is the flat-bin index of anchor k.
    # Subanchor j of arc i lives at bin offsets[i] + 1 + j  (j = 0..counts[i]-1).
    anchor_offsets: list[int] = [0]
    for c in counts:
        anchor_offsets.append(anchor_offsets[-1] + 1 + c)

    anchor_lens: list[int] = []
    gap_lens: list[int] = []

    region_start = clusters[active_region[0]].start
    region_end = clusters[active_region[-1]].end

    # Build break boundaries.  breaks[k] = left edge of bin k.
    # For each arc i: anchor i ends at ca.end, then counts[i] subanchor bins
    # tile [ca.end, cb.start], then anchor (i+1) begins at cb.start.  If
    # counts[i] == 0 there's no subanchor bin for that arc - we still need
    # a single break separating the two anchor bins, set to the midpoint
    # of the gap so each anchor absorbs half of it (negligible when count
    # collapses to 0 only for short arcs anyway).
    breaks: list[int] = [region_start]
    anchor_lens.append(clusters[active_region[0]].end - clusters[active_region[0]].start)

    for i in range(n_anchors - 1):
        ca_end = clusters[active_region[i]].end
        cb_start = clusters[active_region[i + 1]].start
        gap = max(cb_start - ca_end, 0)
        anchor_len = clusters[active_region[i + 1]].end - clusters[active_region[i + 1]].start
        gap_lens.append(gap)
        anchor_lens.append(anchor_len)
        c = counts[i]
        if c >= 1:
            # ca.end closes anchor i's bin; c-1 interior breaks tile the gap;
            # cb.start opens anchor (i+1)'s bin.
            breaks.append(ca_end)
            for j in range(1, c):
                breaks.append(ca_end + int(gap * j / c))
            breaks.append(cb_start)
        else:
            # No subanchor bin for this arc; split the gap evenly between
            # the two flanking anchors with a midpoint break.
            breaks.append((ca_end + cb_start) // 2)

    breaks.append(region_end)

    # Bin singleton contacts into the subanchor heatmap.  Kept as a `bisect` loop
    # (NOT vectorized): `breaks` can be non-monotonic when consecutive anchors
    # overlap (gap=0 -> cb_start < ca_end), and `np.searchsorted` (which assumes a
    # sorted array) diverges from `bisect` exactly there.  The loop's bisect
    # behavior on those breaks mirrors the reference, so it must be preserved; it's
    # also cheap (O(#singletons in region), tiny next to the N^2 matrix work below).
    #
    # Note: Python filters by chromosome (both ends on chr_).  Reference's
    # createSingletonSubanchorHeatmap does NOT - see
    # [[project-singleton-chr-filter-divergence]] (intentional divergence).
    import bisect

    h_sub: F64Array = np.zeros((N, N), dtype=np.float64)
    for c1, p1, c2, p2, sc in state.singletons:
        if c1 != chr_ or c2 != chr_:
            continue
        if p1 < region_start or p1 > region_end or p2 < region_start or p2 > region_end:
            continue
        si = bisect.bisect_right(breaks, p1) - 1
        ei = bisect.bisect_right(breaks, p2) - 1
        if si < 0 or ei < 0 or si >= N or ei >= N or si == ei:
            continue
        h_sub[si, ei] += sc
        h_sub[ei, si] += sc

    # Anchor heatmap from raw subanchor values (BEFORE normalization), normalized
    # by anchor area in Mbp^2.  Mirrors Reference lines 1267-1273; vectorized over
    # the anchor bins (diagonal stays 0, off-diagonal symmetric).
    anchor_off = np.asarray(anchor_offsets[:n_anchors], dtype=np.intp)
    al = np.maximum(np.asarray(anchor_lens, dtype=np.float64), 1.0)  # (n_anchors,)
    h_anchor: F64Array = h_sub[np.ix_(anchor_off, anchor_off)] / (np.outer(al, al) / 1e6)
    np.fill_diagonal(h_anchor, 0.0)

    # Which arc each subanchor bin belongs to (-1 at anchor bins).
    bin_arc_idx: list[int] = [-1] * N
    for i, c in enumerate(counts):
        base = anchor_offsets[i] + 1
        for j in range(c):
            bin_arc_idx[base + j] = i

    # Normalize subanchor heatmap: divide by avg count, then by bin areas (kb^2).
    # Mirrors Reference lines 1294-1320; vectorized.
    avg_count = float(h_sub.mean())
    if avg_count > 1e-6:
        h_sub /= avg_count

        # Bin sizes in kb: anchor bins use the anchor's own length; subanchor bins
        # use gap_len / count for their arc.
        bin_sizes: F64Array = np.empty(N, dtype=np.float64)
        bai = np.asarray(bin_arc_idx, dtype=np.intp)
        gl = np.asarray(gap_lens, dtype=np.float64)
        cnt = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
        per_arc = np.maximum(gl / cnt, 1.0) / 1000.0  # (n_anchors-1,)
        sub_mask = bai >= 0
        bin_sizes[sub_mask] = per_arc[bai[sub_mask]]
        bin_sizes[anchor_off] = al / 1000.0

        denom = np.outer(bin_sizes, bin_sizes)
        off = denom > 0.0
        np.fill_diagonal(off, False)  # diagonal untouched (matches the i<j loop)
        h_sub[off] = h_sub[off] / denom[off]

    return h_anchor, h_sub.astype(np.float32)


# --- RNG-ordered positioning ops (the coarse MC spine) ----------------------
#
# These consume the global RNG stream (Python `random` via `random_vector_np` and
# the numba kernels `mc_heatmap`/`mc_ib`) in a fixed order; the coarse stages call
# them on a linear chain so the layout stays byte-exact.  Each mutates the shared
# cluster graph in place (writing `clusters[i].pos`).


def reconstruct_heatmap(state: CoarseState) -> None:
    """
    Position beads at segment level using singleton heatmap MC.
    Mirrors Reference LooperSolver::reconstructClustersHeatmap().

    Dispatches to one of three branches: random-walk segment level, single-segment
    at origin, or the chr-level + segment-level heatmap MC.  Subsequent levels
    (IB centroid positioning) run afterwards regardless of branch.
    """
    current_level = set_level(
        Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
    )

    if state.s.random_walk:
        random_walk_segment_level(state, current_level)
        return

    total_segs = sum(len(v) for v in current_level.values())
    single_seg = len(state.chrs) == 1 and total_segs <= 1

    if single_seg:
        place_single_segment(state, current_level)
        return

    # Multi-chromosome runs: position chromosome roots first.
    if len(state.chrs) > 1:
        reconstruct_chromosome_level(state)

    LOG.info("segment level")
    reconstruct_segment_level(state, current_level)


def place_single_segment(state: CoarseState, current_level: ChrLevel) -> None:
    """Single-segment single-chr case: place the segment at the origin and
    interpolate its children.  Mirrors the `single_seg` branch of the reference
    reconstructClustersHeatmap()."""
    LOG.info("single segment -> place at origin")
    chr_ = state.chrs[0]
    if current_level[chr_]:
        state.clusters[current_level[chr_][0]].pos = np.zeros(3, dtype=np.float32)
        interpolate_children_linear(state, current_level[chr_])


def reconstruct_chromosome_level(state: CoarseState) -> None:
    """
    Position chromosome root beads via heatmap MC over an n_chr * n_chr
    inter-chromosomal singleton heatmap.  Mirrors Reference
    LooperSolver::reconstructClustersHeatmapSingleLevel(0).

    For each pair of chromosomes (i, j), the heatmap cell h[i][j] is the
    sum of singleton scores between those chromosomes.  Self-self contacts
    are ignored (diagonal stays zero, which `normalize_heatmap_diagonal_total`
    treats correctly - it normalizes the first non-zero diagonal).

    If no inter-chromosomal singletons are available the MC has no signal
    to optimize against, so chr roots are scattered randomly around the
    origin instead.
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    n_chr = len(state.chrs)
    if n_chr <= 1:
        return

    chr_to_idx = {chr_: i for i, chr_ in enumerate(state.chrs)}
    h: list[list[float]] = [[0.0] * n_chr for _ in range(n_chr)]
    n_inter = 0
    for c1, _p1, c2, _p2, sc in state.singletons:
        i = chr_to_idx.get(c1, -1)
        j = chr_to_idx.get(c2, -1)
        if i < 0 or j < 0 or i == j:
            continue
        h[i][j] += sc
        h[j][i] += sc
        n_inter += 1

    LOG.info("chromosome level  (%d chr, %d inter-chr singletons)", n_chr, n_inter)

    if n_inter == 0:
        LOG.info("no inter-chromosomal singletons; scattering chr roots randomly")
        for chr_ in state.chrs:
            root_idx = state.chr_root.get(chr_)
            if root_idx is not None:
                clusters[root_idx].pos = random_vector_np(100.0, in_2d=s.use_2d)
        return

    # Normalize first non-zero diagonal to 1.0; convert freq → expected dist.
    hd = normalize_heatmap_diagonal_total(h, n_chr, 1.0)
    heatmap_dist, avg_dist = create_distance_heatmap(s, hd, n_chr, inter=False)
    heatmap_dist = np.array(heatmap_dist, dtype=np.float64)
    heatmap_dist_diag = get_diagonal_size(hd, n_chr)

    step_size = avg_dist * s.noise_lvl1

    # Initial chr root positions: scatter around the origin, mirroring Reference
    # LooperSolver.cpp:308 which does `initial + random_vector(avg_dist)`.
    initial_pos: F32Array = np.zeros((n_chr, 3), dtype=np.float32)
    for i in range(n_chr):
        initial_pos[i] = random_vector_np(step_size, in_2d=s.use_2d)

    pos: F32Array = initial_pos.copy()
    best_pos: F32Array = pos.copy()
    best_score = -1.0

    for run in range(s.steps_lvl1):
        with log.step(LOG, f"chr-level run {run + 1}/{s.steps_lvl1}", "(%d chr)", n_chr):
            for i in range(n_chr):
                pos[i] = initial_pos[i] + random_vector_np(step_size, in_2d=s.use_2d)

            score = mc_numba.mc_heatmap_numba(pos, heatmap_dist, heatmap_dist_diag, step_size, s)
            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()

            LOG.info("-> score=%.6f  best=%.6f", score, best_score)

    # Write best positions back into the chr root clusters; segment-level
    # MC will pick these up as origin via clusters[seg.parent].pos.
    for i, chr_ in enumerate(state.chrs):
        root_idx = state.chr_root.get(chr_)
        if root_idx is not None:
            clusters[root_idx].pos = best_pos[i].copy()


def random_walk_segment_level(state: CoarseState, current_level: ChrLevel) -> None:
    """
    Position segment-level beads via a chained random walk.
    Each chromosome starts at the origin, takes one big step of size
    genomic_length_to_distance(1Mb) to break symmetry, then takes
    len(segs) small steps of 50 units each (Reference constant).
    Mirrors Reference LooperSolver.cpp:84-97.
    """
    clusters = state.clusters
    in_2d = state.s.use_2d
    size = float(state.s.genomic_length_to_distance(1_000_000))
    with log.step(LOG, "random walk for segment level", "(size=%.2f, 2D=%s)", size, in_2d):
        for chr_ in state.chrs:
            segs = current_level.get(chr_, [])
            if not segs:
                continue
            LOG.info("%s: %d segment(s)", chr_, len(segs))
            rw_pos: F32Array = np.zeros(3, dtype=np.float32)
            rw_pos = rw_pos + random_vector_np(size, in_2d=in_2d)  # first point
            for seg_idx in segs:
                rw_pos = rw_pos + random_vector_np(50.0, in_2d=in_2d)
                clusters[seg_idx].pos = rw_pos.copy()
            # smoothly propagate to IB / anchor levels
            interpolate_children_linear(state, segs)


def reconstruct_segment_level(state: CoarseState, current_level: ChrLevel) -> None:
    """
    Reconstruct segment-level positions using singleton heatmap MC.
    Mirrors Reference reconstructClustersHeatmapSingleLevel(1) (segment level).
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    bins, start_ind, total_size, bin_lengths_mb = compute_segment_bins(state, current_level)

    LOG.info("create segment heatmap")
    h_raw = create_singleton_heatmap(
        state.singletons, bins, start_ind, total_size, bin_lengths_mb=bin_lengths_mb
    )

    # Fold long-range PET arcs into the segment heatmap.
    add_long_pet_to_segment_heatmap(state, h_raw, bins, start_ind, total_size)

    # Normalize heatmap rows to equal expected sum
    h_norm = normalize_heatmap(h_raw, total_size)
    # Normalize diagonal total to 1.0
    h_norm = normalize_heatmap_diagonal_total(h_norm, total_size, 1.0)
    # Scale inter-chr contacts (no-op for single chr)
    if len(state.chrs) > 1:
        h_norm = normalize_heatmap_inter(h_norm, total_size, current_level, s.heatmap_inter_scaling)

    # Convert freq -> distance heatmap
    heatmap_dist, avg_dist = create_distance_heatmap(s, h_norm, total_size, inter=False)
    heatmap_dist = np.array(heatmap_dist, dtype=np.float64)
    heatmap_dist_diag = get_diagonal_size(h_norm, total_size)

    # Place initial positions: parent IB position for all segments in chr
    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        if not segs:
            continue
        par = clusters[segs[0]].parent
        origin = clusters[par].pos.copy() if par >= 0 else np.zeros(3, dtype=np.float32)
        for seg_idx in segs:
            clusters[seg_idx].pos = origin.copy()

    # Concatenate all segment indices into active_region, keeping a parallel
    # chromosome list so the epigenomic tracks can be binned per chromosome.
    active_region: list[int] = []
    chr_of: list[str] = []
    for chr_ in state.chrs:
        segs_here = current_level.get(chr_, [])
        active_region.extend(segs_here)
        chr_of.extend([chr_] * len(segs_here))

    if len(active_region) <= 1:
        return

    step_size = avg_dist * s.noise_lvl2

    seg_comp, seg_acc, seg_chrom_id, seg_chrom_w = coarse_track_arrays(state, active_region, chr_of)

    pos: F32Array = np.array([clusters[i].pos for i in active_region], dtype=np.float32)
    n = len(active_region)
    best_score = -1.0
    best_pos: F32Array = pos.copy()

    for run in range(s.steps_lvl2):
        with log.step(LOG, f"heatmap run {run + 1}/{s.steps_lvl2}", "(%d beads)", n):
            for i in range(n):
                pos[i] = clusters[active_region[i]].pos + random_vector_np(step_size)

            score = mc_numba.mc_heatmap_numba(
                pos,
                heatmap_dist,
                heatmap_dist_diag,
                step_size,
                s,
                seg_comp,
                seg_acc,
                seg_chrom_id,
                seg_chrom_w,
            )
            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()
            LOG.info("-> score=%.6f  best=%.6f", score, best_score)

    for i, idx in enumerate(active_region):
        clusters[idx].pos = best_pos[i].copy()

    # Interpolate IB and anchor positions from segment positions
    for chr_ in state.chrs:
        segs = current_level.get(chr_, [])
        if segs:
            interpolate_children_linear(state, segs)


def position_interaction_blocks(state: CoarseState, segs: list[int], chr_: str) -> None:
    """
    Position IB clusters between segment positions.
    Mirrors Reference positionInteractionBlocks().

    When `use_ib_mc` is set, follows up the initial placement with a small MC
    pass on the IB centroids (chain bonds + excluded volume) so subsequent
    IB-level smooth-MC spheres have room to breathe.
    """
    clusters = state.clusters
    if len(segs) > 1:
        interpolate_children_linear(state, segs)
    else:
        # Random walk
        seg = clusters[segs[0]]
        pos: F32Array = np.zeros(3, dtype=np.float32)
        for ib_idx in seg.children:
            pos = pos + random_vector_np(100.0)
            clusters[ib_idx].pos = pos.copy()

    if state.s.use_ib_mc:
        ib_mc_refine(state, segs, chr_)


def ib_arc_target_distances(state: CoarseState, ibs: list[int], chr_: str) -> F64Array | None:
    """Attraction-only distance targets between block centroids, from arcs crossing boundaries.

    IB placement otherwise sees only the chain bonds between consecutive centroids, so two blocks
    joined by many CTCF loops are placed no closer than two joined by none.

    Two things this deliberately does not do, both learned from a version that failed.

    It does not feed the summed arc support to `freq_to_distance`. That law is calibrated for one
    arc's PET count between two anchors, and a sum over hundreds of arcs pins it to the
    `count_dist_base_level` floor: targets came out at 0.20 against chain bonds of 32 to 60. The
    support matrix is instead normalised so adjacent blocks sit at unit frequency and mapped
    through `freq_to_dist_heatmap`, the law the segment level uses for its own aggregate counts.

    And it does not keep every pair that normalisation produces. Normalising against adjacency
    encodes distance decay, so a non-adjacent pair with ordinary support gets a target LONGER
    than its genomic separation implies and the term pushes it apart. Long-range
    enhancer-promoter pairs live in exactly that population, and the first version made them
    worse in proportion to its weight. Only pairs whose arc support implies a distance shorter
    than `genomic_length_to_distance` of their separation are kept, so the term can pull blocks
    together and never drive them apart.

    Returns an (n, n) matrix aligned with `ibs`, zero for every pair without a kept target. Zero
    is the kernel's own skip value for its pairwise distance term, so those pairs cost nothing.
    Returns None when nothing survives, which leaves the term inert.
    """
    clusters = state.clusters
    chr_arcs = state.arcs.get(chr_, [])
    n = len(ibs)
    if not chr_arcs or n < 2:
        return None

    anchor_to_slot: dict[int, int] = {}
    for slot, ib in enumerate(ibs):
        for anchor in clusters[ib].children:
            anchor_to_slot[anchor] = slot

    support: F64Array = np.zeros((n, n), dtype=np.float64)
    for anchor, slot in anchor_to_slot.items():
        for arc_local in clusters[anchor].arcs:
            if arc_local >= len(chr_arcs):
                continue
            arc = chr_arcs[arc_local]
            other = arc.end if arc.start == anchor else arc.start
            other_slot = anchor_to_slot.get(other)
            # Each arc is reachable from both ends; the ordering counts it once.
            if other_slot is None or other_slot <= slot:
                continue
            w = float(arc.eff_score or arc.score)
            support[slot, other_slot] += w
            support[other_slot, slot] += w

    if not support.any():
        return None

    adjacent = np.diagonal(support, 1)
    ref = float(adjacent[adjacent > 0].mean()) if (adjacent > 0).any() else float(support.max())
    if ref <= 0.0:
        return None

    gpos = [clusters[ib].genomic_pos for ib in ibs]
    out: F64Array = np.zeros((n, n), dtype=np.float64)
    kept = 0
    for i, j in zip(*np.nonzero(np.triu(support, 1) > 0.0), strict=True):
        target = state.s.freq_to_dist_heatmap(float(support[i, j]) / ref)
        natural = state.s.genomic_length_to_distance(abs(int(gpos[j]) - int(gpos[i])))
        if target >= natural:
            continue  # no more attraction than the chain already provides: leave it alone
        out[i, j] = target
        out[j, i] = target
        kept += 1
    return out if kept else None


def ib_mc_refine(state: CoarseState, segs: list[int], chr_: str) -> None:
    """
    Refine IB centroid positions with a small chain-bond + EV + confinement
    MC pass.  Calls mc_ib directly - no settings clone, no field renaming.
    EV and confinement read their own IB-level settings (`*_ib`).  Opt-in via
    `use_ib_mc`.

    `ib_refine_scope` chooses what forms one chain.

    "segment", the default, refines each segment's blocks separately and skips a
    segment holding one block or fewer.  Placement then depends on how blocks are
    grouped, which is a property of the tree rather than of the chromatin, and
    denser segment boundaries leave more blocks wherever interpolation put them.

    "chromosome" refines every block on the chromosome as one chain, which removes
    that dependency.  It is not the default because it also puts every block pair
    inside the excluded-volume term, and the structures inflate: measured over four
    GM12878 regions the mean simulated contact density fell from 0.087 to 0.035 and
    within-block over between-block enrichment worsened about threefold.  The
    compartment saddle improves, but sparsity alone drags that statistic toward 1.0,
    so the gain is not separable from the inflation.  Using this scope means
    re-tuning the IB-level excluded-volume and confinement settings.
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    if str(getattr(s, "ib_refine_scope", "segment")).strip().lower() == "chromosome":
        groups = [
            sorted(
                (ib for seg_idx in segs for ib in clusters[seg_idx].children),
                key=lambda i: clusters[i].genomic_pos,
            )
        ]
    else:
        groups = [list(clusters[seg_idx].children) for seg_idx in segs]

    for ibs in groups:
        if len(ibs) <= 1:
            continue
        pos: F32Array = np.array([clusters[ib].pos for ib in ibs], dtype=np.float32)
        dtn: F32Array = np.zeros(len(ibs) - 1, dtype=np.float32)
        for i in range(len(ibs) - 1):
            gap = abs(clusters[ibs[i + 1]].genomic_pos - clusters[ibs[i]].genomic_pos)
            dtn[i] = float(s.genomic_length_to_distance(gap))

        avg_dtn = float(dtn.mean()) if dtn.size > 0 else 1.0
        step_size = avg_dtn * s.noise_ib

        log_ev_r0 = (
            s.exclusion_radius_ib
            if s.exclusion_radius_ib > 0.0
            else s.exclusion_auto_factor_ib * avg_dtn
        )
        log_conf_R = (
            s.confinement_radius_ib
            if s.confinement_radius_ib > 0.0
            else s.confinement_packing_factor_ib * avg_dtn * (len(ibs) ** (1.0 / 3.0))
        )
        conf_tag = (
            f", tether_R={log_conf_R:.2f}"
            if (s.use_confinement and s.confinement_apply_to_ib)
            else ""
        )

        with log.step(
            LOG,
            f"IB-MC {chr_} ({len(segs)} segments)",
            "%d IBs, avg_bond=%.2f, ev_r0=%.3f, step=%.3f%s",
            len(ibs),
            avg_dtn,
            log_ev_r0,
            step_size,
            conf_tag,
        ):
            gyr_before = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            arc_dist = ib_arc_target_distances(state, ibs, chr_) if s.use_ib_arcs else None
            if arc_dist is not None:
                LOG.info("IB arc targets: %d block pairs pulled in", int((arc_dist > 0).sum() // 2))
            mc_numba.mc_ib_numba(
                pos,
                dtn,
                step_size,
                s,
                compartment_for_clusters(state, ibs, chr_),
                accessibility_for_clusters(state, ibs, chr_),
                arc_dist,
            )
            gyr_after = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            LOG.info("gyr %.2f -> %.2f", gyr_before, gyr_after)

        for i, ib in enumerate(ibs):
            clusters[ib].pos = pos[i].copy()


# Re-export so callers can build the chr/segment distance heatmaps without
# reaching past this module into heatmap.
__all__ = [
    "CoarseState",
    "COARSE_SEED",
    "seed_global_rng",
    "build_state",
    "interpolate_children_linear",
    "compute_segment_bins",
    "add_long_pet_to_segment_heatmap",
    "calc_anchor_expected_distances",
    "subanchor_counts_per_arc",
    "build_contact_heatmaps",
    "create_distance_heatmap",
    # RNG-ordered positioning ops (the coarse MC spine)
    "reconstruct_heatmap",
    "place_single_segment",
    "reconstruct_chromosome_level",
    "random_walk_segment_level",
    "reconstruct_segment_level",
    "position_interaction_blocks",
    "ib_mc_refine",
    "ib_arc_target_distances",
]
