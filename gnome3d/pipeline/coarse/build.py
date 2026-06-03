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
            data.anchors, data.arcs, data.breakpoints, chrs_list
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
    )


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
    mat: F64Array = np.full((n, n), -1.0, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}
    chr_arcs = state.arcs.get(chr_, [])

    for ai, ci in enumerate(active_region):
        for arc_local in clusters[ci].arcs:
            if arc_local >= len(chr_arcs):
                continue
            arc = chr_arcs[arc_local]
            other = arc.end if arc.start == ci else arc.start

            if other < ci or other not in cluster_to_active:
                continue

            bi = cluster_to_active[other]
            exp_d = s.freq_to_distance(arc.score)
            mat[ai, bi] = exp_d
            mat[bi, ai] = exp_d

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

    return mat


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

    # Concatenate all segment indices into active_region
    active_region: list[int] = []
    for chr_ in state.chrs:
        active_region.extend(current_level.get(chr_, []))

    if len(active_region) <= 1:
        return

    step_size = avg_dist * s.noise_lvl2

    pos: F32Array = np.array([clusters[i].pos for i in active_region], dtype=np.float32)
    n = len(active_region)
    best_score = -1.0
    best_pos: F32Array = pos.copy()

    for run in range(s.steps_lvl2):
        with log.step(LOG, f"heatmap run {run + 1}/{s.steps_lvl2}", "(%d beads)", n):
            for i in range(n):
                pos[i] = clusters[active_region[i]].pos + random_vector_np(step_size)

            score = mc_numba.mc_heatmap_numba(pos, heatmap_dist, heatmap_dist_diag, step_size, s)
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


def position_interaction_blocks(state: CoarseState, segs: list[int]) -> None:
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
        ib_mc_refine(state, segs)


def ib_mc_refine(state: CoarseState, segs: list[int]) -> None:
    """
    Refine IB centroid positions with a small chain-bond + EV + confinement
    MC pass.  Calls mc_ib directly - no settings clone, no field renaming.
    Each segment's IB centroids form a chain (bond targets from
    genomic_length_to_distance).  EV and confinement read their own
    IB-level settings (`*_ib`).  Opt-in via `use_ib_mc`.
    """
    from gnome3d.mc import numba as mc_numba

    s = state.s
    clusters = state.clusters
    for seg_idx in segs:
        ibs = list(clusters[seg_idx].children)
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
            f"IB-MC seg{seg_idx}",
            "%d IBs, avg_bond=%.2f, ev_r0=%.3f, step=%.3f%s",
            len(ibs),
            avg_dtn,
            log_ev_r0,
            step_size,
            conf_tag,
        ):
            gyr_before = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
            mc_numba.mc_ib_numba(pos, dtn, step_size, s)
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
]
