"""
Build and manage the hierarchical bead-and-spring model.

Levels mirror cudaMMC createTreeChromosome (LooperSolver.cpp:1021-1158):
    1 = chromosome root    (cudaMMC: curr_level=1 after two decrements)
    2 = segment            (cudaMMC: curr_level=2, ~segmentSize=2 Mb)
    3 = interaction block  (cudaMMC: curr_level=3, bounded by arc-free gaps)
    4 = anchor / leaf      (cudaMMC: curr_level=4, one per CTCF anchor)

cudaMMC source files:
  cudammc/src/LooperSolver.cpp   – createTreeChromosome, findGaps, findSplit,
                                   interpolateChildrenPositionSpline, genomicLengthToDistance
  cudammc/thirdparty/common.cpp  – interpolateSpline (Catmull-Rom), mirrorPoint
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch

from .data_structures import Anchor, Cluster, InteractionArc
from .distances import genomic_length_to_distance
from .settings import Settings


# ── Gap / split detection ────────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:856-894  findGaps(string chr)
#                 LooperSolver.cpp:900-962  findSplit(vector<int> gaps, int exp_size, string chr)
#
# Two-step hierarchy matching cudaMMC:
#   find_all_gaps (findGaps)  — arc-sweep: returns ALL anchor indices where no
#                               arc spans that position (arcs_cnt == 0).
#   find_segments (findSplit) — selects a coarse subset of those gaps as segment
#                               boundaries (~segment_size bp apart).
#   find_ibs                  — the REMAINING gap positions within each segment
#                               become IB boundaries (one IB per consecutive pair
#                               of gaps). This matches cudaMMC LooperSolver.cpp:
#                               1082-1138 where IBs come directly from gaps[].

def find_all_gaps(anchors: List[Anchor],
                  arcs: List[InteractionArc]) -> List[int]:
    """
    cudaMMC findGaps (LooperSolver.cpp:856-894): arc-sweep returning every anchor
    index i where arcs_cnt == 0 (no arc is currently active at i).

    Sweep order matches cudaMMC exactly (LooperSolver.cpp:866-888):
      For each position i:
        1. Process ALL arcs attached to i: if other_end > i → arcs_cnt++,
           if other_end < i → arcs_cnt--.
        2. After ALL arcs at i are processed, if arcs_cnt == 0, record gap.
      Position 0 and N-1 are always gaps (cpp:863, cpp:891).

    Consequence: touching non-overlapping arcs (A ends at X, B starts at X)
    do NOT create a gap at X — the decrement and increment cancel before the check.
    """
    N = len(anchors)
    if N == 0:
        return []

    # Precompute arc start/end lists indexed by anchor position
    # cudaMMC LooperSolver.cpp:864-888: arcs_cnt sweep
    ends_at: List[int] = [0] * N    # count of arcs whose hi == i
    starts_at: List[int] = [0] * N  # count of arcs whose lo == i
    for a in arcs:
        lo, hi = min(a.start, a.end), max(a.start, a.end)
        if 0 <= lo < N and 0 <= hi < N and lo != hi:
            starts_at[lo] += 1
            ends_at[hi] += 1

    # cudaMMC LooperSolver.cpp:863: gaps.push_back(start) — first position always a gap
    gaps: List[int] = [0]
    arcs_cnt = 0
    for i in range(N):
        # cudaMMC cpp:866-879: process ALL arcs at position i (both ending and starting)
        # THEN check gap at cpp:881. Both increments and decrements happen before the check,
        # so touching non-overlapping arcs (A ends at X, B starts at X) do NOT create a gap.
        arcs_cnt += starts_at[i] - ends_at[i]
        if arcs_cnt == 0 and i > 0:   # i==0 already added above
            gaps.append(i)
    # cudaMMC cpp:891: vector_insert_unique(gaps, clusters.size()-1) — last pos always a gap
    if N - 1 not in gaps:
        gaps.append(N - 1)
    return gaps


def find_segments(all_gaps: List[int], anchors: List[Anchor],
                  segment_size: int = 2_000_000,
                  segments_predefined: Optional[List[Tuple[int, int]]] = None
                  ) -> List[int]:
    """
    cudaMMC ``findSplit`` (LooperSolver.cpp:900-994).  Two branches:

    **Branch A — predefined breakpoints (cpp:911-962).**  If
    ``segments_predefined`` is non-empty (loaded from
    ``Settings.dataSegmentsSplit`` via :func:`data_loading.load_segments_split`),
    iterate the interior gaps; for each gap whose span
    ``[anchors[g].end, anchors[g+1].start]`` contains the *start* coordinate of
    a predefined region, promote that gap to a segment boundary.  The first and
    last gaps are always boundaries (cpp:913, 960).

    **Branch B — fallback (cpp:964-994).**  cudaMMC returns ``gaps`` unchanged
    after a diagnostic-only L/S computation; the ``exp_size`` parameter is
    dead code in this branch.  We mirror that exactly: ``return list(all_gaps)``.
    ``segment_size`` is therefore unused in Branch B and kept only for the
    signature parity (# cudaMMC: bug-preserved — dead parameter).
    """
    if not all_gaps:
        return [0]

    # ── Branch A: predefined-segments BED ────────────────────────────────
    # cudaMMC LooperSolver.cpp:911-962
    if segments_predefined:
        splits = [all_gaps[0]]
        curr_ind = 0
        last_seg_index = len(segments_predefined) - 1
        for i in range(1, len(all_gaps) - 1):
            g = all_gaps[i]
            # cpp:936-937: gap_start = clusters[gaps[i]].end,
            #              gap_end   = clusters[gaps[i]+1].start
            gap_start = anchors[g].end
            gap_end = anchors[g + 1].start if g + 1 < len(anchors) else gap_start
            seg_break = (segments_predefined[curr_ind][0]
                         if curr_ind <= last_seg_index else -1)
            # cpp:947-952: advance past predefined regions that end before this gap
            while seg_break >= 0 and seg_break < gap_start:
                curr_ind += 1
                seg_break = (segments_predefined[curr_ind][0]
                             if curr_ind <= last_seg_index else -1)
            # cpp:954-957: gap span contains a predefined breakpoint → split
            if 0 <= seg_break <= gap_end and seg_break >= gap_start:
                curr_ind += 1
                splits.append(g)
        # cpp:960: vector_insert_unique(splits, gaps.back())
        if all_gaps[-1] not in splits:
            splits.append(all_gaps[-1])
        return splits

    # ── Branch B: no predefined segments — return gaps unchanged ─────────
    # cudaMMC LooperSolver.cpp:964-994: every gap becomes a segment boundary;
    # each segment contains exactly ONE IB.  The L/S arrays are diagnostic.
    return list(all_gaps)


def find_ibs(all_gaps: List[int], seg_start: int, seg_end: int) -> List[int]:
    """
    cudaMMC IB creation (LooperSolver.cpp:1082-1138): IB boundaries within
    [seg_start, seg_end) are exactly the gaps from all_gaps that fall in that range.
    seg_start is always returned as the first boundary.
    """
    return [g for g in all_gaps if seg_start <= g < seg_end]


# ── Catmull-Rom spline interpolation ─────────────────────────────────────────
# cudaMMC source: common.cpp:452-462  interpolateSpline(float t, p1,p2,p3,p4)

def catmull_rom(t: float, p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> Tuple:
    """
    Catmull-Rom spline: returns interpolated (x, y, z) at parameter t ∈ [0,1].
    Interpolates between p2 and p3 using p1 and p4 as tangent control points.

    cudaMMC source: common.cpp:452-462  interpolateSpline(float t, p1,p2,p3,p4)
    Coefficients match exactly.
    """
    # cudaMMC common.cpp:455-461:
    #   t2=t*t; t3=t2*t;
    #   b1=.5*(-t3+2*t2-t); b2=.5*(3*t3-5*t2+2);
    #   b3=.5*(-3*t3+4*t2+t); b4=.5*(t3-t2);
    t2 = t * t
    t3 = t2 * t
    b1 = 0.5 * (-t3 + 2 * t2 - t)
    b2 = 0.5 * (3 * t3 - 5 * t2 + 2)
    b3 = 0.5 * (-3 * t3 + 4 * t2 + t)
    b4 = 0.5 * (t3 - t2)
    x = b1 * p1[0] + b2 * p2[0] + b3 * p3[0] + b4 * p4[0]
    y = b1 * p1[1] + b2 * p2[1] + b3 * p3[1] + b4 * p4[1]
    z = b1 * p1[2] + b2 * p2[2] + b3 * p3[2] + b4 * p4[2]
    return (x, y, z)


def interpolate_children_spline(parent_positions: List[Tuple[float, float, float]],
                                  n_children: int) -> List[Tuple[float, float, float]]:
    """
    Equidistant Catmull-Rom sampling helper.  ``n_children`` points are
    distributed uniformly along the spline through ``parent_positions``.

    cudaMMC source: ``LooperSolver.cpp:2939-3056``
    ``interpolateChildrenPositionSpline`` (equidistant branch, cpp:3021-3032)
    together with the **two-ghost reflection** at each end (cpp:2946-2954):

      pts[1] = mirrorPoint(P[0], P[1])              # first reflection
      pts[0] = mirrorPoint(pts[1], P[0])            # second reflection
      end_pt  = mirrorPoint(P[n-1], P[n-2])
      end_pt2 = mirrorPoint(end_pt, P[n-1])

    Mirrors :func:`mirrorPoint` from ``thirdparty/common.cpp:582``
    (``2*fixed - pt``).  This helper is used only for the cosmetic CIF
    output smoothing (solver.py:603); the algorithmic call path uses
    :meth:`ChromosomeTree.init_children_from_parents_spline` which carries
    per-parent knot lists and the sliding control-point window.
    """
    M = len(parent_positions)
    if M < 2:
        return [parent_positions[0]] * n_children if M == 1 else []
    if n_children == 0:
        return []

    def mirror(fixed, pt):
        return (2 * fixed[0] - pt[0], 2 * fixed[1] - pt[1], 2 * fixed[2] - pt[2])

    # cudaMMC cpp:2946-2954 — TWO reflections at each end
    p0, p1 = parent_positions[0], parent_positions[1]
    pm2, pm1 = parent_positions[-2], parent_positions[-1]
    g1 = mirror(p0, p1)          # cpp:2946 pts[1]
    g0 = mirror(g1, p0)          # cpp:2947 pts[0]
    e1 = mirror(pm1, pm2)        # cpp:2952-2953 end_pt
    e2 = mirror(e1, pm1)         # cpp:2954 end_pt2
    pts = [g0, g1] + list(parent_positions) + [e1, e2]

    # Equidistant parameterisation matching cudaMMC cpp:3022-3032:
    # for each parent i we lay (n / M) children on knots starting at 0.5,
    # incrementing by dst=1/n, wrapping at 1.0 with control-point slide.
    # Here we use the simpler global linear parameterisation since this is
    # only the cosmetic CIF smoother (caller passes raw anchor positions).
    n_segments = M - 1
    result = []
    for k in range(n_children):
        u = k / max(n_children - 1, 1) * n_segments
        seg = min(int(u), n_segments - 1)
        t = u - seg
        # Use 4 control points pts[seg..seg+3] (pts now offset by 2 ghosts)
        result.append(catmull_rom(t, pts[seg + 1], pts[seg + 2],
                                  pts[seg + 3], pts[seg + 4]
                                  if seg + 4 < len(pts) else pts[-1]))
    return result


# ── Tree builder ──────────────────────────────────────────────────────────────
# cudaMMC source: LooperSolver.cpp:1021-1158  createTreeChromosome(string chr)

class ChromosomeTree:
    """
    Complete hierarchical model for one chromosome.

    clusters: flat list of Cluster objects.
    anchors_idx: [cluster index for each anchor (level-4 bead)]

    cudaMMC source: LooperSolver.cpp:1021-1158  createTreeChromosome(string chr)
    Level numbering (cudaMMC cpp:1026-1035):
        4 = anchor (leaf)  — curr_level=4 when anchors created
        3 = interaction block — curr_level-- at cpp:1083
        2 = segment        — cs.level = curr_level-1 at cpp:1121
        1 = chromosome root — rootc.level = curr_level after curr_level-=2 at cpp:1142
    """

    def __init__(self, chrom: str, anchors: List[Anchor],
                 arcs: List[InteractionArc], settings: Settings,
                 segments_predefined: Optional[List[Tuple[int, int]]] = None):
        self.chrom = chrom
        self.settings = settings
        self.segments_predefined = segments_predefined or []
        self.clusters: List[Cluster] = []
        self.anchors_idx: List[int] = []  # cluster indices for level-4 beads
        self._build(anchors, arcs)

    # ── Private build ─────────────────────────────────────────────────────────

    def _new_cluster(self, level: int, start: int, end: int,
                     parent: int = -1) -> int:
        # cudaMMC LooperSolver.cpp:1031: Cluster c(start, end); c.level = curr_level
        # cudaMMC LooperSolver.cpp:1149: rootc.genomic_pos = (rootc.start + rootc.end)/2
        idx = len(self.clusters)
        gpos = (start + end) // 2
        c = Cluster(self.chrom, start, end, gpos, level=level, parent=parent)
        c.base_start = start
        c.base_end = end
        self.clusters.append(c)
        return idx

    def _build(self, anchors: List[Anchor], arcs: List[InteractionArc]):
        # cudaMMC LooperSolver.cpp:1021-1158  createTreeChromosome
        N = len(anchors)
        if N == 0:
            return

        ss = self.settings

        # cudaMMC LooperSolver.cpp:1030-1036: create level-4 anchor clusters
        # level-1: chromosome root (created last in cudaMMC, first here for index clarity)
        root_idx = self._new_cluster(1, anchors[0].start, anchors[-1].end)
        root = self.clusters[root_idx]

        # cudaMMC LooperSolver.cpp:856-962: findGaps → all arc-free positions,
        # findSplit → coarse segment boundaries from those positions.
        all_gaps = find_all_gaps(anchors, arcs)
        seg_starts = find_segments(all_gaps, anchors, ss.segment_size,
                                   self.segments_predefined)
        seg_starts.append(N)  # sentinel (cudaMMC: vector_insert_unique(gaps, clusters.size()-1))

        for s_i, seg_s in enumerate(seg_starts[:-1]):
            seg_e = seg_starts[s_i + 1]  # exclusive

            # cudaMMC LooperSolver.cpp:1113-1134: create segment cluster (level curr_level-1 = 2)
            seg_idx = self._new_cluster(2,
                                         anchors[seg_s].start,
                                         anchors[seg_e - 1].end,
                                         root_idx)
            root.children.append(seg_idx)

            # cudaMMC LooperSolver.cpp:1084-1111: IBs come from all_gaps within segment.
            # cpp:1087-1089: prev_gap = (i==1 ? gaps[i-1] : gaps[i-1]+1)  "boundary gaps (both inclusive)"
            # cpp:1106:      for k in [prev_gap, curr_gap]  ← inclusive on BOTH ends
            ib_gaps = find_ibs(all_gaps, seg_s, seg_e)
            # ib_gaps = [seg_s, (interior gaps...), seg_e-1]

            for ib_i in range(1, len(ib_gaps)):
                # Mirror cpp:1087-1089 exactly
                prev_gap = ib_gaps[0] if ib_i == 1 else ib_gaps[ib_i - 1] + 1
                curr_gap = ib_gaps[ib_i]

                # cudaMMC LooperSolver.cpp:1098-1111: Cluster c(start_pos, end_pos); c.level=3
                ib_idx = self._new_cluster(3,
                                           anchors[prev_gap].start,
                                           anchors[curr_gap].end,
                                           seg_idx)
                self.clusters[seg_idx].children.append(ib_idx)

                # cudaMMC LooperSolver.cpp:1106: for (int k = prev_gap; k <= curr_gap; ++k)
                for ai in range(prev_gap, curr_gap + 1):  # inclusive
                    a = anchors[ai]
                    anc_idx = self._new_cluster(4, a.start, a.end, ib_idx)
                    anc_c = self.clusters[anc_idx]
                    # cudaMMC LooperSolver.cpp:1032: c.orientation = arcs.anchors[chr][i].orientation
                    anc_c.orientation = a.orientation
                    anc_c.genomic_pos = a.mid
                    self.clusters[ib_idx].children.append(anc_idx)
                    self.anchors_idx.append(anc_idx)

        # cudaMMC LooperSolver.cpp:1039-1048: wire arcs onto anchor clusters.
        # cudaMMC unconditionally pushes the arc index on BOTH endpoints
        # (cpp:1047-1048) — it relies on markArcs having validated both ends.
        # We mirror that strictness: skip the arc entirely if either endpoint
        # is out of range, rather than producing a half-bound dangling arc
        # (AUDIT §H7).
        N_anc = len(self.anchors_idx)
        for arc_i, arc in enumerate(arcs):
            if arc.start >= N_anc or arc.end >= N_anc:
                continue
            self.clusters[self.anchors_idx[arc.start]].arcs.append(arc_i)
            self.clusters[self.anchors_idx[arc.end]].arcs.append(arc_i)

    # ── Position initialisation (top-down, cudaMMC-style) ───────────────────
    # The bottom-up `init_positions_linear` / `init_positions_random` /
    # `_propagate_positions_up` helpers from the old Python prototype have
    # been deleted (AUDIT §H9): cudaMMC always places **parents first**, then
    # children inherit `parent.pos + random_vector(avg_dist)` or are sampled
    # along a Catmull-Rom spline through the parent chain.

    def init_positions_random_walk(self, step: Optional[float] = None,
                                    seed: Optional[int] = None):
        """Random-walk initialisation for **direct children of the root**
        (segment beads, level 2).

        Mirrors cudaMMC ``positionInteractionBlocks`` single-segment branch
        (LooperSolver.cpp:2716-2722) and the top-level ``runLooper`` random-walk
        path: start at the origin, then for each subsequent bead displace by a
        uniform vector in a cube of side ``2*step``.  Step size defaults to
        ``Settings.ib_random_walk_jumps`` (Settings.cpp:192).
        """
        if step is None:
            step = self.settings.ib_random_walk_jumps
        rng = random.Random(seed) if seed is not None else random
        roots = [i for i, c in enumerate(self.clusters) if c.level == 1]
        for ri in roots:
            self.clusters[ri].set_pos(0.0, 0.0, 0.0)
            x = y = z = 0.0
            for ch in self.clusters[ri].children:
                # cudaMMC common.cpp:14-25 random_vector: (2u-1)*step per axis
                x += (2.0 * rng.random() - 1.0) * step
                y += (2.0 * rng.random() - 1.0) * step
                if not self.settings.use_2d:
                    z += (2.0 * rng.random() - 1.0) * step
                self.clusters[ch].set_pos(x, y, z)

    def position_interaction_blocks(self, parent_level: int = 2,
                                    step: Optional[float] = None):
        """Place IBs (children of segments) along the segment chain.

        Mirrors cudaMMC ``positionInteractionBlocks`` (LooperSolver.cpp:2709-2725).
        For each segment with ≥ 2 sibling segments → spline interpolation
        (genomic-distance mode).  For a singleton segment → random-walk fallback
        with step size ``Settings.ib_random_walk_jumps``.
        """
        if step is None:
            step = self.settings.ib_random_walk_jumps
        # cpp:2710-2712: if segments.size() > 1 → spline, else → random walk
        # We dispatch per grand-parent (chromosome root) so multi-chrom builds
        # behave identically.
        by_root: Dict[int, List[int]] = {}
        for ci, c in enumerate(self.clusters):
            if c.level == parent_level:
                by_root.setdefault(c.parent, []).append(ci)
        for root_idx, sibs in by_root.items():
            sibs.sort(key=lambda i: self.clusters[i].genomic_pos)
            if len(sibs) > 1:
                # cpp:2712: interpolateChildrenPositionSpline(segments, true)
                self._interp_children_spline_region(sibs, use_genomic_dist=True)
            else:
                # cpp:2716-2722: random walk inside the lone segment
                seg = self.clusters[sibs[0]]
                x, y, z = seg.x, seg.y, seg.z
                for ch in seg.children:
                    x += (2.0 * random.random() - 1.0) * step
                    y += (2.0 * random.random() - 1.0) * step
                    if not self.settings.use_2d:
                        z += (2.0 * random.random() - 1.0) * step
                    self.clusters[ch].set_pos(x, y, z)

    def init_anchor_positions_from_parent(self, avg_dist: float):
        """Place anchors at ``parent.pos + random_vector(avg_dist)``.

        Mirrors cudaMMC ``reconstructClusterArcsDistances`` initial loop
        (LooperSolver.cpp:2624-2625) and ``reconstructClustersHeatmapSingleLevel``
        cpp:357-363.  Called by the solver once IB positions are settled.
        """
        for c in self.clusters:
            if c.level == 4:
                p = self.clusters[c.parent]
                # cudaMMC common.cpp:14-25 random_vector (uniform per-axis)
                dx = (2.0 * random.random() - 1.0) * avg_dist
                dy = (2.0 * random.random() - 1.0) * avg_dist
                dz = 0.0 if self.settings.use_2d else \
                     (2.0 * random.random() - 1.0) * avg_dist
                c.set_pos(p.x + dx, p.y + dy, p.z + dz)

    def densify_active_region(self, ib_cidx: int, fix: bool = True,
                                add: Optional[int] = None) -> List[int]:
        """Insert ``loopDensity`` subanchor beads between each pair of anchors.

        Mirrors cudaMMC ``densifyActiveRegion`` (LooperSolver.cpp:2448-2510):

          * for each consecutive anchor pair ``(i, i+1)``: insert ``add`` new
            beads at linearly-interpolated positions (cpp:2485-2497, ``c.pos =
            interpolate(a.pos, b.pos, st)``);
          * anchors are marked ``is_fixed = True`` when ``fix`` is set (cpp:2466,
            cpp:2502);
          * new beads inherit ``level = anchor_level + 1`` (cpp:2461) and have
            ``start = end = p`` where ``p = a.end + j*d`` is the genomic position
            (cpp:2486, 2488);
          * the IB cluster's ``children`` list is rewritten to be the new dense
            active region (cpp:2509).

        Returns the new list of cluster indices (anchors + inserted subanchors)
        in chain order.  ``add`` defaults to ``Settings.loop_density``
        (= 5, cudaMMC ``Settings::loopDensity``, Settings.cpp:152).
        """
        ib = self.clusters[ib_cidx]
        active = list(ib.children)
        if len(active) == 0:
            return active
        if add is None:
            add = self.settings.loop_density
        # cpp:2461: level for newly added points
        new_level = self.clusters[active[0]].level + 1

        new_active: List[int] = []
        for i in range(len(active) - 1):
            a_idx = active[i]
            b_idx = active[i + 1]
            a = self.clusters[a_idx]
            b = self.clusters[b_idx]
            if fix:
                a.is_fixed = True                                # cpp:2466
            new_active.append(a_idx)
            # cpp:2469-2472: genomic step between the right edge of `a` and the
            # left edge of `b`.  range = b.start - a.end ; d = range / (add+1).
            grange = b.start - a.end
            d_gen = grange // (add + 1) if (add + 1) > 0 else 0
            p_gen = a.end                                        # cpp:2472
            dst = 1.0 / (add + 1)                                # cpp:2482
            st = dst                                             # cpp:2483
            for _j in range(add):
                p_gen += d_gen                                   # cpp:2486
                # cpp:2488: Cluster c(p, p)   start == end == p
                # cpp:2489-2490: c.pos = interpolate(a.pos, b.pos, st)
                px = a.x + (b.x - a.x) * st
                py = a.y + (b.y - a.y) * st
                pz = a.z + (b.z - a.z) * st
                sub_idx = self._new_cluster(new_level, p_gen, p_gen, ib_cidx)
                self.clusters[sub_idx].set_pos(px, py, pz)
                new_active.append(sub_idx)
                st += dst                                        # cpp:2497
        # cpp:2501-2503: append the trailing anchor (and mark it fixed too)
        last = active[-1]
        new_active.append(last)
        if fix:
            self.clusters[last].is_fixed = True
        # cpp:2509: replace the IB's children list with the dense active region
        ib.children = new_active
        return new_active

    # ── Phase-5-pending shim (TO BE REMOVED) ─────────────────────────────────
    # AUDIT §G14: bottom-up propagation has NO cudaMMC counterpart and
    # produces collapsed IB centroids when paired with random-sphere anchor
    # init.  It is kept here only so the legacy solver.py (which has not yet
    # been rewritten to the top-down cascade — Phase 5) can still execute the
    # pipeline end-to-end.  All callers in solver.py are flagged with
    # `# TODO Phase 5` and will be replaced by `position_interaction_blocks`
    # + `init_anchor_positions_from_parent` once the multi-level heatmap MC
    # lands.
    def _propagate_positions_up(self):
        for level in [3, 2, 1]:
            for c in self.clusters:
                if c.level == level and c.children:
                    xs = [self.clusters[ch].x for ch in c.children]
                    ys = [self.clusters[ch].y for ch in c.children]
                    zs = [self.clusters[ch].z for ch in c.children]
                    c.set_pos(sum(xs) / len(xs),
                              sum(ys) / len(ys),
                              sum(zs) / len(zs))

    def init_children_from_parents_spline(self, parent_level: int,
                                          use_genomic_dist: bool = False):
        """
        Port of cudaMMC ``interpolateChildrenPositionSpline``
        (LooperSolver.cpp:2939-3056).  For every parent at ``parent_level``,
        sample children positions along a Catmull-Rom spline through the
        ordered list of parents (sharing the same grand-parent).

        Implements:
          * **two-ghost reflections** at each end of the parent chain
            (cpp:2946-2954) using ``mirrorPoint`` from
            ``thirdparty/common.cpp:582`` (= ``2*fixed - pt``);
          * **sliding 4-control-point window** per parent — when a child
            crosses parameter 1.0, slide the 4 control points forward by one
            (cpp:3041-3050); ``end_pt`` / ``end_pt2`` fill the last two slots;
          * **equidistant mode** (cpp:3022-3032): ``dst=1/n``,
            ``st=0.5+dst/2``, wrap at 1.0;
          * **genomic-distance mode** (cpp:2969-3020) when
            ``use_genomic_dist=True``: per-child knot in [0.5, 1.0) for
            ``genomic_pos < center`` and [0.0, 0.5) otherwise, with
            flanking-aware boundaries.
        """
        # Group parents by their grand-parent so each spline runs through a
        # contiguous chain of siblings — matches cudaMMC's per-region call.
        # cudaMMC dispatches with ``current_level[chr]`` which is exactly the
        # full ordered parent list of a single chromosome at one level.
        by_grandparent: Dict[int, List[int]] = {}
        for ci, c in enumerate(self.clusters):
            if c.level == parent_level:
                by_grandparent.setdefault(c.parent, []).append(ci)

        for region_idxs in by_grandparent.values():
            region_idxs.sort(key=lambda i: self.clusters[i].genomic_pos)
            self._interp_children_spline_region(region_idxs, use_genomic_dist)

    def _interp_children_spline_region(self, regions: List[int],
                                       use_genomic_dist: bool):
        """One call of cudaMMC ``interpolateChildrenPositionSpline`` over the
        parent indices in ``regions``.  See cpp:2939-3056."""
        reg_cnt = len(regions)
        if reg_cnt == 0:
            return
        if reg_cnt < 2:
            # Single parent → place all children at the parent's pos.
            p = self.clusters[regions[0]]
            for ch in p.children:
                self.clusters[ch].set_pos(p.x, p.y, p.z)
            return

        def mirror(fixed, pt):
            return (2 * fixed[0] - pt[0], 2 * fixed[1] - pt[1], 2 * fixed[2] - pt[2])

        # cudaMMC cpp:2946-2949: initial 4-control-point window
        p_pos = lambda i: self.clusters[regions[i]].pos
        _zero = (0.0, 0.0, 0.0)
        pts: List[Tuple[float, float, float]] = [_zero, _zero, _zero, _zero]
        pts[1] = mirror(p_pos(0), p_pos(1))      # cpp:2946
        pts[0] = mirror(pts[1], p_pos(0))         # cpp:2947
        pts[2] = p_pos(0)                          # cpp:2948-2949 (i=2..3)
        pts[3] = p_pos(1)
        end_pt = mirror(p_pos(reg_cnt - 1), p_pos(reg_cnt - 2))  # cpp:2952-2953
        end_pt2 = mirror(end_pt, p_pos(reg_cnt - 1))             # cpp:2954

        for i in range(reg_cnt):
            parent = self.clusters[regions[i]]
            children = parent.children
            n = len(children)
            if n == 0:
                continue

            switch_controls = -1
            knots: List[float] = []

            if use_genomic_dist:
                # cudaMMC cpp:2985-3020
                center_loc = parent.genomic_pos
                start_loc = parent.start
                end_loc = parent.end
                left_flank = (start_loc if i == 0
                              else self.clusters[regions[i - 1]].end)
                right_flank = (end_loc if i == reg_cnt - 1
                               else self.clusters[regions[i + 1]].start)
                start_loc = (left_flank + start_loc) // 2
                end_loc = (right_flank + end_loc) // 2
                left_length = max(float(center_loc - start_loc), 1e-6)
                right_length = max(float(end_loc - center_loc), 1e-6)
                for j, ch in enumerate(children):
                    cpos = self.clusters[ch].genomic_pos
                    if cpos < center_loc:
                        p = (cpos - start_loc) / left_length
                        p = 0.5 + p * 0.5
                    else:
                        p = (cpos - center_loc) / right_length
                        p = p * 0.5
                        if switch_controls == -1:
                            switch_controls = j
                    knots.append(p)
            else:
                # cudaMMC cpp:3021-3032 equidistant
                dst = 1.0 / n
                st = 0.5 + dst / 2.0
                for j in range(n):
                    knots.append(st)
                    st += dst
                    if st > 1.0:
                        if switch_controls == -1:
                            switch_controls = j
                        st -= 1.0

            # cudaMMC cpp:3041-3053: iterate children, slide pts[] forward at switch.
            for j in range(n):
                if j == switch_controls or (switch_controls == -1 and j == n - 1):
                    pts[0] = pts[1]; pts[1] = pts[2]; pts[2] = pts[3]
                    if i + 2 == reg_cnt:
                        pts[3] = end_pt
                    elif i + 2 == reg_cnt + 1:
                        pts[3] = end_pt2
                    else:
                        pts[3] = p_pos(i + 2)
                pos = catmull_rom(knots[j], pts[0], pts[1], pts[2], pts[3])
                self.clusters[children[j]].set_pos(*pos)

    # ── Orientation (recomputed from neighbour geometry) ──────────────────
    # cudaMMC source: LooperSolver.cpp:3437-3454  calcOrientation(int cind)
    # Tangent direction from neighbour anchors, flipped for 'L'-strand anchors,
    # normalised to unit length.  Replaces the static-label approximation.

    def calc_orientation(self, anchor_idx: int,
                         active_region: Optional[List[int]] = None
                         ) -> Tuple[float, float, float]:
        """Geometric orientation of an anchor.

        ``anchor_idx`` is the index into ``active_region`` (= a list of
        cluster indices forming an ordered chain — mirrors cudaMMC's
        ``active_region``).  When ``active_region`` is ``None`` we use the
        full anchor chain (``self.anchors_idx``).

        cudaMMC LooperSolver.cpp:3437-3454::

            if cind == 0:      orn = pos[1]   - pos[0]
            elif cind == last: orn = pos[end] - pos[end-1]
            else:              orn = pos[cind+1] - pos[cind-1]
            if orientation_label == 'L': orn *= -1
            orn.normalize()
        """
        ar = active_region if active_region is not None else self.anchors_idx
        n = len(ar)
        if n == 0:
            return (0.0, 0.0, 0.0)
        c = self.clusters[ar[anchor_idx]]
        if anchor_idx == 0:
            a = self.clusters[ar[anchor_idx + 1]] if n > 1 else c
            ox, oy, oz = a.x - c.x, a.y - c.y, a.z - c.z
        elif anchor_idx == n - 1:
            b = self.clusters[ar[anchor_idx - 1]]
            ox, oy, oz = c.x - b.x, c.y - b.y, c.z - b.z
        else:
            a = self.clusters[ar[anchor_idx + 1]]
            b = self.clusters[ar[anchor_idx - 1]]
            ox, oy, oz = a.x - b.x, a.y - b.y, a.z - b.z
        # cpp:3449-3450: strand-'L' anchors invert the tangent
        if c.orientation == 'L':
            ox, oy, oz = -ox, -oy, -oz
        # cpp:3452: normalise
        norm = math.sqrt(ox * ox + oy * oy + oz * oz)
        if norm < 1e-12:
            return (0.0, 0.0, 0.0)
        return (ox / norm, oy / norm, oz / norm)


    # ── Tensor extraction ─────────────────────────────────────────────────────

    def positions_tensor(self, indices: Optional[List[int]] = None,
                         device: str = "cpu") -> torch.Tensor:
        """Return (N, 3) float32 tensor of positions for given cluster indices."""
        if indices is None:
            indices = list(range(len(self.clusters)))
        data = [[self.clusters[i].x, self.clusters[i].y, self.clusters[i].z]
                for i in indices]
        return torch.tensor(data, dtype=torch.float32, device=device)

    def set_positions_from_tensor(self, pos: torch.Tensor,
                                   indices: Optional[List[int]] = None):
        """Write positions back from a (N, 3) tensor."""
        if indices is None:
            indices = list(range(len(self.clusters)))
        pos_cpu = pos.detach().cpu().tolist()
        for i, idx in enumerate(indices):
            self.clusters[idx].set_pos(*pos_cpu[i])

    def anchor_positions_tensor(self, device: str = "cpu") -> torch.Tensor:
        return self.positions_tensor(self.anchors_idx, device)

    def set_anchor_positions_from_tensor(self, pos: torch.Tensor):
        self.set_positions_from_tensor(pos, self.anchors_idx)

    def chain_lengths_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Expected linker lengths between consecutive anchors, shape (N-1,).

        cudaMMC source: LooperSolver.cpp:2767-2771
          int d = abs(clusters[ar[i+1]].genomic_pos - clusters[ar[i]].genomic_pos);
          dist = genomicLengthToDistance(d);  // base + scale*(d/1000)^power
          clusters[ar[i]].dist_to_next = dist;

        cudaMMC uses CENTER-TO-CENTER genomic distance between consecutive anchor
        midpoints.  Using gap-only (c2.start - c1.end) would give a shorter
        distance: ~half for typical ~1kb anchors spaced ~2kb apart.

        genomicLengthToDistance (LooperSolver.cpp:2512-2516):
          return base + scale * pow(length/1000.0, power)
          defaults: base=0.0, scale=1.0, power=0.5  (Settings.cpp:212-214)
        """
        from .distances import genomic_length_to_distance as g2d
        s = self.settings
        n = len(self.anchors_idx)
        lengths = []
        for i in range(n - 1):
            c1 = self.clusters[self.anchors_idx[i]]
            c2 = self.clusters[self.anchors_idx[i + 1]]
            # cudaMMC LooperSolver.cpp:2768-2769:
            #   d = abs(clusters[ar[i+1]].genomic_pos - clusters[ar[i]].genomic_pos)
            # Center-to-center distance (both midpoints); never negative.
            d = abs(c2.genomic_pos - c1.genomic_pos)
            lengths.append(g2d(d, s.genomic_dist_scale,
                               s.genomic_dist_power, s.genomic_dist_base))
        return torch.tensor(lengths, dtype=torch.float32, device=device)
