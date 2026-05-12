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
                  segment_size: int = 2_000_000) -> List[int]:
    """
    cudaMMC findSplit: select segment boundaries from all_gaps.
    A new segment starts at a gap when:
      - The genomic gap from the previous anchor to this one exceeds segment_size, OR
      - The genomic distance from the last segment boundary to this gap ≥ segment_size.
    Index 0 (= all_gaps[0]) is always a segment boundary.
    """
    if not all_gaps:
        return [0]

    boundaries = [all_gaps[0]]
    for g in all_gaps[1:]:
        # Always split on a large physical gap between consecutive anchors
        gap_bp = anchors[g].start - anchors[g - 1].end
        if gap_bp > segment_size:
            boundaries.append(g)
            continue
        # cudaMMC findSplit: split once the accumulated genomic span hits segment_size
        last_b = boundaries[-1]
        if anchors[g].mid - anchors[last_b].mid >= segment_size:
            boundaries.append(g)
    return boundaries


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
    Given M parent control points, generate n_children equidistant points
    along the Catmull-Rom spline through them.

    cudaMMC source: LooperSolver.cpp:2939-3056  interpolateChildrenPositionSpline()
                    common.cpp:582  mirrorPoint(fixed, pt) = 2*fixed - pt

    Ghost point construction matches: mirrorPoint(P[0], P[1]) = 2*P[0]-P[1].

    Parameterisation difference vs cudaMMC equidistant mode:
      cudaMMC: st = 0.5 + dst/2, increments by dst=1/n, wraps at 1.0 with
               window shift (children span the [0.5, 1.5) range per segment).
      Python:  u = k/(n_children-1) * n_segments  (spans [0, n_segments] uniformly).
    The two produce slightly different child positions within each parent's range.

    Also: cudaMMC prepends TWO ghost points (lines 2946-2949):
      pts[1] = mirrorPoint(P[0], P[1])        = 2*P[0]-P[1]
      pts[0] = mirrorPoint(pts[1], P[0])      = 3*P[0]-2*P[1]  (second reflection)
    Python prepends ONE ghost (ghost_start = 2*P[0]-P[1]).
    The double-reflection gives cudaMMC a sharper start tangent.
    """
    M = len(parent_positions)
    if M < 2:
        return [parent_positions[0]] * n_children if M == 1 else []
    if n_children == 0:
        return []

    # cudaMMC common.cpp:582: mirrorPoint(fixed, pt) = 2*fixed - pt
    def ghost_start(p0, p1):
        return (2 * p0[0] - p1[0], 2 * p0[1] - p1[1], 2 * p0[2] - p1[2])

    def ghost_end(pm2, pm1):
        return (2 * pm1[0] - pm2[0], 2 * pm1[1] - pm2[1], 2 * pm1[2] - pm2[2])

    # ONE ghost at each end (cudaMMC uses two ghost reflections at each end)
    pts = ([ghost_start(parent_positions[0], parent_positions[1])]
           + parent_positions
           + [ghost_end(parent_positions[-2], parent_positions[-1])])

    # sample n_children points uniformly in parameter space
    n_segments = M - 1
    result = []
    for k in range(n_children):
        u = k / max(n_children - 1, 1) * n_segments
        seg = int(u)
        seg = min(seg, n_segments - 1)
        t = u - seg
        p1 = pts[seg]
        p2 = pts[seg + 1]
        p3 = pts[seg + 2]
        p4 = pts[seg + 3]
        # cudaMMC LooperSolver.cpp:3052: pos = interpolateSpline(knots[j], pts[0..3])
        result.append(catmull_rom(t, p1, p2, p3, p4))

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
                 arcs: List[InteractionArc], settings: Settings):
        self.chrom = chrom
        self.settings = settings
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
        seg_starts = find_segments(all_gaps, anchors, ss.segment_size)
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

        # cudaMMC LooperSolver.cpp:1039-1048: wire arcs onto anchor clusters
        for arc_i, arc in enumerate(arcs):
            if arc.start < len(self.anchors_idx):
                self.clusters[self.anchors_idx[arc.start]].arcs.append(arc_i)
            if arc.end < len(self.anchors_idx):
                self.clusters[self.anchors_idx[arc.end]].arcs.append(arc_i)

    # ── Position initialisation ───────────────────────────────────────────────

    def init_positions_random(self, radius: float = 10.0, seed: int = 42):
        """Place all clusters randomly inside a sphere of given radius."""
        # No direct cudaMMC equivalent for this helper.
        # cudaMMC uses setLevel + MonteCarloHeatmap/Arcs with random initial structure.
        rng = random.Random(seed)
        for c in self.clusters:
            r = radius * rng.random() ** (1 / 3)
            theta = math.acos(2 * rng.random() - 1)
            phi = 2 * math.pi * rng.random()
            c.set_pos(r * math.sin(theta) * math.cos(phi),
                      r * math.sin(theta) * math.sin(phi),
                      r * math.cos(theta))

    def init_positions_linear(self):
        """Place anchors along a line; parent clusters at their midpoints."""
        n = len(self.anchors_idx)
        for i, idx in enumerate(self.anchors_idx):
            self.clusters[idx].set_pos(float(i), 0.0, 0.0)
        self._propagate_positions_up()

    def _propagate_positions_up(self):
        """Set each non-leaf cluster position to mean of its children."""
        # No direct cudaMMC equivalent — Python-specific helper for bottom-up init.
        for level in [3, 2, 1]:
            for c in self.clusters:
                if c.level == level and c.children:
                    xs = [self.clusters[ch].x for ch in c.children]
                    ys = [self.clusters[ch].y for ch in c.children]
                    zs = [self.clusters[ch].z for ch in c.children]
                    c.set_pos(sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))

    def init_children_from_parents_spline(self, parent_level: int):
        """
        For all clusters at `parent_level`, interpolate their children's
        positions using Catmull-Rom spline.

        cudaMMC source: LooperSolver.cpp:2939-3056  interpolateChildrenPositionSpline()
        See interpolate_children_spline docstring for parameterisation differences.
        """
        # cudaMMC LooperSolver.cpp:2958: for every parent cluster i in regions
        parents = sorted([c for c in self.clusters if c.level == parent_level],
                         key=lambda c: c.genomic_pos)
        for p in parents:
            if not p.children:
                continue
            # cudaMMC: uses sibling regions as control points (regions[i-1..i+2])
            siblings = (self.clusters[p.parent].children
                        if p.parent >= 0 else [p])
            # cudaMMC LooperSolver.cpp:2946: pts[] = cluster positions of regions[i-2..i+1]
            ctrl_pts = [self.clusters[s].pos for s in siblings]
            # cudaMMC LooperSolver.cpp:3041-3053: iterate children, call interpolateSpline
            child_positions = interpolate_children_spline(ctrl_pts, len(p.children))
            for ci, ch_idx in enumerate(p.children):
                ch = self.clusters[ch_idx]
                # cudaMMC LooperSolver.cpp:3053: clusters[children[j]].pos = pos
                ch.set_pos(*child_positions[ci])

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
