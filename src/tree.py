"""
Build and manage the hierarchical bead-and-spring model.

Levels mirror cudaMMC createTreeChromosome:
    1 = chromosome root
    2 = segment        (~2 Mb)
    3 = interaction block (IB)
    4 = anchor (leaf / CTCF site)

Sub-anchor linker beads are inserted between anchors during reconstruction.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import torch

from .data_structures import Anchor, Cluster, InteractionArc
from .distances import genomic_length_to_distance
from .settings import Settings


# ── Gap / split detection ────────────────────────────────────────────────────

def find_gaps(anchors: List[Anchor],
              arcs: List[InteractionArc],
              segment_size: int = 2_000_000) -> List[int]:
    """
    Return sorted list of anchor indices that start a new segment.
    A new segment begins when:
      - The gap between anchor[i-1].end and anchor[i].start exceeds segment_size, OR
      - There are no arcs bridging anchor[i-1] and anchor[i].
    Index 0 is always a segment start.
    """
    N = len(anchors)
    if N == 0:
        return []

    # build set of all pairs connected by arcs
    arc_pairs = set()
    for a in arcs:
        arc_pairs.add((min(a.start, a.end), max(a.start, a.end)))

    gaps = [0]
    for i in range(1, N):
        gap_bp = anchors[i].start - anchors[i - 1].end
        if gap_bp > segment_size:
            gaps.append(i)
            continue
        # check if any arc bridges across position i-1 → i
        bridged = any(p[0] < i <= p[1] for p in arc_pairs)
        if not bridged:
            gaps.append(i)

    return gaps


def find_ibs(anchors: List[Anchor],
             arcs: List[InteractionArc],
             seg_start: int, seg_end: int) -> List[int]:
    """
    Within a segment [seg_start, seg_end) return IB boundary anchor indices.
    An IB boundary is defined by an arc whose start < boundary <= end within
    the segment.  Index seg_start is always a boundary.
    """
    if seg_end <= seg_start:
        return [seg_start]

    boundaries = {seg_start}
    for arc in arcs:
        lo = min(arc.start, arc.end)
        hi = max(arc.start, arc.end)
        if lo >= seg_start and hi < seg_end:
            boundaries.add(lo)
            # IB split at the arc endpoint
            if hi + 1 < seg_end:
                boundaries.add(hi + 1)

    return sorted(boundaries)


# ── Catmull-Rom spline interpolation ─────────────────────────────────────────

def catmull_rom(t: float, p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> Tuple:
    """Catmull-Rom spline: returns interpolated (x, y, z) at parameter t ∈ [0,1]."""
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
    """
    M = len(parent_positions)
    if M < 2:
        return [parent_positions[0]] * n_children if M == 1 else []
    if n_children == 0:
        return []

    # Catmull-Rom needs ghost points at each end
    def ghost_start(p0, p1):
        return (2 * p0[0] - p1[0], 2 * p0[1] - p1[1], 2 * p0[2] - p1[2])

    def ghost_end(pm2, pm1):
        return (2 * pm1[0] - pm2[0], 2 * pm1[1] - pm2[1], 2 * pm1[2] - pm2[2])

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
        result.append(catmull_rom(t, p1, p2, p3, p4))

    return result


# ── Tree builder ──────────────────────────────────────────────────────────────

class ChromosomeTree:
    """
    Complete hierarchical model for one chromosome.

    clusters: flat list of Cluster objects.
    anchors_idx: [cluster index for each anchor (level-4 bead)]
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
        idx = len(self.clusters)
        gpos = (start + end) // 2
        c = Cluster(self.chrom, start, end, gpos, level=level, parent=parent)
        c.base_start = start
        c.base_end = end
        self.clusters.append(c)
        return idx

    def _build(self, anchors: List[Anchor], arcs: List[InteractionArc]):
        N = len(anchors)
        if N == 0:
            return

        ss = self.settings

        # level-1: chromosome root
        root_idx = self._new_cluster(1, anchors[0].start, anchors[-1].end)
        root = self.clusters[root_idx]

        # find segment boundaries
        seg_starts = find_gaps(anchors, arcs, ss.segment_size)
        seg_starts.append(N)  # sentinel

        for s_i, seg_s in enumerate(seg_starts[:-1]):
            seg_e = seg_starts[s_i + 1]  # exclusive

            # level-2: segment
            seg_idx = self._new_cluster(2,
                                         anchors[seg_s].start,
                                         anchors[seg_e - 1].end,
                                         root_idx)
            root.children.append(seg_idx)

            # find IB boundaries within this segment
            ib_starts = find_ibs(anchors, arcs, seg_s, seg_e)
            ib_starts.append(seg_e)  # sentinel

            for ib_i, ib_s in enumerate(ib_starts[:-1]):
                ib_e = ib_starts[ib_i + 1]  # exclusive

                # level-3: interaction block
                ib_idx = self._new_cluster(3,
                                           anchors[ib_s].start,
                                           anchors[ib_e - 1].end,
                                           seg_idx)
                self.clusters[seg_idx].children.append(ib_idx)

                # level-4: anchors (leaves)
                for ai in range(ib_s, ib_e):
                    a = anchors[ai]
                    anc_idx = self._new_cluster(4, a.start, a.end, ib_idx)
                    anc_c = self.clusters[anc_idx]
                    anc_c.orientation = a.orientation
                    anc_c.genomic_pos = a.mid
                    self.clusters[ib_idx].children.append(anc_idx)
                    self.anchors_idx.append(anc_idx)

        # wire arcs onto anchor clusters
        for arc_i, arc in enumerate(arcs):
            if arc.start < len(self.anchors_idx):
                self.clusters[self.anchors_idx[arc.start]].arcs.append(arc_i)
            if arc.end < len(self.anchors_idx):
                self.clusters[self.anchors_idx[arc.end]].arcs.append(arc_i)

    # ── Position initialisation ───────────────────────────────────────────────

    def init_positions_random(self, radius: float = 10.0, seed: int = 42):
        """Place all clusters randomly inside a sphere of given radius."""
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
        # process bottom-up by level
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
        """
        # collect parents at the given level, in genomic order
        parents = sorted([c for c in self.clusters if c.level == parent_level],
                         key=lambda c: c.genomic_pos)
        for p in parents:
            if not p.children:
                continue
            # siblings (same parent-of-parent) as control points
            siblings = (self.clusters[p.parent].children
                        if p.parent >= 0 else [p])
            ctrl_pts = [self.clusters[s].pos for s in siblings]
            child_positions = interpolate_children_spline(ctrl_pts, len(p.children))
            for ci, ch_idx in enumerate(p.children):
                ch = self.clusters[ch_idx]
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
        """Expected linker lengths between consecutive anchors, shape (N-1,)."""
        from .distances import genomic_length_to_distance as g2d
        s = self.settings
        n = len(self.anchors_idx)
        lengths = []
        for i in range(n - 1):
            c1 = self.clusters[self.anchors_idx[i]]
            c2 = self.clusters[self.anchors_idx[i + 1]]
            gap = max(0, c2.start - c1.end)
            lengths.append(g2d(gap, s.genomic_dist_scale,
                               s.genomic_dist_power, s.genomic_dist_base))
        return torch.tensor(lengths, dtype=torch.float32, device=device)
