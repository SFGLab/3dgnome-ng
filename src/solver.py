"""
src/solver.py  —  High-level LooperSolver analog for 3dgnome-torch.

Orchestrates data loading, hierarchy building, and MC reconstruction.
Mirrors C++ LooperSolver methods:
  - setContactData()              → set_contact_data()
  - reconstructClustersHeatmap()  → reconstruct_heatmap()
  - reconstructClustersArcsDistances() → reconstruct_arcs()
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np

from .settings import Settings
from .io import (
    BedRegion,
    load_anchors,
    load_arcs,
    mark_arcs,
    remove_empty_anchors,
    load_breakpoints,
    create_singleton_heatmap,
)
from .hierarchy import (
    Cluster, LVL_ANCHOR, LVL_SEGMENT, LVL_INTERACTION_BLOCK, LVL_CHROMOSOME,
    build_cluster_tree, set_level, set_top_level,
)
from .mc import mc_heatmap, mc_arcs
from .energy import random_vector_np


class Solver:

    def __init__(self, settings: Settings):
        self.s = settings
        self.clusters: list[Cluster] = []
        self.chr_root: dict[str, int] = {}
        self.chr_first_cluster: dict[str, int] = {}
        self.chrs: list[str] = []
        self.current_chr: str = ""

        # Arc data (after mark_arcs / remove_empty_anchors)
        self.anchors: dict = {}   # chr → list[Anchor]
        self.arcs: dict = {}      # chr → list[InteractionArc] (global indices)

        # Heatmap structures
        self.heatmap_dist: np.ndarray | None = None   # (N, N) expected distances
        self.heatmap_dist_diag: int = 0

        # Anchor-level expected distances (per IB)
        self.exp_dist_anchor: np.ndarray | None = None  # (n_active, n_active)

        self.selected_region: BedRegion | None = None

    # -----------------------------------------------------------------------
    # Data loading and hierarchy construction

    def set_contact_data(
        self,
        chrs_list: list,
        region: BedRegion | None,
        data_dir: str,
    ) -> None:
        """
        Load anchors + arcs, build cluster hierarchy.
        Mirrors C++ LooperSolver::setContactData().
        """
        s = self.s
        self.chrs = chrs_list
        self.selected_region = region
        chr_set = set(chrs_list)

        anchor_path = s.data_path(s.data_anchors)
        arc_path = s.data_path(s.data_pet_clusters)
        seg_split_path = s.data_path(s.data_segment_split)

        print("[solver] load anchors")
        self.anchors = load_anchors(anchor_path, chr_set, region)

        print("[solver] load arcs")
        raw_arcs = load_arcs(arc_path, chr_set, region, s.max_pet_length)

        print("[solver] mark arcs")
        marked = mark_arcs(self.anchors, raw_arcs)

        print("[solver] remove empty anchors")
        self.anchors = remove_empty_anchors(self.anchors, marked)
        self.arcs = marked

        print("[solver] load breakpoints")
        breakpoints = load_breakpoints(seg_split_path, chrs_list)

        print("[solver] build cluster hierarchy")
        self.clusters, self.chr_root, self.chr_first_cluster = build_cluster_tree(
            self.anchors, self.arcs, breakpoints, chrs_list
        )
        print(f"  total clusters: {len(self.clusters)}")

    # -----------------------------------------------------------------------
    # Segment-level reconstruction (heatmap MC)

    def reconstruct_heatmap(self) -> None:
        """
        Position beads at segment level using singleton heatmap MC.
        Mirrors C++ LooperSolver::reconstructClustersHeatmap().
        """
        s = self.s

        # setLevel(LVL_SEGMENT) → current_level contains segment cluster indices
        current_level = set_level(
            LVL_SEGMENT - LVL_CHROMOSOME,  # steps down from root
            self.chr_root, self.clusters, self.chrs
        )

        # Check if only 1 segment total across all chromosomes
        total_segs = sum(len(v) for v in current_level.values())
        single_seg = (len(self.chrs) == 1 and total_segs <= 1)

        if single_seg:
            print("[solver] single segment → place at origin")
            chr_ = self.chrs[0]
            if current_level[chr_]:
                self.clusters[current_level[chr_][0]].pos = np.zeros(3, dtype=np.float32)
                # Set child IB and anchor positions too
                self._interpolate_children_linear(current_level[chr_])
            return

        # Multiple segments: build singleton heatmap
        print("\n[solver] segment level")
        self._reconstruct_heatmap_single_level(current_level)

    def _compute_segment_bins(self, current_level: dict) -> tuple:
        """
        Compute heatmap bin boundaries for segment-level clusters.
        Mirrors bin calculation in createSingletonHeatmap().
        Returns (bins, start_ind, total_size).
        """
        bins = {}
        start_ind = {}
        curr_idx = 0

        for chr_ in self.chrs:
            segs = current_level.get(chr_, [])
            breaks = [0]
            for i in range(len(segs) - 1):
                pos = (self.clusters[segs[i]].end + self.clusters[segs[i + 1]].start) // 2
                breaks.append(pos)
            breaks.append(int(1e9))
            bins[chr_] = breaks
            start_ind[chr_] = curr_idx
            curr_idx += len(breaks) - 1

        return bins, start_ind, curr_idx

    def _reconstruct_heatmap_single_level(self, current_level: dict) -> None:
        """
        Reconstruct segment-level positions using singleton heatmap MC.
        Mirrors C++ reconstructClustersHeatmapSingleLevel(1) (segment level).
        """
        s = self.s
        bins, start_ind, total_size = self._compute_segment_bins(current_level)

        # Create singleton heatmap from file
        print("[solver] create segment heatmap")
        singleton_path = s.data_path(s.data_singletons)
        h_raw = create_singleton_heatmap(
            singleton_path, bins, start_ind, total_size,
            set(self.chrs), self.selected_region
        )

        # Normalize heatmap rows to equal expected sum
        h_norm = self._normalize_heatmap(h_raw, total_size)

        # Normalize diagonal total to 1.0
        self._normalize_heatmap_diagonal_total(h_norm, total_size, 1.0)

        # Scale inter-chr contacts (no-op for single chr)
        if len(self.chrs) > 1:
            self._normalize_heatmap_inter(h_norm, total_size, current_level, s.heatmap_inter_scaling)

        # Convert freq → distance heatmap
        heatmap_dist, avg_dist = self._create_distance_heatmap(
            h_norm, total_size, inter=False
        )

        self.heatmap_dist = np.array(heatmap_dist)
        self.heatmap_dist_diag = self._get_diagonal_size(h_norm, total_size)

        # Place initial positions: parent IB position for all segments in chr
        for chr_ in self.chrs:
            segs = current_level.get(chr_, [])
            if not segs:
                continue
            # Get the segment's parent (IB) position
            par = self.clusters[segs[0]].parent
            if par >= 0:
                origin = self.clusters[par].pos.copy()
            else:
                origin = np.zeros(3, dtype=np.float32)
            for seg_idx in segs:
                self.clusters[seg_idx].pos = origin.copy()

        # Concatenate all segment indices into active_region
        active_region = []
        for chr_ in self.chrs:
            active_region.extend(current_level.get(chr_, []))

        if len(active_region) <= 1:
            return

        # Compute step size
        step_size = avg_dist * s.noise_lvl2

        # Build position array for active beads
        pos = np.array([self.clusters[i].pos for i in active_region], dtype=np.float32)

        n = len(active_region)
        best_score = -1.0
        best_pos = pos.copy()

        for run in range(s.steps_lvl2):
            print(f"[solver] MC heatmap, segment level, run {run + 1}/{s.steps_lvl2}")
            # Randomise initial positions
            for i in range(n):
                pos[i] = self.clusters[active_region[i]].pos + random_vector_np(step_size)

            score = mc_heatmap(pos, self.heatmap_dist, self.heatmap_dist_diag,
                               step_size, s)
            print(f"  score = {score:.6f}, best = {best_score:.6f}")

            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()

        # Restore best
        for i, idx in enumerate(active_region):
            self.clusters[idx].pos = best_pos[i].copy()

        # Interpolate IB and anchor positions from segment positions
        for chr_ in self.chrs:
            segs = current_level.get(chr_, [])
            if segs:
                self._interpolate_children_linear(segs)

    def _interpolate_children_linear(self, parent_indices: list) -> None:
        """
        Set child cluster positions by linear interpolation between parents.
        Used for IBs between segments, and anchors within IBs.
        Simplified version of C++ interpolateChildrenPositionSpline().
        """
        clusters = self.clusters
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
                    self._interpolate_children_linear([child_idx])
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
                    self._interpolate_children_linear([child_idx])

    # -----------------------------------------------------------------------
    # Anchor-level reconstruction (arc spring MC)

    def reconstruct_arcs(self) -> None:
        """
        Position anchor beads using arc spring MC.
        Mirrors C++ LooperSolver::reconstructClustersArcsDistances().
        """
        s = self.s

        # Set segment level
        seg_level = set_level(
            LVL_SEGMENT - LVL_CHROMOSOME,
            self.chr_root, self.clusters, self.chrs
        )

        for chr_ in self.chrs:
            self.current_chr = chr_
            segs = seg_level.get(chr_, [])
            if not segs:
                continue

            print(f"\n[solver] anchor level: {chr_}")

            # positionInteractionBlocks: set initial IB positions from segment positions
            self._position_interaction_blocks(segs)

            # Get IB-level cluster indices
            ib_level = {}
            for seg_idx in segs:
                ib_level.setdefault(chr_, []).extend(self.clusters[seg_idx].children)

            ibs = ib_level.get(chr_, [])
            n_ibs = len(ibs)

            for ib_i, ib_idx in enumerate(ibs):
                print(f"  {chr_} IB {ib_i + 1}/{n_ibs}")

                ib = self.clusters[ib_idx]
                active_region = list(ib.children)

                if len(active_region) <= 1:
                    continue

                # Place all anchors at IB position initially
                for a_idx in active_region:
                    self.clusters[a_idx].pos = ib.pos.copy()

                # Compute expected distances from arcs
                self._calc_anchor_expected_distances(active_region)

                # Reconstruct anchor positions
                self._reconstruct_cluster_arcs(ib_idx, active_region)

    def _position_interaction_blocks(self, segs: list) -> None:
        """
        Position IB clusters between segment positions.
        Mirrors C++ positionInteractionBlocks().
        """
        if len(segs) > 1:
            self._interpolate_children_linear(segs)
        else:
            # Random walk
            seg = self.clusters[segs[0]]
            pos = np.zeros(3, dtype=np.float32)
            for ib_idx in seg.children:
                pos = pos + random_vector_np(100.0)
                self.clusters[ib_idx].pos = pos.copy()

    def _calc_anchor_expected_distances(self, active_region: list) -> None:
        """
        Build expected distance matrix for anchor-level active region.
        Mirrors C++ calcAnchorExpectedDistancesHeatmap().

        exp_dist_anchor[i][j]:
          -1   → pair with no arc (repulsion)
           0   → diagonal (self)
          >0   → arc expected distance from freqToDistance(score)
        """
        n = len(active_region)
        mat = np.full((n, n), -1.0, dtype=np.float64)
        np.fill_diagonal(mat, 0.0)

        # Map global cluster index → active index
        cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}

        chr_ = self.current_chr
        chr_arcs = self.arcs.get(chr_, [])

        for ai, ci in enumerate(active_region):
            for arc_local in self.clusters[ci].arcs:
                if arc_local >= len(chr_arcs):
                    continue
                arc = chr_arcs[arc_local]
                other = arc.end if arc.start == ci else arc.start

                if other < ci:
                    continue  # only forward

                if other not in cluster_to_active:
                    continue

                bi = cluster_to_active[other]
                freq = arc.score
                exp_d = self.s.freq_to_distance(freq)

                mat[ai, bi] = exp_d
                mat[bi, ai] = exp_d

        self.exp_dist_anchor = mat

    def _reconstruct_cluster_arcs(
        self,
        ib_idx: int,
        active_region: list,
    ) -> None:
        """
        MC reconstruction for one interaction block (anchor level).
        Mirrors C++ reconstructClusterArcsDistances().
        """
        s = self.s
        active_size = len(active_region)

        # Compute noise size (avg expected distance between consecutive anchors * noise_arcs)
        # C++ uses hardcoded noise_size_small = 0.005 for anchor level
        noise_size_small = 0.005

        # Compute dist_to_next for each anchor
        for i in range(active_size - 1):
            d = abs(self.clusters[active_region[i + 1]].genomic_pos
                    - self.clusters[active_region[i]].genomic_pos)
            self.clusters[active_region[i]].dist_to_next = s.genomic_length_to_distance(d)

        # Store initial positions
        initial_pos = np.array([self.clusters[i].pos for i in active_region], dtype=np.float32)

        best_score = -1.0
        best_pos = initial_pos.copy()

        for run in range(s.steps_arcs):
            # Random initial displacement for anchors
            pos = initial_pos.copy()
            for i in range(active_size):
                pos[i] += random_vector_np(noise_size_small)

            score = mc_arcs(pos, self.exp_dist_anchor, noise_size_small, s)

            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()

        # Restore best anchor positions
        for i, ci in enumerate(active_region):
            self.clusters[ci].pos = best_pos[i].copy()

    # -----------------------------------------------------------------------
    # Heatmap normalisation helpers

    @staticmethod
    def _get_diagonal_size(h: list, n: int) -> int:
        """Find smallest w such that any cell at distance w from diagonal is non-zero."""
        for w in range(n):
            for i in range(n - w):
                if h[i][i + w] > 1e-6:
                    return w
        return 0

    @staticmethod
    def _normalize_heatmap(h: list, n: int) -> list:
        """
        Row-normalize: scale each row so all rows have equal sum (avg).
        Then symmetrize: h[i][j] = (h[i][j] + h[j][i]) / 2.
        Mirrors C++ LooperSolver::normalizeHeatmap().
        """
        row_sums = [sum(h[i]) for i in range(n)]
        total = sum(row_sums)
        if total < 1e-10:
            return h
        expected = total / n

        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            mn = expected / row_sums[i] if row_sums[i] > 1e-10 else 1.0
            for j in range(n):
                out[i][j] = h[i][j] * mn

        # Symmetrize
        for i in range(n):
            for j in range(i + 1, n):
                avg = (out[i][j] + out[j][i]) / 2.0
                out[i][j] = avg
                out[j][i] = avg

        return out

    @staticmethod
    def _normalize_heatmap_diagonal_total(h: list, n: int, val: float) -> None:
        """
        Normalize so the average of the first non-zero diagonal equals val.
        Mirrors C++ normalizeHeatmapDiagonalTotal().
        Modifies h in place.
        """
        # Find diagonal size
        diag = 0
        for w in range(n):
            found = False
            for i in range(n - w):
                if h[i][i + w] > 1e-6:
                    found = True
                    break
            if found:
                diag = w
                break

        # Average of that diagonal
        count = n - diag
        if count <= 0:
            return
        avg = sum(h[i][i + diag] for i in range(count)) / count
        if avg < 1e-10:
            return

        mn = val / avg
        for i in range(n):
            for j in range(n):
                h[i][j] *= mn

    @staticmethod
    def _normalize_heatmap_inter(
        h: list,
        n: int,
        current_level: dict,
        scale: float,
    ) -> None:
        """
        Scale inter-chromosomal entries.
        Mirrors C++ normalizeHeatmapInter().
        Modifies h in place.
        """
        # This is only relevant for multi-chromosome runs; skip for now.
        pass

    def _create_distance_heatmap(
        self,
        h: list,
        n: int,
        inter: bool = False,
    ) -> tuple:
        """
        Convert normalized contact frequency heatmap to expected distance heatmap.
        Mirrors C++ createDistanceHeatmap().

        Returns (dist_heatmap, avg_dist) where dist_heatmap is a 2D list.
        Entries within diagonal_size are set to -1 (ignored in scoring).
        """
        s = self.s
        diag = self._get_diagonal_size(h, n)

        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                val = h[i][j]
                if val < 1e-6:
                    dist[i][j] = 0.0
                elif abs(i - j) < diag:
                    dist[i][j] = -1.0
                else:
                    if inter:
                        dist[i][j] = s.freq_to_dist_heatmap_inter(val)
                    else:
                        dist[i][j] = s.freq_to_dist_heatmap(val)
                dist[j][i] = dist[i][j]

        # Clip large distances to avg * stretching
        vals = [dist[i][j] for i in range(n) for j in range(n)
                if dist[i][j] > 0]
        avg = sum(vals) / len(vals) if vals else 1.0
        max_d = avg * s.heatmap_distance_stretching

        for i in range(n):
            for j in range(n):
                if dist[i][j] > max_d:
                    dist[i][j] = max_d

        return dist, avg

    # -----------------------------------------------------------------------
    # Output helpers

    def get_leaf_positions(self, chr_: str) -> list:
        """
        Return anchor-level bead positions for chr_ as list of (midpoint_bp, x, y, z),
        sorted by genomic midpoint.
        """
        result = []
        first = self.chr_first_cluster.get(chr_, -1)
        if first < 0:
            return result
        for i in range(first, len(self.clusters)):
            c = self.clusters[i]
            if c.level != LVL_ANCHOR:
                break
            mid = c.genomic_pos
            x, y, z = float(c.pos[0]), float(c.pos[1]), float(c.pos[2])
            result.append((mid, x, y, z))
        result.sort(key=lambda b: b[0])
        return result

    def get_anchor_positions(self) -> list:
        """All anchor beads from all chromosomes, sorted by chr then genomic position."""
        result = []
        for chr_ in self.chrs:
            first = self.chr_first_cluster.get(chr_, -1)
            if first < 0:
                continue
            for i in range(first, len(self.clusters)):
                c = self.clusters[i]
                if c.level != LVL_ANCHOR:
                    break
                mid = c.genomic_pos
                x, y, z = float(c.pos[0]), float(c.pos[1]), float(c.pos[2])
                result.append((mid, x, y, z))
        return sorted(result, key=lambda b: b[0])
