"""
High-level LooperSolver analog for 3dgnome-ng.

Orchestrates data loading, hierarchy building, and MC reconstruction.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from .data import ContactData
from .util import random_vector_np
from .hierarchy import (
    Cluster, LVL_ANCHOR, LVL_SEGMENT, LVL_CHROMOSOME,
    build_cluster_tree, set_level,
)
from .io import create_singleton_heatmap
from .mc import mc_heatmap, mc_arcs, mc_smooth
from .settings import Settings
from .types import *

# Anchor map entry produced by densification: (bead_index, cluster_index).
AnchorMapEntry = tuple[int, ClusterIndex]


class Solver:

    def __init__(self, settings: Settings) -> None:
        self.s: Settings = settings
        self.clusters: list[Cluster] = []
        self.chr_root: ChrRootMap = {}
        self.chr_first_cluster: ChrFirstClusterMap = {}
        self.chrs: list[str] = []

        # Arc data (after mark_arcs / remove_empty_anchors)
        self.anchors: AnchorMap = {}
        self.arcs: ArcMap = {}

        # Heatmap structures
        self.heatmap_dist: F64Array | None = None  # (N, N) expected distances
        self.heatmap_dist_diag: int = 0

        self.selected_region: BedRegion | None = None
        self.dense_active_regions: dict[str, list[BeadOut]] = {}
        self._singletons: list[SingletonContact] = []

    # Data loading and hierarchy construction

    def load(
        self,
        data: ContactData,
        chrs_list: list[str],
        region: BedRegion | None = None,
    ) -> None:
        """
        Accept pre-loaded contact data and build the cluster hierarchy.
        Mirrors Reference LooperSolver::setContactData().
        """
        self.chrs = chrs_list
        self.selected_region = region
        self.anchors = data.anchors
        self.arcs = data.arcs
        self._singletons = data.singletons

        print("[solver] build cluster hierarchy")
        self.clusters, self.chr_root, self.chr_first_cluster = build_cluster_tree(
            self.anchors, self.arcs, data.breakpoints, chrs_list
        )
        print(f"  total clusters: {len(self.clusters)}")

    # Segment-level reconstruction (heatmap MC)

    def reconstruct_heatmap(self) -> None:
        """
        Position beads at segment level using singleton heatmap MC.
        Mirrors Reference LooperSolver::reconstructClustersHeatmap().
        """
        # setLevel(LVL_SEGMENT) -> current_level contains segment cluster indices
        current_level = set_level(
            LVL_SEGMENT - LVL_CHROMOSOME,  # steps down from root
            self.chr_root, self.clusters, self.chrs
        )

        # Check if only 1 segment total across all chromosomes
        total_segs = sum(len(v) for v in current_level.values())
        single_seg = (len(self.chrs) == 1 and total_segs <= 1)

        if single_seg:
            if self.s.output_level >= 1:
                print("[solver] single segment -> place at origin")
            chr_ = self.chrs[0]
            if current_level[chr_]:
                self.clusters[current_level[chr_][0]].pos = np.zeros(3, dtype=np.float32)
                self._interpolate_children_linear(current_level[chr_])
            return

        if self.s.output_level >= 1:
            print("\n[solver] segment level")
        self._reconstruct_heatmap_single_level(current_level)

    def _compute_segment_bins(
        self, current_level: ChrLevel
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
        bins: dict[str, list[int]] = {}
        start_ind: dict[str, int] = {}
        curr_idx = 0
        bin_lengths_mb: list[float] = []

        for chr_ in self.chrs:
            segs = current_level.get(chr_, [])
            breaks = [0]
            for i in range(len(segs) - 1):
                pos = (self.clusters[segs[i]].end + self.clusters[segs[i + 1]].start) // 2
                breaks.append(pos)
            breaks.append(int(1e9))
            bins[chr_] = breaks
            start_ind[chr_] = curr_idx

            n = len(segs)
            for i in range(n):
                if n == 1:
                    bp = self.clusters[segs[0]].end - self.clusters[segs[0]].start
                elif i == 0:
                    bp = breaks[1] - self.clusters[segs[0]].start
                elif i == n - 1:
                    bp = self.clusters[segs[-1]].end - breaks[-2]
                else:
                    bp = breaks[i + 1] - breaks[i]
                bin_lengths_mb.append(max(bp, 1) / 1e6)

            curr_idx += len(breaks) - 1

        return bins, start_ind, curr_idx, bin_lengths_mb

    def _reconstruct_heatmap_single_level(self, current_level: ChrLevel) -> None:
        """
        Reconstruct segment-level positions using singleton heatmap MC.
        Mirrors Reference reconstructClustersHeatmapSingleLevel(1) (segment level).
        """
        s = self.s
        bins, start_ind, total_size, bin_lengths_mb = self._compute_segment_bins(current_level)

        if self.s.output_level >= 1:
            print("[solver] create segment heatmap")
        h_raw = create_singleton_heatmap(
            self._singletons, bins, start_ind, total_size, bin_lengths_mb=bin_lengths_mb
        )

        # Normalize heatmap rows to equal expected sum
        h_norm = self._normalize_heatmap(h_raw, total_size)

        # Normalize diagonal total to 1.0
        self._normalize_heatmap_diagonal_total(h_norm, total_size, 1.0)

        # Scale inter-chr contacts (no-op for single chr)
        if len(self.chrs) > 1:
            self._normalize_heatmap_inter(h_norm, total_size, current_level, s.heatmap_inter_scaling)

        # Convert freq -> distance heatmap
        heatmap_dist, avg_dist = self._create_distance_heatmap(
            h_norm, total_size, inter=False
        )

        self.heatmap_dist = np.array(heatmap_dist, dtype=np.float64)
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
        active_region: list[int] = []
        for chr_ in self.chrs:
            active_region.extend(current_level.get(chr_, []))

        if len(active_region) <= 1:
            return

        # Compute step size
        step_size = avg_dist * s.noise_lvl2

        # Build position array for active beads
        pos: F32Array = np.array(
            [self.clusters[i].pos for i in active_region], dtype=np.float32
        )

        n = len(active_region)
        best_score = -1.0
        best_pos: F32Array = pos.copy()

        log1 = s.output_level >= 1
        log2 = s.output_level >= 2
        assert self.heatmap_dist is not None
        for run in range(s.steps_lvl2):
            if log1:
                print(f"[solver] heatmap run {run + 1}/{s.steps_lvl2}  ({n} beads)")
            for i in range(n):
                pos[i] = self.clusters[active_region[i]].pos + random_vector_np(step_size)

            score = mc_heatmap(pos, self.heatmap_dist, self.heatmap_dist_diag,
                               step_size, s, label=f"heatmap run {run + 1}",
                               verbose=log2)

            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()
            if log1:
                print(f"  -> score={score:.6f}  best={best_score:.6f}")

        # Restore best
        for i, idx in enumerate(active_region):
            self.clusters[idx].pos = best_pos[i].copy()

        # Interpolate IB and anchor positions from segment positions
        for chr_ in self.chrs:
            segs = current_level.get(chr_, [])
            if segs:
                self._interpolate_children_linear(segs)

    def _interpolate_children_linear(self, parent_indices: list[int]) -> None:
        """
        Set child cluster positions by linear interpolation between parents.
        Used for IBs between segments, and anchors within IBs.
        Simplified version of Reference interpolateChildrenPositionSpline().
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

    # Anchor-level reconstruction (arc spring MC)

    def reconstruct_arcs(self) -> None:
        """
        Position anchor beads using arc spring MC.
        Mirrors Reference LooperSolver::reconstructClustersArcsDistances().
        """
        self.dense_active_regions = {}

        seg_level = set_level(
            LVL_SEGMENT - LVL_CHROMOSOME,
            self.chr_root, self.clusters, self.chrs
        )

        for chr_ in self.chrs:
            segs = seg_level.get(chr_, [])
            if not segs:
                continue

            if self.s.output_level >= 1:
                print(f"\n[solver] anchor level: {chr_}")
            self._position_interaction_blocks(segs)

            ibs: list[int] = []
            for seg_idx in segs:
                ibs.extend(self.clusters[seg_idx].children)
            n_ibs = len(ibs)

            work: list[tuple[int, int, str, list[int]]] = []
            for ib_i, ib_idx in enumerate(ibs):
                ib = self.clusters[ib_idx]
                active_region = list(ib.children)
                ib_label = f"{chr_} IB {ib_i + 1}/{n_ibs}"
                if len(active_region) <= 1:
                    if self.s.output_level >= 1:
                        print(f"  {ib_label}  ({len(active_region)} anchors - skip)")
                    continue
                for a_idx in active_region:
                    self.clusters[a_idx].pos = ib.pos.copy()
                work.append((ib_i, ib_idx, ib_label, active_region))

            for ib_i, ib_idx, ib_label, active_region in work:
                beads = self._process_ib(ib_idx, ib_label, active_region, chr_)
                self.dense_active_regions.setdefault(chr_, []).extend(beads)

    def _position_interaction_blocks(self, segs: list[int]) -> None:
        """
        Position IB clusters between segment positions.
        Mirrors Reference positionInteractionBlocks().
        """
        if len(segs) > 1:
            self._interpolate_children_linear(segs)
        else:
            # Random walk
            seg = self.clusters[segs[0]]
            pos: F32Array = np.zeros(3, dtype=np.float32)
            for ib_idx in seg.children:
                pos = pos + random_vector_np(100.0)
                self.clusters[ib_idx].pos = pos.copy()

    def _process_ib(
        self,
        ib_idx: int,
        ib_label: str,
        active_region: list[int],
        chr_: str,
    ) -> list[BeadOut]:
        """
        All work for one IB: arc MC + smooth MC.  Safe to call from a thread
        because each IB owns a disjoint subset of cluster indices.

        If small_ib_boost is enabled and this IB is below threshold, spring
        constants are multiplied for the duration of this IB only — a local
        settings copy is used so we never mutate self.s (thread-safe).
        """
        log1 = self.s.output_level >= 1
        log2 = self.s.output_level >= 2
        if log1:
            print(f"\n[solver] {ib_label}  ({len(active_region)} anchors)")

        s_ib = self._settings_for_ib(active_region, ib_label, log1)

        # Build singleton contact heatmaps if either feature is enabled.
        # Both heatmaps are derived from a single singleton-binning pass so
        # we do it here before any MC and pass results down.
        anchor_heat: F64Array | None = None
        subanchor_heat_raw: F64Array | None = None
        if (self.s.use_anchor_heatmap or self.s.use_subanchor_heatmap) and self._singletons:
            anchor_heat, subanchor_heat_raw = self._build_contact_heatmaps(active_region, chr_)

        exp_dist = self._calc_anchor_expected_distances(active_region, chr_, anchor_heat)
        self._reconstruct_cluster_arcs(ib_idx, active_region, exp_dist, ib_label,
                                       log1=log1, log2=log2, s_override=s_ib)
        return self._reconstruct_cluster_smooth(active_region, chr_, ib_label,
                                                subanchor_heat_raw=subanchor_heat_raw,
                                                log1=log1, log2=log2, s_override=s_ib)

    def _settings_for_ib(
        self, active_region: list[int], ib_label: str, log1: bool
    ) -> Settings:
        """Return self.s or a boosted copy if this IB qualifies as 'small'."""
        if not self.s.use_small_ib_boost:
            return self.s
        threshold = self.s.small_ib_threshold
        if len(active_region) >= threshold:
            return self.s
        m = self.s.small_ib_spring_multiplier
        if m == 1.0:
            return self.s
        s_boost: Settings = copy.copy(self.s)
        s_boost.spring_stretch_arcs = self.s.spring_stretch_arcs * m
        s_boost.spring_squeeze_arcs = self.s.spring_squeeze_arcs * m
        s_boost.spring_stretch = self.s.spring_stretch * m
        s_boost.spring_squeeze = self.s.spring_squeeze * m
        s_boost.spring_angular = self.s.spring_angular * m
        if log1:
            print(f"  {ib_label}  small-IB spring boost ×{m}")
        return s_boost

    def _calc_anchor_expected_distances(
        self,
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
        n = len(active_region)
        mat: F64Array = np.full((n, n), -1.0, dtype=np.float64)
        np.fill_diagonal(mat, 0.0)

        cluster_to_active = {ci: ai for ai, ci in enumerate(active_region)}
        chr_arcs = self.arcs.get(chr_, [])

        for ai, ci in enumerate(active_region):
            for arc_local in self.clusters[ci].arcs:
                if arc_local >= len(chr_arcs):
                    continue
                arc = chr_arcs[arc_local]
                other = arc.end if arc.start == ci else arc.start

                if other < ci or other not in cluster_to_active:
                    continue

                bi = cluster_to_active[other]
                exp_d = self.s.freq_to_distance(arc.score)
                mat[ai, bi] = exp_d
                mat[bi, ai] = exp_d

        # Apply anchor heatmap: scale down expected distances for high-contact pairs.
        # Mirrors Reference post-processing in calcAnchorExpectedDistancesHeatmap().
        if anchor_heatmap is not None and self.s.use_anchor_heatmap:
            max_val = float(anchor_heatmap.max())
            influence = float(self.s.anchor_heatmap_influence)
            if max_val > 1e-6:
                for i in range(n):
                    for j in range(i + 1, n):
                        if mat[i, j] <= 0.0:
                            continue
                        s_val = (anchor_heatmap[i, j] / max_val) * influence
                        if s_val > 1.0:
                            s_val = 1.0
                        mat[i, j] *= (1.0 - s_val)
                        mat[j, i] = mat[i, j]

        return mat

    def _reconstruct_cluster_arcs(
        self,
        ib_idx: int,
        active_region: list[int],
        exp_dist: F64Array,
        label: str = "",
        log1: bool = True,
        log2: bool = True,
        s_override: Settings | None = None,
    ) -> None:
        """
        MC reconstruction for one interaction block (anchor level).
        Mirrors Reference reconstructClusterArcsDistances().
        """
        s = s_override if s_override is not None else self.s
        active_size = len(active_region)

        # Compute noise size (avg expected distance between consecutive anchors * noise_arcs)
        # Reference uses hardcoded noise_size_small = 0.005 for anchor level
        noise_size_small = 0.005

        # Compute dist_to_next for each anchor
        for i in range(active_size - 1):
            d = abs(self.clusters[active_region[i + 1]].genomic_pos
                    - self.clusters[active_region[i]].genomic_pos)
            self.clusters[active_region[i]].dist_to_next = s.genomic_length_to_distance(d)

        # Store initial positions
        initial_pos: F32Array = np.array(
            [self.clusters[i].pos for i in active_region], dtype=np.float32
        )

        best_score = -1.0
        best_pos: F32Array = initial_pos.copy()

        for run in range(s.steps_arcs):
            run_label = f"{label} run {run + 1}/{s.steps_arcs}" if label else f"arcs run {run + 1}"
            if log1:
                print(f"  {run_label}")
            pos: F32Array = initial_pos.copy()
            for i in range(active_size):
                pos[i] += random_vector_np(noise_size_small)

            score = mc_arcs(pos, exp_dist, noise_size_small, s,
                            label=run_label, verbose=log2)

            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos.copy()

        # Restore best anchor positions
        for i, ci in enumerate(active_region):
            self.clusters[ci].pos = best_pos[i].copy()

    def _build_contact_heatmaps(
        self,
        active_region: list[int],
        chr_: str,
    ) -> tuple[F64Array, F64Array]:
        """
        Build anchor-level and subanchor-level singleton contact heatmaps.
        Mirrors Reference createSingletonSubanchorHeatmap().

        Returns (anchor_heatmap, subanchor_heatmap_raw) where:
          anchor_heatmap:      (n_anchors, n_anchors) float64 - normalized contact
                               density between anchor pairs; used for expected-distance
                               scaling in arc MC.
          subanchor_heatmap_raw: (N, N) float64 where N = n_anchors + (n_anchors-1)*ld
                               - normalized contact density at densified-bead resolution;
                               used for heat energy in smooth MC.
        """
        import bisect

        n_anchors = len(active_region)
        ld = self.s.loop_density

        # Total densified beads = n_anchors + (n_anchors-1)*ld = 1 + (n_anchors-1)*(ld+1)
        N = n_anchors + (n_anchors - 1) * ld

        # Build genomic break boundaries mirroring Reference createSingletonSubanchorHeatmap().
        # Anchor k occupies bin k*(ld+1).  Subanchor j in span k→k+1 occupies
        # bin k*(ld+1)+j  (j=1..ld).
        anchor_lens: list[int] = []
        gap_lens: list[int] = []

        region_start = self.clusters[active_region[0]].start
        region_end = self.clusters[active_region[-1]].end

        # breaks[i] is the left boundary of bin i
        breaks: list[int] = [region_start]
        anchor_lens.append(
            self.clusters[active_region[0]].end - self.clusters[active_region[0]].start
        )

        for i in range(1, n_anchors):
            ca_end = self.clusters[active_region[i - 1]].end
            cb_start = self.clusters[active_region[i]].start
            gap = max(cb_start - ca_end, 0)
            anchor_len = (self.clusters[active_region[i]].end
                          - self.clusters[active_region[i]].start)
            gap_lens.append(gap)
            anchor_lens.append(anchor_len)
            # ld+1 new break boundaries: span_start, ld-1 interior, span_end
            breaks.append(ca_end)
            for j in range(1, ld):
                breaks.append(ca_end + int(gap * j / ld))
            breaks.append(cb_start)

        breaks.append(region_end)
        # Number of bins = len(breaks)-1 = 1 + (n_anchors-1)*(ld+1) = N
        # Bin singleton contacts into subanchor heatmap.
        # Note: Python filters by chromosome (c1 != chr_ or c2 != chr_). Reference's
        # createSingletonSubanchorHeatmap does NOT filter by chromosome, so it
        # bins cross-chromosomal contacts whose midpoints fall in the region.
        # See [[project-singleton-chr-filter-divergence]] - this is intentional.
        h_sub: F64Array = np.zeros((N, N), dtype=np.float64)

        for c1, p1, c2, p2, sc in self._singletons:
            if c1 != chr_ or c2 != chr_:
                continue
            if p1 < region_start or p1 > region_end:
                continue
            if p2 < region_start or p2 > region_end:
                continue

            si = bisect.bisect_right(breaks, p1) - 1
            ei = bisect.bisect_right(breaks, p2) - 1

            if si < 0 or ei < 0 or si >= N or ei >= N or si == ei:
                continue

            h_sub[si, ei] += sc
            h_sub[ei, si] += sc

        # Extract anchor heatmap from raw subanchor values (BEFORE normalization),
        # normalized by anchor area in Mbp^2.  Mirrors Reference lines 1267-1273.
        h_anchor: F64Array = np.zeros((n_anchors, n_anchors), dtype=np.float64)
        for i in range(n_anchors):
            ai = i * (ld + 1)
            al_i = max(anchor_lens[i], 1)
            for j in range(i + 1, n_anchors):
                aj = j * (ld + 1)
                al_j = max(anchor_lens[j], 1)
                val = h_sub[ai, aj] / (al_i * al_j / 1e6)
                h_anchor[i, j] = val
                h_anchor[j, i] = val

        # Normalize subanchor heatmap: divide by avg count, then by bin areas.
        # Mirrors Reference lines 1294-1320.
        avg_count = float(h_sub.mean())
        if avg_count > 1e-6:
            h_sub /= avg_count

            # Bin sizes in kb: anchor bins use anchor_len, subanchor bins use gap/ld
            bin_sizes: F64Array = np.empty(N, dtype=np.float64)
            for k in range(N):
                anchor_idx = k // (ld + 1)
                if k % (ld + 1) == 0:
                    bin_sizes[k] = max(anchor_lens[anchor_idx], 1) / 1000.0
                else:
                    gap_idx = anchor_idx
                    gl = gap_lens[gap_idx] if gap_idx < len(gap_lens) else 1
                    bin_sizes[k] = max(gl / ld, 1) / 1000.0

            for i in range(N):
                for j in range(i + 1, N):
                    denom = bin_sizes[i] * bin_sizes[j]
                    if denom > 0.0:
                        v = h_sub[i, j] / denom
                        h_sub[i, j] = v
                        h_sub[j, i] = v

        return h_anchor, h_sub

    def _build_heat_dist_subanchor(
        self,
        pos: F32Array,
        fixed: BoolArray,
        dtn: F32Array,
        subanchor_heat_raw: F64Array,
        step_size: float,
        label: str = "",
        log2: bool = False,
    ) -> F64Array | None:
        """
        Estimate expected pairwise distances for subanchor heat energy.
        Mirrors Reference pipeline: run N dry smooth MC passes, average pairwise
        distances between all beads, then create target distance matrix.

        Returns (N, N) float64 target distance matrix or None if heatmap empty.
        """
        import time

        s = self.s
        n = len(pos)
        n_reps = int(s.subanchor_estimate_replicates)
        n_steps = int(s.subanchor_estimate_steps)
        log1 = s.output_level >= 1

        if log1:
            n_movable = int((~fixed).sum())
            print(f"  [{label}] heat dist matrix: {n} beads "
                  f"({n_movable} movable), {n_reps} reps × {n_steps} steps")

        avg_dist: F64Array = np.zeros((n, n), dtype=np.float64)
        t_mc_total = 0.0

        # Mirrors Reference: for each replicate, run n_steps MC passes from pos+noise,
        # keep the best structure, then accumulate pairwise distances from it.
        for rep in range(n_reps):
            t_rep = time.perf_counter()
            rep_best_score = -1.0
            rep_best_pos: F32Array = pos.copy()
            for step in range(n_steps):
                pos_trial: F32Array = pos.copy()
                for i in range(n):
                    if not fixed[i]:
                        pos_trial[i] += random_vector_np(step_size)
                score = mc_smooth(pos_trial, dtn, fixed, step_size, s,
                                  label=f"{label} est {rep + 1}/{n_reps} step {step + 1}/{n_steps}",
                                  verbose=log2)
                if score < rep_best_score or rep_best_score < 0.0:
                    rep_best_score = score
                    rep_best_pos = pos_trial.copy()
            diff = rep_best_pos[:, np.newaxis, :] - rep_best_pos[np.newaxis, :, :]
            avg_dist += np.sqrt((diff * diff).sum(axis=2))
            t_rep = time.perf_counter() - t_rep
            t_mc_total += t_rep
            if log1:
                print(f"    rep {rep + 1}/{n_reps}: best_score={rep_best_score:.4f}  "
                      f"({t_rep:.2f}s)")

        avg_dist /= n_reps

        # Create expected distance matrix mirroring Reference createExpectedDistSubanchorHeatmap().
        avg_heat = float(subanchor_heat_raw.mean())
        if avg_heat < 1e-6:
            if log1:
                print(f"  [{label}] heat dist matrix: empty heatmap (mean<1e-6), skipped")
            return None

        influence = float(s.subanchor_heatmap_influence)
        heat_dist: F64Array = np.zeros((n, n), dtype=np.float64)
        n_pairs_active = 0
        n_pairs_capped = 0

        for i in range(n):
            for j in range(i + 1, n):
                s_val = (subanchor_heat_raw[i, j] / avg_heat) * influence
                if s_val > 0.0:
                    n_pairs_active += 1
                if s_val > 1.0:
                    s_val = 1.0
                    n_pairs_capped += 1
                target = avg_dist[i, j] * (1.0 - s_val)
                heat_dist[i, j] = target
                heat_dist[j, i] = target

        if log1:
            n_pairs_total = n * (n - 1) // 2
            iu = np.triu_indices(n, k=1)
            upper = heat_dist[iu]
            avg_dist_upper = avg_dist[iu]
            mean_reduction = (1.0 - upper.mean() / avg_dist_upper.mean()) * 100.0 \
                if avg_dist_upper.mean() > 0 else 0.0
            print(f"  [{label}] heat dist matrix: avg_pair_dist={avg_dist_upper.mean():.3f}  "
                  f"avg_heat={avg_heat:.4g}  influence={influence}")
            print(f"  [{label}] heat dist matrix: {n_pairs_active}/{n_pairs_total} pairs active "
                  f"({n_pairs_capped} capped at full reduction); "
                  f"mean target reduction {mean_reduction:.1f}%  "
                  f"(total MC time {t_mc_total:.2f}s)")

        return heat_dist

    def _densify_active_region(
        self, active_region: list[int]
    ) -> tuple[F32Array, BoolArray, list[int], F32Array, list[AnchorMapEntry]]:
        """
        Insert loop_density subanchor beads between each consecutive anchor pair.
        Returns (pos, fixed, gpos, dtn, anchor_map) where:
          pos        : (N, 3) float32 bead positions
          fixed      : (N,) bool - True for original anchor beads
          gpos       : list[int] genomic midpoints
          dtn        : (N-1,) float32 expected consecutive distances
          anchor_map : list of (pos_index, cluster_index) for anchor beads
        Mirrors Reference LooperSolver::densifyActiveRegion().
        """
        ld = self.s.loop_density
        bead_starts: list[int] = []
        bead_ends: list[int] = []
        bead_pos: list[F32Array] = []
        bead_gpos: list[int] = []
        bead_fixed: list[bool] = []
        anchor_map: list[AnchorMapEntry] = []

        for i in range(len(active_region) - 1):
            ai = active_region[i]
            aj = active_region[i + 1]
            ca = self.clusters[ai]
            cb = self.clusters[aj]

            k = len(bead_pos)
            bead_starts.append(ca.start)
            bead_ends.append(ca.end)
            bead_pos.append(ca.pos.copy())
            bead_gpos.append(ca.genomic_pos)
            bead_fixed.append(True)
            anchor_map.append((k, ai))

            gap_bp = max(cb.start - ca.end, 0)  # clamp: overlapping anchors -> place subanchors at boundary
            d_bp = gap_bp // (ld + 1)
            p = ca.end
            for j in range(ld):
                p += d_bp
                t = (j + 1.0) / (ld + 1)
                sub_pos: F32Array = ((1.0 - t) * ca.pos + t * cb.pos).astype(np.float32)
                bead_starts.append(p)
                bead_ends.append(p)
                bead_pos.append(sub_pos)
                bead_gpos.append(p)
                bead_fixed.append(False)

        last_ci = active_region[-1]
        k = len(bead_pos)
        cl = self.clusters[last_ci]
        bead_starts.append(cl.start)
        bead_ends.append(cl.end)
        bead_pos.append(cl.pos.copy())
        bead_gpos.append(cl.genomic_pos)
        bead_fixed.append(True)
        anchor_map.append((k, last_ci))

        n = len(bead_pos)
        pos_arr: F32Array = np.array(bead_pos, dtype=np.float32)
        fixed_arr: BoolArray = np.array(bead_fixed, dtype=np.bool_)

        dtn: F32Array = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            gap = max(bead_gpos[i + 1] - bead_gpos[i], 0)
            dtn[i] = float(self.s.genomic_length_to_distance(gap))

        return pos_arr, fixed_arr, bead_gpos, dtn, anchor_map

    def _reconstruct_cluster_smooth(
        self,
        active_region: list[int],
        chr_: str = "",
        label: str = "",
        subanchor_heat_raw: F64Array | None = None,
        log1: bool = True,
        log2: bool = False,
        s_override: Settings | None = None,
    ) -> list[BeadOut]:
        """
        Densify active region, then run smooth MC (chain + angle energy).
        Writes final anchor positions back to self.clusters.
        Returns list of (genomic_pos, x, y, z) for ALL beads (anchors + subanchors).
        Mirrors Reference MonteCarloArcsSmooth loop in reconstructClustersArcsDistances().

        When subanchor_heat_raw is provided and use_subanchor_heatmap is True:
          - runs dry smooth MC passes to estimate avg pairwise distances
          - builds target distance matrix
          - adds heat energy term to the final smooth MC
        """
        import math as _math
        s = s_override if s_override is not None else self.s
        pos, fixed, gpos, dtn, anchor_map = self._densify_active_region(active_region)
        n = len(pos)
        if n <= 2:
            return []

        avg_dtn = float(dtn.mean())
        step_size = avg_dtn * s.noise_smooth

        # Build CTCF orientation data if enabled
        char_orn: np.ndarray[Any, Any] | None = None
        anchor_neighbors: dict[int, list[int]] | None = None
        anchor_neighbor_weights: dict[int, list[float]] | None = None
        if s.use_ctcf_motif and chr_:
            char_orn = np.array(['N'] * n, dtype='<U1')
            n_anchors_orn = len(anchor_map)
            cluster_to_anchor_k = {ci: k for k, (_, ci) in enumerate(anchor_map)}
            for k, (bi, ci) in enumerate(anchor_map):
                char_orn[bi] = self.clusters[ci].orientation or 'N'

            chr_arcs = self.arcs.get(chr_, [])
            anchor_neighbors = {k: [] for k in range(n_anchors_orn)}
            anchor_neighbor_weights = {k: [] for k in range(n_anchors_orn)}
            for k, (bi, ci) in enumerate(anchor_map):
                for arc_local in self.clusters[ci].arcs:
                    if arc_local >= len(chr_arcs):
                        continue
                    arc = chr_arcs[arc_local]
                    other_ci = arc.end if arc.start == ci else arc.start
                    if other_ci in cluster_to_anchor_k:
                        other_k = cluster_to_anchor_k[other_ci]
                        anchor_neighbors[k].append(other_k)
                        anchor_neighbor_weights[k].append(
                            _math.sqrt(max(arc.score, 0)))

        # Build subanchor heat distance matrix if enabled.
        # This runs N dry smooth MC passes without heat to estimate avg pairwise
        # distances, then scales them down for high-contact pairs.
        heat_dist: F64Array | None = None
        if subanchor_heat_raw is not None and s.use_subanchor_heatmap:
            heat_dist = self._build_heat_dist_subanchor(
                pos, fixed, dtn, subanchor_heat_raw, step_size, label=label, log2=False)

        best_score = -1.0
        best_pos: F32Array = pos.copy()

        for run in range(s.steps_smooth):
            run_label = f"{label} smooth {run + 1}/{s.steps_smooth}"
            if log1:
                print(f"  {run_label}")
            pos_run: F32Array = best_pos.copy()
            for i in range(n):
                if not fixed[i]:
                    pos_run[i] += random_vector_np(step_size)

            score = mc_smooth(pos_run, dtn, fixed, step_size, s,
                              char_orientations=char_orn,
                              anchor_neighbors=anchor_neighbors,
                              anchor_neighbor_weights=anchor_neighbor_weights,
                              heat_dist=heat_dist,
                              label=run_label, verbose=log2)

            if score < best_score or best_score < 0:
                best_score = score
                best_pos = pos_run.copy()

        for bead_idx, cluster_idx in anchor_map:
            self.clusters[cluster_idx].pos = best_pos[bead_idx].copy()

        return [
            (gpos[i], float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_pos[i, 2]))
            for i in range(n)
        ]

    # Heatmap normalisation helpers

    @staticmethod
    def _get_diagonal_size(h: list[list[float]], n: int) -> int:
        """Find smallest w such that any cell at distance w from diagonal is non-zero."""
        for w in range(n):
            for i in range(n - w):
                if h[i][i + w] > 1e-6:
                    return w
        return 0

    @staticmethod
    def _normalize_heatmap(h: list[list[float]], n: int) -> list[list[float]]:
        """
        Row-normalize: scale each row so all rows have equal sum (avg).
        Then symmetrize: h[i][j] = (h[i][j] + h[j][i]) / 2.
        Mirrors Reference LooperSolver::normalizeHeatmap().
        """
        row_sums = [sum(h[i]) for i in range(n)]
        total = sum(row_sums)
        if total < 1e-10:
            return h
        expected = total / n

        out: list[list[float]] = [[0.0] * n for _ in range(n)]
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
    def _normalize_heatmap_diagonal_total(
        h: list[list[float]], n: int, val: float
    ) -> None:
        """
        Normalize so the average of the first non-zero diagonal equals val.
        Mirrors Reference normalizeHeatmapDiagonalTotal().
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
        h: list[list[float]],
        n: int,
        current_level: ChrLevel,
        scale: float,
    ) -> None:
        """
        Scale inter-chromosomal entries.
        Mirrors Reference normalizeHeatmapInter().
        Modifies h in place.
        """
        # This is only relevant for multi-chromosome runs; skip for now.
        pass

    def _create_distance_heatmap(
        self,
        h: list[list[float]],
        n: int,
        inter: bool = False,
    ) -> tuple[list[list[float]], float]:
        """
        Convert normalized contact frequency heatmap to expected distance heatmap.
        Mirrors Reference createDistanceHeatmap().

        Returns (dist_heatmap, avg_dist) where dist_heatmap is a 2D list.
        Entries within diagonal_size are set to -1 (ignored in scoring).
        """
        s = self.s
        diag = self._get_diagonal_size(h, n)

        dist: list[list[float]] = [[0.0] * n for _ in range(n)]
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

    # Output helpers

    def get_leaf_positions(self, chr_: str) -> list[BeadOut]:
        """
        Return all bead positions for chr_ as list of (midpoint_bp, x, y, z),
        sorted by genomic midpoint.  Includes subanchor beads when smooth MC
        has been run; falls back to anchor-only otherwise.
        """
        dense = self.dense_active_regions.get(chr_)
        if dense:
            return sorted(dense, key=lambda b: b[0])
        # Fallback: anchor-level beads only
        result: list[BeadOut] = []
        first = self.chr_first_cluster.get(chr_, -1)
        if first < 0:
            return result
        for i in range(first, len(self.clusters)):
            c = self.clusters[i]
            if c.level != LVL_ANCHOR:
                break
            result.append((c.genomic_pos, float(c.pos[0]), float(c.pos[1]), float(c.pos[2])))
        result.sort(key=lambda b: b[0])
        return result

    def get_anchor_positions(self) -> list[BeadOut]:
        """All anchor beads from all chromosomes, sorted by chr then genomic position."""
        result: list[BeadOut] = []
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
