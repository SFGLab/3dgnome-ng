"""
LooperSolver - main orchestrator.

Reproduces the runLooper() pipeline from cudaMMC main.cpp:
    1. setContactData()           → load anchors + arcs
    2. createTreeGenome()         → build hierarchical model
    3. reconstructClustersHeatmap()  → Phase 1 GPU heatmap MC per level
    4. reconstructClustersArcsDistances() → Phase 2+3 CPU arc MC
    5. save()                     → write output

All four reconstruction phases are run level by level, top → bottom.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch

from .data_loading import load_anchors, load_pet_clusters, load_segments_split, mark_arcs
from .data_structures import Anchor, Cluster, InteractionArc
from .distances import count_to_distance, genomic_length_to_distance
from .heatmap import (
    build_singleton_heatmap, build_singleton_heatmap_bins,
    normalize_heatmap, heatmap_to_expected_distances,
)
from .mc import monte_carlo_heatmap, monte_carlo_arcs, monte_carlo_arcs_smooth, _random_displacements
from .regions import Region, build_filter, chrom_included, filter_anchors, default_regions
from .settings import Settings
from .tree import ChromosomeTree, interpolate_children_spline


class LooperSolver:
    """Reconstruct 3D chromatin structure from ChIA-PET data."""

    def __init__(self, settings: Optional[Settings] = None,
                 regions: Optional[List[Region]] = None):
        self.settings = settings or Settings()
        self.device = torch.device(self.settings.device)

        # region filter – None means all chromosomes
        self._region_filter = build_filter(regions) if regions is not None else None

        # populated by set_contact_data()
        self.anchors_by_chr: Dict[str, List[Anchor]] = {}
        self.arcs_by_chr: Dict[str, List[InteractionArc]] = {}
        self.trees: Dict[str, ChromosomeTree] = {}

        # heatmap expected-distance matrices (one per chromosome)
        self.heatmap_expected: Dict[str, torch.Tensor] = {}
        # Data-driven diagonal sizes alongside the heatmaps (cudaMMC
        # Heatmap.cpp:58 getDiagonalSize, AUDIT §B4/§C1).
        self.heatmap_diagonal_size: Dict[str, int] = {}

        # Predefined segment-split breakpoints (cudaMMC ``segments_predefined``,
        # loaded from ``Settings.dataSegmentsSplit`` at LooperSolver.cpp:41-44).
        # Empty dict ⇒ findSplit Branch B (every gap is a segment boundary).
        self.segments_predefined: Dict[str, List[Tuple[int, int]]] = {}

    # ── Data loading ──────────────────────────────────────────────────────────

    def set_contact_data(
        self,
        anchors_bed: str,
        pet_clusters_bedpe: str,
        singletons_bedpe: Optional[str] = None,
        factor: int = 0,
    ):
        """Load all input files and map arcs to anchor indices."""
        print("Loading anchors...")
        all_anchors = load_anchors(anchors_bed)

        # apply chromosome / region filter
        flt = self._region_filter
        if flt is not None:
            self.anchors_by_chr = {
                chrom: filter_anchors(ancs, flt)
                for chrom, ancs in all_anchors.items()
                if chrom_included(chrom, flt) and filter_anchors(ancs, flt)
            }
            skipped = set(all_anchors) - set(self.anchors_by_chr)
            if skipped:
                print(f"  Skipping chromosomes not in region filter: "
                      f"{', '.join(sorted(skipped))}")
        else:
            self.anchors_by_chr = all_anchors

        print("Loading PET clusters...")
        raw_arcs = load_pet_clusters(pet_clusters_bedpe, factor)
        # Singletons are used ONLY for the heatmap expected-distance matrix (heatmap MC phase).
        # cudaMMC never adds singletons to the arc-spring list — individual PET reads are too
        # noisy to be structural constraints; only statistically significant clusters are.

        # drop arcs that reference chromosomes we're not modelling
        if flt is not None:
            raw_arcs = [
                a for a in raw_arcs
                if a.chrom1 in self.anchors_by_chr and a.chrom2 in self.anchors_by_chr
            ]

        print("Marking arcs...")
        self.arcs_by_chr, _ = mark_arcs(raw_arcs, self.anchors_by_chr)
        print(f"  {sum(len(v) for v in self.arcs_by_chr.values())} arcs mapped.")

        # cudaMMC LooperSolver.cpp:41-44: load predefined segment-split BED if
        # Settings::dataSegmentsSplit is set.  Used by findSplit Branch A
        # (LooperSolver.cpp:911-962) to promote arc-sweep gaps whose span
        # contains a breakpoint into segment boundaries.
        if self.settings.data_segments_split:
            self.segments_predefined = load_segments_split(
                self.settings.data_segments_split)
            n_seg = sum(len(v) for v in self.segments_predefined.values())
            print(f"  Loaded {n_seg} predefined segment breakpoints.")

    def load_heatmap(self, singletons_bedpe: str):
        """Build per-chromosome **segment-level** singleton heatmaps.

        Mirrors cudaMMC ``createSingletonHeatmap(1)`` invoked from
        ``reconstructClustersHeatmap`` (LooperSolver.cpp:252) at
        ``LVL_SEGMENT``: each row/column of the matrix is one **segment
        bead** (level-2 cluster in the tree), and singletons are binned by
        whichever segment span contains the midpoint of each PET endpoint.

        AUDIT §A1, §G9: this replaces the old anchor-level heatmap, which
        was running heatmap MC at the wrong scale (N ≈ 26 k anchor beads
        per chrom).  After this change the heatmap MC runs at the segment
        scale (typically tens to a few hundred beads — same order as
        cudaMMC), and the cascade flows top-down from there.

        The normalised expected-distance matrix is stored in
        ``self.heatmap_expected[chrom]`` keyed by chromosome, with shape
        ``(n_segments, n_segments)`` fp32 and the data-driven diagonal
        size (cudaMMC Heatmap.cpp:58, AUDIT §B4/§C1) carried alongside
        in ``self.heatmap_diagonal_size[chrom]``.
        """
        self.heatmap_diagonal_size: Dict[str, int] = {}
        for chrom, tree in self.trees.items():
            # cudaMMC LooperSolver.cpp:252: createSingletonHeatmap binned at
            # the current_level — for setLevel(LVL_SEGMENT) that means each
            # row is one segment bead (level-2 cluster).
            seg_bins: List[Tuple[str, int, int]] = []
            seg_cidxs: List[int] = []
            for ci, c in enumerate(tree.clusters):
                if c.level == 2:
                    seg_bins.append((chrom, c.start, c.end))
                    seg_cidxs.append(ci)
            n_seg = len(seg_bins)
            if n_seg < 2:
                # Single-segment chrom: cpp:116-120 skips MC entirely and
                # parks the bead at (0,0,0).  We do the same downstream;
                # no heatmap needed here.
                print(f"  {chrom}: single-segment region — heatmap MC will be skipped.")
                continue

            print(f"  Building segment heatmap for {chrom} "
                  f"({n_seg} segments)...")
            raw = build_singleton_heatmap_bins(singletons_bedpe, seg_bins)
            raw = raw.to(self.device)
            # cudaMMC LooperSolver.cpp:256-258: normalize → diagonalTotal → inter.
            # We skip normalizeHeatmapInter here (single-chrom path is dominant).
            # AUDIT §B5: multi-chrom inter-scaling lands when §G9 LVL_CHROMOSOME
            # cascade does.
            norm, diag = normalize_heatmap(raw, anchors=None)
            del raw
            # cudaMMC LooperSolver.cpp:264: createDistanceHeatmap(1)
            exp = heatmap_to_expected_distances(
                norm,
                scale=self.settings.freq_dist_scale,
                power=self.settings.freq_dist_power,
                diagonal_size=diag,
                max_stretching=self.settings.heatmap_distance_heatmap_stretching,
            )
            del norm
            self.heatmap_expected[chrom] = exp
            self.heatmap_diagonal_size[chrom] = diag
            # Remember the level-2 cluster indices so the MC writes positions
            # back into the right beads.
            tree._segment_cidxs = seg_cidxs   # solver-attached helper attr

    # ── Tree construction ─────────────────────────────────────────────────────

    def create_tree_genome(self):
        """Build hierarchical bead-spring models for every chromosome.

        Positions are filled later by the top-down cascade:

          * segment beads (level 2) — by Phase-1 heatmap MC
            (:meth:`reconstruct_clusters_heatmap`);
          * IB beads (level 3) — by :meth:`ChromosomeTree.position_interaction_blocks`
            inside Phase-2 (genomic-distance spline along the segment chain,
            cpp:2599 → 2709-2725);
          * anchor beads (level 4) — by per-IB arc MC starting from
            ``ib.pos + random_vector(noise_size_small)`` (cpp:2624-2625, 2847-2848);
          * subanchor beads (level 5) — inserted by
            :meth:`ChromosomeTree.densify_active_region` between the arc and
            smooth phases (cpp:2645 → 2448-2510).

        Nothing is positioned here; tree construction is purely topological.
        """
        print("Building hierarchical models...")
        for chrom, anchors in self.anchors_by_chr.items():
            arcs = self.arcs_by_chr.get(chrom, [])
            tree = ChromosomeTree(chrom, anchors, arcs, self.settings,
                                  segments_predefined=self.segments_predefined.get(chrom, []))
            self.trees[chrom] = tree
            n_seg = sum(1 for c in tree.clusters if c.level == 2)
            n_ib = sum(1 for c in tree.clusters if c.level == 3)
            n_anc = len(tree.anchors_idx)
            print(f"  {chrom}: {len(tree.clusters)} clusters, "
                  f"{n_seg} segments, {n_ib} IBs, {n_anc} anchors")

            # Detect the "no breakpoints, dense ChIA-PET" degenerate case:
            # find_all_gaps returns only [0, N-1] when arcs_cnt never reaches
            # zero, collapsing the whole chromosome into one IB.  cudaMMC
            # produces the same useless tree here, but it ALWAYS ships with a
            # ``segment_split`` BED, so this state never arises in practice.
            # Fail loud with an actionable message instead of grinding through
            # an O(N²) arc MC on 20 k anchors.
            if (n_anc > 500 and n_ib <= 2
                    and not self.segments_predefined.get(chrom)):
                raise RuntimeError(
                    f"\n  {chrom}: {n_anc} anchors collapsed into {n_ib} "
                    "interaction block(s).\n"
                    "  This happens when the arc-sweep (cudaMMC findGaps, "
                    "LooperSolver.cpp:856-894) finds no arc-free positions —\n"
                    "  expected for full chromosomes with dense ChIA-PET data.\n"
                    "  cudaMMC handles this via a predefined segment-split "
                    "BED (Settings::dataSegmentsSplit, cpp:911-962).\n\n"
                    "  Fix: pass --config pointing to an INI whose [data] "
                    "section sets `segment_split = <breakpoints.bed>`.\n"
                    "  The bundled data/<celltype>/config.ini already does "
                    "this; auto-detection looks for config.ini next to\n"
                    "  --anchors, so usually `mv` or symlink your config there."
                )

    # ── Arc expected distances (per-IB dense matrix, AUDIT §G1) ──────────────

    def _build_anchor_expected_dist_ib(
        self, chrom: str, anchor_cidxs: List[int],
    ) -> torch.Tensor:
        """Build the **dense N×N** anchor expected-distance matrix for one IB.

        Mirror of cudaMMC ``calcAnchorExpectedDistancesHeatmap``
        (LooperSolver.cpp:3837-3916).  Phase-2 score
        :func:`scores.score_distances_active_region` iterates every (i,j)
        pair of this matrix and contributes either a spring (arc-connected,
        ``> 1e-6``), a ``1/d`` repulsion (``< 0`` sentinel), or skip
        (``< 1e-6``).  AUDIT §G1, §G2, §G3.

        Construction (cpp:3841-3882):

          1. ``init(n)`` then ``add(-1.0f)``: every entry defaults to ``-1``.
          2. ``clearDiagonal(1)``: the diagonal (and immediate sub/super-
             diagonal — cudaMMC's ``Heatmap::clearDiagonal(1)`` zeroes the
             main diagonal only at offset 0) is set to 0.
          3. For each anchor in the IB, walk its ``clusters[ai].arcs``, find
             the other endpoint in the same IB via ``cluster_to_active_index``,
             and overwrite ``h[a][b] = h[b][a] = freqToDistance(freq, true)``
             (cpp:3875-3881).
          4. ``useAnchorHeatmap`` modulation (cpp:3886-3914) is **not**
             ported — gated off in :class:`Settings`.

        AUDIT §G2: iterates ``clusters[ai].arcs`` (the per-anchor arc index
        list set up by ``mark_arcs`` → ``tree._build``), not the chromosome
        arc list, so cross-IB arcs are silently filtered.
        """
        n = len(anchor_cidxs)
        # cpp:3842-3843: init(n) ; add(-1.0f) — every off-diag entry → -1
        exp = torch.full((n, n), -1.0, dtype=torch.float32, device=self.device)
        if n == 0:
            return exp
        # cpp:3844: clearDiagonal(1) — main diagonal → 0 (cudaMMC zeroes the
        # i==j entries only; cf. Heatmap::clearDiagonal(width=0) semantics).
        exp.fill_diagonal_(0.0)

        s = self.settings
        tree = self.trees[chrom]
        # cpp:3849-3853: cluster_to_active_index (cluster index → local 0..n-1).
        c2a = {ci: li for li, ci in enumerate(anchor_cidxs)}

        chrom_arcs = self.arcs_by_chr.get(chrom, [])
        for ai_local, ci in enumerate(anchor_cidxs):
            anchor_c = tree.clusters[ci]
            # cpp:3857-3858: iterate the arc indices stored on this anchor.
            for arc_i in anchor_c.arcs:
                if not (0 <= arc_i < len(chrom_arcs)):
                    continue
                arc = chrom_arcs[arc_i]
                # cpp:3860: otherEnd — resolve the other endpoint's CLUSTER index
                #   via tree.anchors_idx[arc.start|end] (arc.start/end are
                #   ANCHOR indices, mark_arcs convention).
                a_cluster = tree.anchors_idx[arc.start] \
                            if arc.start < len(tree.anchors_idx) else -1
                b_cluster = tree.anchors_idx[arc.end] \
                            if arc.end   < len(tree.anchors_idx) else -1
                other = b_cluster if a_cluster == ci else a_cluster
                if other == ci or other < 0:
                    continue
                # cpp:3877-3878: cluster_to_active_index — silently skip cross-IB.
                bj_local = c2a.get(other, -1)
                if bj_local < 0:
                    continue
                # cpp:3869-3870: only consider forward loops to count once
                #   (we'll write both v[a][b] and v[b][a]; the forward-only
                #   gate is an optimisation upstream, the matrix is symmetric).
                if bj_local <= ai_local:
                    continue
                # cpp:3874-3875: freqToDistance(freq, true) — memoised for
                #   freq ∈ [1, 100] in cudaMMC; we just call directly.
                dist = count_to_distance(
                    arc.score,
                    s.count_dist_a, s.count_dist_scale,
                    s.count_dist_shift, s.count_dist_base_level,
                )
                # cpp:3880-3881: write both halves of the symmetric matrix.
                exp[ai_local, bj_local] = dist
                exp[bj_local, ai_local] = dist
        return exp

    def _chain_lengths_ib(self, chrom: str, anchor_cidxs: List[int]
                           ) -> torch.Tensor:
        """Expected linker lengths for consecutive (sub)anchors within one IB.

        Center-to-center genomic distance through ``genomic_length_to_distance``,
        matching ``LooperSolver.cpp:2767-2771``.
        """
        tree = self.trees[chrom]
        s = self.settings
        lengths = []
        for i in range(len(anchor_cidxs) - 1):
            c1 = tree.clusters[anchor_cidxs[i]]
            c2 = tree.clusters[anchor_cidxs[i + 1]]
            d = abs(c2.genomic_pos - c1.genomic_pos)
            lengths.append(genomic_length_to_distance(
                d, s.genomic_dist_scale, s.genomic_dist_power, s.genomic_dist_base,
            ))
        if not lengths:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        return torch.tensor(lengths, dtype=torch.float32, device=self.device)

    # ── Phase 1: heatmap MC at segment level (cudaMMC reconstructClustersHeatmap) ──

    def reconstruct_clusters_heatmap(self):
        """Run segment-level heatmap Monte Carlo for each chromosome.

        Mirrors ``LooperSolver::reconstructClustersHeatmap`` (cpp:85-294) +
        ``reconstructClustersHeatmapSingleLevel(1)`` (cpp:297-419) at
        ``LVL_SEGMENT``.  AUDIT §A1-A5, §G9 closed by this rewrite.

        Pipeline per chromosome:

          1. If the chromosome has ≤ 1 segment (cpp:116-120): park the
             segment bead at (0,0,0) and return.
          2. Compute ``avg_dist = mean(non-zero heatmap_dist) *
             noiseCoefficientLevelSegment`` (cpp:312, AUDIT §A5).
          3. Initial structure = parent (root) position; we reset every
             segment bead to it before the multi-restart loop (cpp:326-328).
          4. ``for k in range(simulationStepsLevelSegment):`` (cpp:357)
             - reset positions to ``initial + random_vector(avg_dist)``
               per segment (cpp:360-363);
             - run :func:`monte_carlo_heatmap` with ``step_size = avg_dist``;
             - keep the best-scored structure (cpp:399-405).
          5. Write the best segment positions back into the tree.

        Multi-chromosome runs trigger the ``LVL_CHROMOSOME`` cascade in
        cudaMMC (cpp:128-181); that path is not yet ported (AUDIT §G9).
        """
        from .scores import score_heatmap_chunked as _score_heat

        if len(self.trees) > 1:
            # AUDIT §G9: LVL_CHROMOSOME cascade for multi-chrom is not ported.
            # cudaMMC builds a separate chromosome-level singleton heatmap
            # (cpp:128-180) and runs MC on one bead per chromosome before
            # descending to LVL_SEGMENT.  Raise rather than silently diverge.
            raise NotImplementedError(
                "Multi-chromosome runs require the LVL_CHROMOSOME cascade "
                "(cudaMMC LooperSolver.cpp:128-180) which is not yet ported. "
                "Run one chromosome at a time, or extend reconstruct_clusters_heatmap."
            )

        s = self.settings
        for chrom, tree in self.trees.items():
            seg_cidxs = [i for i, c in enumerate(tree.clusters) if c.level == 2]
            n_seg = len(seg_cidxs)
            if n_seg == 0:
                continue

            print(f"\n[Heatmap MC LVL_SEGMENT] {chrom}  ({n_seg} segments)")

            # cudaMMC cpp:116-120: single-segment region → park at origin, skip MC.
            if n_seg <= 1:
                tree.clusters[seg_cidxs[0]].set_pos(0.0, 0.0, 0.0)
                print("  Single-segment region — segment placed at (0,0,0), "
                      "skipping heatmap MC (cudaMMC cpp:116-120).")
                continue

            if chrom not in self.heatmap_expected:
                # No singletons heatmap available — fall back to a genomic-
                # distance-derived expected matrix.  Not in cudaMMC, but the
                # only sensible thing to do without a real heatmap input.
                exp = self._genomic_expected_matrix_segments(tree, seg_cidxs)
                diag = 1
            else:
                exp = self.heatmap_expected[chrom]
                diag = self.heatmap_diagonal_size.get(chrom, 1)

            # cudaMMC cpp:307,312: avg_dist = heatmap_dist.getAvg() * noiseCoef
            # getAvg averages every entry of heatmap_dist (including 0 and the
            # -1 sentinel — matches cpp:1779-1784 inside createDistanceHeatmap).
            avg_dist = float(exp.mean().item()) * s.noise_coefficient_level_segment
            if avg_dist <= 0.0:
                avg_dist = float(exp[exp > 0].mean().item()) \
                            if (exp > 0).any() else 1.0
            print(f"  avg_dist = {avg_dist:.4f}  diagonal_size = {diag}")

            # cudaMMC cpp:326-328: set every segment bead to parent.pos (root).
            # In our tree the chromosome root sits at level 1, parent of all
            # segments.  Default position is (0,0,0) — matches cudaMMC's
            # implicit initialisation when ``useDensity`` / ``useTelomere`` are off.
            root_idxs = [i for i, c in enumerate(tree.clusters) if c.level == 1]
            origin = (0.0, 0.0, 0.0)
            if root_idxs:
                r = tree.clusters[root_idxs[0]]
                origin = (r.x, r.y, r.z)
            initial_structure = torch.tensor(
                [origin] * n_seg, dtype=torch.float32, device=self.device,
            )

            n_restarts = s.simulation_steps_level_segment
            fixed = torch.zeros(n_seg, dtype=torch.bool, device=self.device)

            best_pos: Optional[torch.Tensor] = None
            best_score = float("inf")

            for k in range(n_restarts):
                # cudaMMC cpp:360-363: pos = initial + random_vector(avg_dist)
                # — uniform per-axis, common.cpp:14-25.
                pos = initial_structure + _random_displacements(
                    n_seg, avg_dist, s.use_2d, self.device,
                )
                pos = monte_carlo_heatmap(
                    pos, exp, fixed, s,
                    diagonal_size=diag,
                    step_size=avg_dist,                          # cpp:390
                    verbose=(k == 0),
                )
                # cpp:392, 400-405: score the structure; keep the best.
                # We use the double-counted convention to match _full_score in mc.py.
                sc = 2.0 * _score_heat(pos, exp, diag).item()
                is_best = sc < best_score
                if is_best:
                    best_score = sc
                    best_pos = pos.clone()
                print(f"  restart {k+1}/{n_restarts}  score={sc:.6f}  "
                      f"best={best_score:.6f}{'  *' if is_best else ''}")

            # cpp:408-410: restore the best structure into the tree.
            if best_pos is not None:
                tree.set_positions_from_tensor(best_pos, seg_cidxs)

    def _genomic_expected_matrix_segments(self, tree: ChromosomeTree,
                                          seg_cidxs: List[int]) -> torch.Tensor:
        """Fallback expected-distance matrix from genomic positions for segments.

        Used when no singleton heatmap was provided.  Not in cudaMMC.
        """
        s = self.settings
        gpos = torch.tensor(
            [tree.clusters[i].genomic_pos for i in seg_cidxs],
            dtype=torch.float32, device=self.device,
        )
        diff_kb = (gpos.unsqueeze(1) - gpos.unsqueeze(0)).abs() / 1000.0
        exp = s.genomic_dist_base + s.genomic_dist_scale * \
              diff_kb.clamp(min=1e-3) ** s.genomic_dist_power
        exp.fill_diagonal_(0.0)
        return exp

    # ── Phase 2+3: per-IB arc / smooth MC ────────────────────────────────────

    def reconstruct_clusters_arcs_distances(self):
        """Per-IB arc + densify + smooth pipeline (cpp:2579-2702).

        AUDIT §G1, §G4, §G5, §G7, §G8 closed by this rewrite.  Sequence
        per chromosome / IB:

          1. ``positionInteractionBlocks`` — spline-place IB beads ALONG the
             segment chain in genomic-distance mode (cpp:2599 → 2709-2725,
             AUDIT §G4 / §H5).  Replaces the old bottom-up
             ``_propagate_positions_up`` workaround (AUDIT §G14).
          2. For each IB with ≥ 2 anchors:
             a. Build the dense expected-distance matrix via
                :meth:`_build_anchor_expected_dist_ib` (AUDIT §G1).
             b. Anchors initialised at ``ib.pos`` (cpp:2624-2625) then
                re-noised each restart by ``random_vector(noise_size_small)``
                (cpp:2847-2848 with ``smooth=False``).
             c. Run :func:`monte_carlo_arcs` on the dense matrix,
                ``step_size = noise_size = avg_chain × noiseCoefLevelAnchor``
                (cpp:2780-2782).  Multi-restart ``simulationStepsLevelAnchor``,
                keep best by full score (cpp:2836-2869).
             d. Densify via :meth:`ChromosomeTree.densify_active_region` —
                ``loopDensity`` subanchor beads inserted by linear interpolation
                between every consecutive anchor pair, anchors marked fixed
                (cpp:2645 → 2448-2510, AUDIT §G5).
             e. Smooth MC on the (anchors + subanchors) chain.  Per-restart
                re-noising at magnitude ``avg_chain × noiseCoefLevelSubanchor``
                (cpp:2781-2782, AUDIT §G7), uniform distribution (AUDIT §G8).
                Anchors stay fixed via ``fixed_mask``.
        """
        s = self.settings
        from .scores import score_distances_active_region as _score_arcs_dense
        from .scores import score_structure_smooth as _score_smooth

        for chrom, tree in self.trees.items():
            n_total = len(tree.anchors_idx)
            if n_total == 0:
                continue

            print(f"\n[Arcs+Smooth MC] {chrom}  ({n_total} total anchors)")

            # cpp:2599: positionInteractionBlocks — spline IBs along segments.
            # Closes AUDIT §G4 / §H5: IB beads inherit positions from the
            # genomic-distance-weighted segment-chain spline, not from a
            # bottom-up mean over their children.
            tree.position_interaction_blocks(parent_level=2)

            n_ibs_done = 0
            for root_ci in (i for i, c in enumerate(tree.clusters) if c.level == 1):
                for seg_ci in tree.clusters[root_ci].children:
                    seg_c = tree.clusters[seg_ci]
                    # Iterate over a *snapshot* of IB children: densify mutates
                    # each IB's ``children`` list, but the segment's children
                    # (= IBs) are unaffected.
                    for ib_ci in list(seg_c.children):
                        n_ibs_done = self._reconstruct_single_ib(
                            chrom, tree, ib_ci, n_ibs_done,
                            score_full_arcs=_score_arcs_dense,
                            score_full_smooth=_score_smooth,
                        )
            print(f"  Done: {n_ibs_done} IBs processed")

    def _reconstruct_single_ib(self, chrom: str, tree: ChromosomeTree,
                                ib_ci: int, n_ibs_done: int,
                                score_full_arcs, score_full_smooth) -> int:
        """Arc + densify + smooth for a single interaction block.

        Mirrors ``reconstructClusterArcsDistances`` (cpp:2735-2875).
        """
        s = self.settings
        ib_c = tree.clusters[ib_ci]
        anchor_cidxs = list(ib_c.children)           # level-4 cluster indices
        n_ib = len(anchor_cidxs)
        if n_ib < 2:
            return n_ibs_done                         # cpp:2741-2742

        # cpp:2625: every anchor reset to the IB centroid.
        ib_pos = torch.tensor(
            [ib_c.x, ib_c.y, ib_c.z],
            dtype=torch.float32, device=self.device,
        )

        # ── Phase 2: arc MC on dense expected-distance matrix ───────────────
        expected = self._build_anchor_expected_dist_ib(chrom, anchor_cidxs)
        chain_lengths_anc = self._chain_lengths_ib(chrom, anchor_cidxs)
        n_arc_pairs = int((expected > 1e-6).sum().item() // 2)

        # cpp:2767-2782: noise_size = avg(genomicLengthToDistance(consec gaps))
        # * noiseCoefficientLevelAnchor (anchors) / Subanchor (smooth).  We
        # already computed this average implicitly via chain_lengths_anc.
        avg_chain = (chain_lengths_anc.mean().item()
                     if len(chain_lengths_anc) > 0 else s.step_size_arcs)
        # cpp:2781 (smooth=false branch): noise_size *= noiseCoefficientLevelAnchor.
        noise_size_arcs = avg_chain * s.noise_coefficient_level_anchor

        print(f"  IB {n_ibs_done}: {n_ib} anchors, {n_arc_pairs} arc pairs, "
              f"avg_chain={avg_chain:.3f}, noise_arcs={noise_size_arcs:.3f}")

        fixed_arcs = torch.zeros(n_ib, dtype=torch.bool, device=self.device)
        n_restarts_arcs = s.simulation_steps_level_anchor

        best_pos_arcs: Optional[torch.Tensor] = None
        best_score_arcs = float("inf")
        for _k in range(n_restarts_arcs):
            # cpp:2844-2848: pos = initial_structure[i] + random_vector(
            #     smooth ? noise_size : noise_size_small, use2D)
            # Arc phase uses the LITERAL 0.05 noise_size_small (cpp:2765).
            pos = ib_pos.unsqueeze(0).expand(n_ib, 3).clone()
            pos = pos + _random_displacements(
                n_ib, s.noise_size_small, s.use_2d, self.device,
            )
            pos = monte_carlo_arcs(
                pos, expected, fixed_arcs, s,
                step_size=noise_size_arcs, verbose=False,
            )
            # cpp:2860, 2864: keep best.
            sc = score_full_arcs(
                pos, expected,
                k_stretch=s.spring_constant_stretch_arcs,
                k_squeeze=s.spring_constant_squeeze_arcs,
                k_repulsion=1.0,
            ).item()
            is_best = sc < best_score_arcs
            if is_best:
                best_score_arcs = sc
                best_pos_arcs = pos.clone()
            print(f"    arc   {_k+1}/{n_restarts_arcs}  "
                  f"score={sc:.4f}  best={best_score_arcs:.4f}"
                  + ("  *" if is_best else ""))

        # Write best anchor positions back into the tree BEFORE densify
        # so that densify_active_region reads them when interpolating
        # subanchor positions (cpp:2489-2490).
        assert best_pos_arcs is not None
        tree.set_positions_from_tensor(best_pos_arcs, anchor_cidxs)

        # ── Densify (cpp:2645 → 2448-2510, AUDIT §G5) ───────────────────────
        # Inserts loopDensity subanchor beads between each consecutive anchor
        # pair via linear interpolation; anchors are marked is_fixed = True.
        dense_active = tree.densify_active_region(ib_ci, fix=True)
        n_dense = len(dense_active)
        # Build boolean mask: True for anchor (fixed in smooth phase),
        # False for inserted subanchor.
        is_anchor = [tree.clusters[ci].level == 4 for ci in dense_active]
        fixed_smooth = torch.tensor(
            is_anchor, dtype=torch.bool, device=self.device,
        )
        is_anchor_mask = fixed_smooth.clone()

        # Load current positions (anchors + freshly-inserted subanchor positions).
        pos_dense = tree.positions_tensor(dense_active, str(self.device))

        chain_lengths_smooth = self._chain_lengths_ib(chrom, dense_active)
        orientations = [tree.clusters[ci].orientation for ci in dense_active]
        # The smooth-phase score does NOT use the arc spring list — pass
        # placeholder zero-length tensors; monte_carlo_arcs_smooth ignores
        # `arc_expected` (legacy compat per its docstring).
        arc_s_dummy = torch.zeros(0, dtype=torch.long, device=self.device)
        arc_e_dummy = torch.zeros(0, dtype=torch.long, device=self.device)
        arc_exp_dummy = torch.zeros(0, dtype=torch.float32, device=self.device)

        # cpp:2781 (smooth=true branch): noise_size *= noiseCoefficientLevelSubanchor.
        avg_chain_smooth = (chain_lengths_smooth.mean().item()
                             if len(chain_lengths_smooth) > 0
                             else s.step_size_smooth)
        noise_size_smooth = avg_chain_smooth * s.noise_coefficient_level_subanchor

        print(f"    densified to {n_dense} beads "
              f"(+{n_dense - n_ib} subanchors, "
              f"noise_smooth={noise_size_smooth:.3f})")

        # ── Phase 3: smooth MC on (anchors + subanchors) ────────────────────
        n_restarts_smooth = s.simulation_steps_level_subanchor
        best_pos_smooth: Optional[torch.Tensor] = None
        best_score_smooth = float("inf")
        initial_smooth = pos_dense.clone()
        for _k in range(n_restarts_smooth):
            # cpp:2844-2849: re-noise the structure.  cpp:2846:
            #   if (!smooth || !clusters[active_region[i]].is_fixed)
            # i.e. for smooth phase, ANCHOR beads (is_fixed=True after densify)
            # are NOT re-noised, only subanchors.  Magnitude = noise_size
            # (smooth) = avg_chain * noiseCoefLevelSubanchor.  AUDIT §G7 closed.
            noise = _random_displacements(
                n_dense, noise_size_smooth, s.use_2d, self.device,
            )
            # Zero out noise on fixed (anchor) beads.
            noise[is_anchor_mask] = 0.0
            pos_in = initial_smooth + noise

            pos_out = monte_carlo_arcs_smooth(
                pos_in, arc_s_dummy, arc_e_dummy, arc_exp_dummy,
                chain_lengths_smooth, orientations, fixed_smooth, s,
                step_size=noise_size_smooth,
                is_anchor_mask=is_anchor_mask,
                verbose=False,
            )

            sc = score_full_smooth(
                pos_out, chain_lengths_smooth,
                k_stretch=s.spring_constant_stretch,
                k_squeeze=s.spring_constant_squeeze,
                angular_k=s.spring_angular_constant,
                weight_dist=s.weight_dist_smooth,
                weight_angle=s.weight_angle_smooth,
            ).item()
            is_best = sc < best_score_smooth
            if is_best:
                best_score_smooth = sc
                best_pos_smooth = pos_out.clone()
            print(f"    smooth {_k+1}/{n_restarts_smooth}  "
                  f"score={sc:.4f}  best={best_score_smooth:.4f}"
                  + ("  *" if is_best else ""))

        assert best_pos_smooth is not None
        tree.set_positions_from_tensor(best_pos_smooth, dense_active)
        return n_ibs_done + 1

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(
        self,
        anchors_bed: str,
        pet_clusters_bedpe: str,
        singletons_bedpe: Optional[str] = None,
        output_prefix: str = "output",
        factor: int = 0,
    ):
        """End-to-end pipeline: load → build → optimise → save."""
        self.set_contact_data(anchors_bed, pet_clusters_bedpe, singletons_bedpe,
                              factor=factor)

        if singletons_bedpe and os.path.exists(singletons_bedpe):
            self.load_heatmap(singletons_bedpe)

        self.create_tree_genome()
        self.reconstruct_clusters_heatmap()
        self.reconstruct_clusters_arcs_distances()
        self.save(output_prefix)

    # ── Output ────────────────────────────────────────────────────────────────

    def save(self, prefix: str = "output"):
        """Write per-chromosome 3D coordinate files in BED4-like format."""
        os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
        for chrom, tree in self.trees.items():
            path = f"{prefix}_{chrom}.3d"
            with open(path, "w") as fh:
                fh.write("chrom\tstart\tend\tx\ty\tz\tlevel\n")
                for c in tree.clusters:
                    fh.write(f"{c.chrom}\t{c.start}\t{c.end}\t"
                             f"{c.x:.6f}\t{c.y:.6f}\t{c.z:.6f}\t{c.level}\n")
            print(f"Saved {path}")

    def save_hcm(self, prefix: str = "output"):
        """Write anchor positions in HCM format (matching cudaMMC output)."""
        for chrom, tree in self.trees.items():
            path = f"{prefix}_{chrom}.hcm"
            with open(path, "w") as fh:
                for idx in tree.anchors_idx:
                    c = tree.clusters[idx]
                    fh.write(f"{c.genomic_pos}\t{c.x:.6f}\t{c.y:.6f}\t{c.z:.6f}\n")
            print(f"Saved {path}")

    def save_cif(self, prefix: str = "output", anchors_only: bool = True,
                 smooth: bool = True, smooth_sample_kb: float = 1.0):
        """Write per-chromosome mmCIF files suitable for 3D structure viewers.

        anchors_only: if True write only anchor (level-4) beads; otherwise all beads.
        smooth: if True and anchors_only, interpolate a Catmull-Rom spline
            through anchor positions and sample at smooth_sample_kb kb intervals,
            producing a smooth backbone like cudaMMC's smooth phase output.
        smooth_sample_kb: genomic sampling interval in kb for spline output.
        """
        _CIF_HEADER = """\
data_3dnome
#
_entry.id 3dgnome
#
_audit_conform.dict_name       mmcif_pdbx.dic
_audit_conform.dict_version    5.296
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
"""
        os.makedirs(os.path.dirname(prefix) if os.path.dirname(prefix) else ".", exist_ok=True)
        for chrom, tree in self.trees.items():
            path = f"{prefix}_{chrom}.cif"

            if anchors_only and smooth and len(tree.anchors_idx) >= 2:
                coords = self._smooth_cif_coords(tree, smooth_sample_kb)
            elif anchors_only:
                coords = [(tree.clusters[i].x,
                           tree.clusters[i].y,
                           tree.clusters[i].z)
                          for i in tree.anchors_idx]
            else:
                coords = [(c.x, c.y, c.z) for c in tree.clusters]

            with open(path, "w") as fh:
                fh.write(_CIF_HEADER)
                for i, (x, y, z) in enumerate(coords, start=1):
                    fh.write(
                        f"ATOM {i} C CA . ALA A 1 {i} ? "
                        f"{x:.4f} {y:.4f} {z:.4f} 1.00 99.99 A\n"
                    )
            print(f"Saved {path} ({len(coords)} atoms)")

    def _smooth_cif_coords(self, tree: "ChromosomeTree",
                           sample_kb: float = 1.0
                           ) -> list:
        """
        Sample a Catmull-Rom spline through anchor positions at genomic
        intervals of `sample_kb` kb, matching cudaMMC smooth output density.
        """
        anchors = [tree.clusters[i] for i in tree.anchors_idx]
        pts = [(c.x, c.y, c.z) for c in anchors]

        # Total genomic span → number of output samples
        g_start = anchors[0].genomic_pos
        g_end = anchors[-1].genomic_pos
        span_kb = max(g_end - g_start, 1) / 1000.0
        n_out = max(len(pts), int(round(span_kb / sample_kb)) + 1)

        return interpolate_children_spline(pts, n_out)
