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

from .data_loading import load_anchors, load_pet_clusters, mark_arcs
from .data_structures import Anchor, Cluster, InteractionArc
from .distances import count_to_distance, genomic_length_to_distance
from .heatmap import build_singleton_heatmap, normalize_heatmap, heatmap_to_expected_distances
from .mc import monte_carlo_heatmap, monte_carlo_arcs, monte_carlo_arcs_smooth
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

    def load_heatmap(self, singletons_bedpe: str):
        """Build and normalise singleton heatmaps for all chromosomes.

        build_singleton_heatmap runs on CPU (file I/O + Python loop).
        Normalisation and freq→distance conversion run on GPU:
          N=26603 float32 ≈ 2.83 GB; float16 result ≈ 1.41 GB.
        """
        for chrom, anchors in self.anchors_by_chr.items():
            print(f"  Building heatmap for {chrom} ({len(anchors)} anchors)...")
            raw = build_singleton_heatmap(singletons_bedpe, anchors)  # CPU float32
            raw = raw.to(self.device)   # → GPU before heavy matrix ops
            norm = normalize_heatmap(raw, anchors, self.settings.diagonal_size)  # GPU float32
            del raw
            exp = heatmap_to_expected_distances(
                norm,
                self.settings.freq_dist_scale,
                self.settings.freq_dist_power,
            )  # GPU float16 (~1.41 GB for N=26603)
            del norm
            self.heatmap_expected[chrom] = exp
            del exp

    # ── Tree construction ─────────────────────────────────────────────────────

    def create_tree_genome(self):
        """Build hierarchical bead-spring models for every chromosome."""
        print("Building hierarchical models...")
        for chrom, anchors in self.anchors_by_chr.items():
            arcs = self.arcs_by_chr.get(chrom, [])
            tree = ChromosomeTree(chrom, anchors, arcs, self.settings)
            tree.init_positions_linear()
            self.trees[chrom] = tree
            print(f"  {chrom}: {len(tree.clusters)} clusters, "
                  f"{len(tree.anchors_idx)} anchors")

    # ── Arc expected distances ────────────────────────────────────────────────

    def _arc_tensors_ib(self, chrom: str, anchor_cidxs: List[int]
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build (arc_starts, arc_ends, arc_expected) for ARC SPRING PAIRS ONLY within one IB.

        cudaMMC arc MC uses calcScoreDistancesActiveRegion (LooperSolver.cpp:3086),
        which scores arc spring pairs only — NOT all-pair repulsion.
        Repulsion was in calcAnchorExpectedDistancesHeatmap (heatmap phase), not arcs phase.

        Local indices (0..n-1) into the anchor_cidxs list.
        """
        n = len(anchor_cidxs)
        if n < 2:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=self.device)

        idx_map = {ci: li for li, ci in enumerate(anchor_cidxs)}
        s = self.settings
        tree = self.trees[chrom]

        arc_s_list, arc_e_list, arc_exp_list = [], [], []
        for arc in self.arcs_by_chr.get(chrom, []):
            ai = tree.anchors_idx[arc.start] if arc.start < len(tree.anchors_idx) else -1
            bi = tree.anchors_idx[arc.end]   if arc.end   < len(tree.anchors_idx) else -1
            li = idx_map.get(ai, -1)
            lj = idx_map.get(bi, -1)
            if li < 0 or lj < 0 or li == lj:
                continue
            dist = count_to_distance(
                arc.score,
                s.count_dist_a, s.count_dist_scale,
                s.count_dist_shift, s.count_dist_base_level,
            )
            # Store as (min, max) so score_arcs_single finds both orderings
            lo, hi = (li, lj) if li < lj else (lj, li)
            arc_s_list.append(lo)
            arc_e_list.append(hi)
            arc_exp_list.append(dist)

        if not arc_s_list:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=self.device)

        starts = torch.tensor(arc_s_list, dtype=torch.long, device=self.device)
        ends   = torch.tensor(arc_e_list, dtype=torch.long, device=self.device)
        exp    = torch.tensor(arc_exp_list, dtype=torch.float32, device=self.device)
        return starts, ends, exp

    def _chain_lengths_ib(self, chrom: str, anchor_cidxs: List[int]
                           ) -> torch.Tensor:
        """
        Expected linker lengths for consecutive anchors within one IB.
        Uses center-to-center genomic distance (matching tree.chain_lengths_tensor).
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

    # ── Phase 1: heatmap MC ───────────────────────────────────────────────────

    def reconstruct_clusters_heatmap(self):
        """
        Run vectorised GPU heatmap Monte Carlo for each chromosome.

        Each outer step proposes displacements for ALL anchors simultaneously
        (Jacobi), computes per-bead score deltas in chunked GPU matmuls, and
        accepts/rejects per-bead — no Python loop over beads.
        """
        for chrom, tree in self.trees.items():
            n = len(tree.anchors_idx)
            if n == 0:
                continue

            s = self.settings
            print(f"\n[Heatmap MC] {chrom}  ({n} anchors, "
                  f"milestone_steps={s.milestone_steps_heatmap})")

            if chrom in self.heatmap_expected:
                exp = self.heatmap_expected[chrom]
            else:
                exp = self._genomic_expected_matrix(chrom, tree)

            # Initial placement: uniform random in a sphere scaled to heatmap units.
            # cudaMMC inherits good positions from hierarchical placement; we start
            # from scratch, so beads must begin at the right scale (~median expected
            # distance) otherwise step_size=0.75 can never reach the 100-500 unit targets.
            valid_mask = (exp > 1e-3) & exp.isfinite()
            R = exp[valid_mask].float().median().item() if valid_mask.any() else 100.0
            dirs = torch.randn(n, 3, device=self.device)
            dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            radii = torch.rand(n, device=self.device).pow(1.0 / 3.0) * R
            pos = (dirs * radii.unsqueeze(1)).float()

            fixed = torch.zeros(n, dtype=torch.bool, device=self.device)

            pos = monte_carlo_heatmap(
                pos, exp, fixed, self.settings,
                verbose=True,
            )

            tree.set_anchor_positions_from_tensor(pos)
            tree._propagate_positions_up()

    def _genomic_expected_matrix(self, chrom: str,
                                   tree: ChromosomeTree) -> torch.Tensor:
        """Fallback: build expected-distance matrix from genomic positions."""
        s = self.settings
        n = len(tree.anchors_idx)
        gpos = torch.tensor(
            [tree.clusters[i].genomic_pos for i in tree.anchors_idx],
            dtype=torch.float32, device=self.device,
        )
        diff_kb = (gpos.unsqueeze(1) - gpos.unsqueeze(0)).abs() / 1000.0  # bp → kb
        exp = s.genomic_dist_base + s.genomic_dist_scale * diff_kb.clamp(min=1e-3) ** s.genomic_dist_power
        # zero out diagonal
        exp.fill_diagonal_(0.0)
        return exp

    # ── Phase 2+3: per-IB arc / smooth MC ────────────────────────────────────

    def reconstruct_clusters_arcs_distances(self):
        """
        Run per-IB arc MC + smooth MC for each chromosome.

        Mirrors cudaMMC reconstructClustersArcsDistances (LooperSolver.cpp:2579-2702)
        and reconstructClusterArcsDistances (cpp:2735-2875):
          - All anchors in IB reinitialised to IB centroid + small noise each attempt.
          - MC run simulationStepsLevelAnchor (default 5) times; best score kept.
            (cudaMMC cpp:2836: for (int k = 0; k < steps; ++k) { ... best_score update })
        """
        for chrom, tree in self.trees.items():
            n_total = len(tree.anchors_idx)
            if n_total == 0:
                continue

            print(f"\n[Arcs+Smooth MC] {chrom}  ({n_total} total anchors)")

            # After heatmap MC on all anchors, propagate means up to IBs / segments.
            # IB positions then serve as starting points for per-IB MC.
            tree._propagate_positions_up()

            n_ibs_done = 0
            for root_ci in (i for i, c in enumerate(tree.clusters) if c.level == 1):
                for seg_ci in tree.clusters[root_ci].children:
                    seg_c = tree.clusters[seg_ci]
                    for ib_ci in seg_c.children:
                        ib_c = tree.clusters[ib_ci]
                        anchor_cidxs = ib_c.children  # level-4 cluster indices
                        n_ib = len(anchor_cidxs)

                        if n_ib < 2:
                            continue  # single anchor — no internal MC needed

                        # IB centroid: mean of anchor positions after heatmap MC.
                        # cudaMMC LooperSolver.cpp:2625:
                        #   clusters[active_region[j]].pos = clusters[ib].pos
                        ib_pos = torch.tensor(
                            [ib_c.x, ib_c.y, ib_c.z],
                            dtype=torch.float32, device=self.device,
                        )

                        fixed = torch.zeros(n_ib, dtype=torch.bool, device=self.device)

                        arc_s, arc_e, arc_exp = self._arc_tensors_ib(chrom, anchor_cidxs)
                        chain_lengths = self._chain_lengths_ib(chrom, anchor_cidxs)
                        orientations = [
                            tree.clusters[ci].orientation for ci in anchor_cidxs
                        ]

                        n_arc = len(arc_exp)
                        print(f"  IB {n_ibs_done}: {n_ib} anchors, {n_arc} arc pairs")

                        # cudaMMC reconstructClusterArcsDistances (cpp:2836):
                        #   for (int k = 0; k < steps; ++k) — multiple restarts, keep best.
                        #   steps = simulationStepsLevelAnchor (default 5) for arc phase,
                        #          simulationStepsLevelSubanchor (default 5) for smooth phase.
                        s = self.settings
                        n_restarts_arcs = s.simulation_steps_level_anchor
                        n_restarts_smooth = s.simulation_steps_level_subanchor

                        from .scores import score_arcs as _score_arcs
                        from .scores import score_structure_smooth as _score_smooth

                        best_pos_arcs: Optional[torch.Tensor] = None
                        best_score_arcs = float("inf")

                        for _k in range(n_restarts_arcs):
                            # cudaMMC cpp:2844-2848: reset to initial_structure + small noise
                            pos = ib_pos.unsqueeze(0).expand(n_ib, 3).clone()
                            pos = pos + s.noise_size_small * torch.randn_like(pos)

                            pos = monte_carlo_arcs(
                                pos, arc_s, arc_e, arc_exp, chain_lengths, fixed,
                                s, verbose=False,
                            )

                            # cudaMMC cpp:2864: if (score < best_score) save best
                            sc = _score_arcs(pos, arc_s, arc_e, arc_exp,
                                             s.k_spring, s.k_spring_repulsion).item()
                            if sc < best_score_arcs:
                                best_score_arcs = sc
                                best_pos_arcs = pos.clone()

                        pos = best_pos_arcs

                        best_pos_smooth: Optional[torch.Tensor] = None
                        best_score_smooth = float("inf")

                        for _k in range(n_restarts_smooth):
                            # cudaMMC cpp:2844-2848: reset to initial_structure + noise
                            pos_in = pos + s.noise_size_small * torch.randn_like(pos)

                            pos_out = monte_carlo_arcs_smooth(
                                pos_in, arc_s, arc_e, arc_exp,
                                chain_lengths, orientations, fixed,
                                s, verbose=False,
                            )

                            sc = _score_smooth(pos_out, chain_lengths,
                                               s.k_chain, s.k_angular).item()
                            if sc < best_score_smooth:
                                best_score_smooth = sc
                                best_pos_smooth = pos_out.clone()

                        tree.set_positions_from_tensor(best_pos_smooth, anchor_cidxs)
                        n_ibs_done += 1

            tree._propagate_positions_up()
            print(f"  Done: {n_ibs_done} IBs processed")

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
