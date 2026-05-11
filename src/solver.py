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

from .data_loading import load_anchors, load_pet_clusters, load_singletons, mark_arcs
from .data_structures import Anchor, Cluster, InteractionArc
from .distances import (
    count_to_distance,
    freq_to_distance_intra,
    freq_to_distance_inter,
    genomic_length_to_distance,
)
from .heatmap import build_singleton_heatmap, normalize_heatmap, heatmap_to_expected_distances
from .mc import monte_carlo_heatmap, monte_carlo_arcs, monte_carlo_arcs_smooth
from .scores import score_arcs, score_structure_smooth, score_orientation
from .regions import Region, build_filter, chrom_included, filter_anchors, default_regions
from .settings import Settings
from .tree import ChromosomeTree, interpolate_children_spline


def _rescale_positions(pos: torch.Tensor,
                       chain_lengths: torch.Tensor,
                       arc_starts: torch.Tensor,
                       arc_ends: torch.Tensor,
                       arc_exp: torch.Tensor,
                       min_ratio: float = 2.0) -> float:
    """
    Return a scale factor S such that pos/S has typical inter-bead distances
    matching chain/arc expected distances.  Returns 1.0 if already compatible.

    Uses consecutive chain distances as the primary reference (always present),
    with arc distances as fallback.  pos/S is passed to arc+smooth MC and
    the output stays at that scale (not multiplied back) — the final coordinates
    are in arc/chain distance units, matching the reference cudaMMC output scale.
    """
    # Primary: compare median consecutive distance to median chain length
    if pos.shape[0] > 1 and chain_lengths.numel() > 0:
        with torch.no_grad():
            actual = (pos[1:] - pos[:-1]).norm(dim=1).median().item()
            target = chain_lengths.median().item()
        if target > 0:
            ratio = actual / target
            if ratio >= min_ratio:
                return ratio

    # Fallback: arc distances
    mask = arc_exp > 0
    if mask.any():
        with torch.no_grad():
            actual = (pos[arc_starts[mask]] - pos[arc_ends[mask]]).norm(dim=1).median().item()
            expected = arc_exp[mask].median().item()
        if expected > 0:
            ratio = actual / expected
            if ratio >= min_ratio:
                return ratio

    return 1.0


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

        if singletons_bedpe and os.path.exists(singletons_bedpe):
            print("Loading singletons...")
            raw_arcs += load_singletons(singletons_bedpe, factor)

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

    def _calc_arc_expected_distances(self, chrom: str,
                                      indices: List[int]) -> torch.Tensor:
        """
        Build arc expected-distance tensor for a set of cluster indices.

        Mirrors calcAnchorExpectedDistancesHeatmap + calcAnchorExpectedDistances:
          - initialise all pairs to -1 (repulsion)
          - for each arc set expected = countToDistance(arc.score)
          - multi-factor arcs: summary arc (factor=-1) gets score=0 → large distance
        """
        tree = self.trees[chrom]
        arcs = self.arcs_by_chr.get(chrom, [])
        s = self.settings

        n = len(indices)
        idx_map = {ci: li for li, ci in enumerate(indices)}

        expected = torch.full((n, n), -1.0, dtype=torch.float32,
                               device=self.device)

        for arc in arcs:
            # map global anchor indices to local indices
            ai = tree.anchors_idx[arc.start] if arc.start < len(tree.anchors_idx) else -1
            bi = tree.anchors_idx[arc.end] if arc.end < len(tree.anchors_idx) else -1
            li = idx_map.get(ai, -1)
            lj = idx_map.get(bi, -1)
            if li < 0 or lj < 0:
                continue

            dist = count_to_distance(
                arc.score,
                s.count_dist_a, s.count_dist_scale,
                s.count_dist_shift, s.count_dist_base_level,
            )
            expected[li, lj] = dist
            expected[lj, li] = dist

        return expected

    def _arc_tensors(self, chrom: str
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (arc_starts, arc_ends, arc_expected) tensors for CPU MC phases."""
        tree = self.trees[chrom]
        arcs = self.arcs_by_chr.get(chrom, [])
        s = self.settings
        n = len(tree.anchors_idx)

        starts, ends, expected = [], [], []
        for arc in arcs:
            ai = arc.start
            bi = arc.end
            if ai >= n or bi >= n:
                continue
            dist = count_to_distance(
                arc.score,
                s.count_dist_a, s.count_dist_scale,
                s.count_dist_shift, s.count_dist_base_level,
            )
            starts.append(ai)
            ends.append(bi)
            expected.append(dist)

        if not starts:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty, torch.zeros(0, dtype=torch.float32, device=self.device)

        arc_starts = torch.tensor(starts, dtype=torch.long, device=self.device)
        arc_ends = torch.tensor(ends, dtype=torch.long, device=self.device)
        arc_exp = torch.tensor(expected, dtype=torch.float32, device=self.device)
        return arc_starts, arc_ends, arc_exp

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

            pos = tree.anchor_positions_tensor(device=str(self.device))

            if chrom in self.heatmap_expected:
                exp = self.heatmap_expected[chrom]
            else:
                exp = self._genomic_expected_matrix(chrom, tree)

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

    # ── Phase 2+3: arc / smooth MC ────────────────────────────────────────────

    def reconstruct_clusters_arcs_distances(self):
        """Run arc-spring MC and smooth MC for each chromosome."""
        for chrom, tree in self.trees.items():
            n = len(tree.anchors_idx)
            if n == 0:
                continue

            print(f"\n[Arcs MC] {chrom}")
            pos = tree.anchor_positions_tensor(device=str(self.device))
            arc_starts, arc_ends, arc_exp = self._arc_tensors(chrom)
            fixed = torch.zeros(n, dtype=torch.bool, device=self.device)

            # add small noise to break symmetry
            pos = pos + self.settings.noise_size_small * torch.randn_like(pos)

            # set dist_to_next from genomic spans
            for i in range(n - 1):
                c1 = tree.clusters[tree.anchors_idx[i]]
                c2 = tree.clusters[tree.anchors_idx[i + 1]]
                gap = max(0, c2.start - c1.end)
                c1.dist_to_next = genomic_length_to_distance(
                    gap,
                    self.settings.genomic_dist_scale,
                    self.settings.genomic_dist_power,
                    self.settings.genomic_dist_base,
                )

            # Compute chain lengths (kb-scale after the bp→kb fix in distances.py)
            chain_lengths = tree.chain_lengths_tensor(device=str(self.device))
            orientations = [
                tree.clusters[tree.anchors_idx[i]].orientation
                for i in range(n)
            ]

            # Heatmap MC leaves positions at a large scale (heatmap expected units)
            # while arc/chain spring targets are much smaller.  Rescale positions
            # into arc/chain distance units before MC.  The output stays at that
            # scale — this is the correct final coordinate unit matching reference.
            arc_scale = _rescale_positions(pos, chain_lengths,
                                           arc_starts, arc_ends, arc_exp)
            if arc_scale != 1.0:
                print(f"  Rescaling positions ÷{arc_scale:.1f} to match arc/chain distance scale")
                pos = pos / arc_scale

            # Phase 2 - arc spring MC
            # NOTE: original cudaMMC runs arcs MC on CPU (sequential per-bead
            # Gauss-Seidel).  All tensors here live on self.device (GPU) so the
            # tensor math runs on GPU; the Python `for bead_idx` control loop
            # is unavoidably CPU.  A full GPU kernel would require atomic ops or
            # graph colouring — not implemented.
            if arc_starts.shape[0] > 0:
                pos = monte_carlo_arcs(
                    pos, arc_starts, arc_ends, arc_exp, chain_lengths, fixed,
                    self.settings, verbose=True,
                )

            # Phase 3 - smooth MC  (same GPU-tensor / CPU-loop note as above)
            print(f"[Smooth MC] {chrom}")
            pos = monte_carlo_arcs_smooth(
                pos, arc_starts, arc_ends, arc_exp,
                chain_lengths, orientations, fixed,
                self.settings, verbose=True,
            )

            tree.set_anchor_positions_from_tensor(pos)
            tree._propagate_positions_up()

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
