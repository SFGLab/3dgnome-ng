"""Trace what each end of run pass does to the compartments of one finished structure.

Runs the placement and the per block chains in process, which is the state the end of run
passes start from, with the true block membership rather than the one the densification rule
recovers from a cif, since that rule splits these blocks about ten times too fine and a
replay on those pieces shreds the within block contacts. Then applies the stitch and the
relaxation, and after each stage reports the compartment saddle over within block and cross block
pairs, how far like and unlike block pairs sit from each other, how many beads touch another
block, and how far beads of each compartment moved.

    python playground/stage_trace.py CONFIG.ini MCOOL TRACK.bedGraph REGION [relax_radius] [keep]

A fifth argument overrides the relaxation's excluded volume radius, and a sixth of `keep`
keeps the compartment term on inside the pass.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg
from gnome3d.pipeline.relax import cross_block_contacts, relax_blocks
from gnome3d.pipeline.stitch import stitch_blocks
from gnome3d.pipeline.coarse import build_state
from gnome3d.pipeline.coarse.stages import build_coarse_dag
from gnome3d.pipeline.ib import ib_node_id
from gnome3d.pipeline.stage import StageKind
from gnome3d.reconstruct import _beads, _block_compartments, pick_executor
from gnome3d.settings import Settings
from gnome3d.types import BeadOut
from playground.saddle_split import split_saddle
from validation.metrics import hic as contacts
from validation.metrics import structure as smetrics
from validation.studies.epigenome import _track_on_bins


def coords(blocks: list[list[BeadOut]]) -> np.ndarray:
    return np.array([[b.x, b.y, b.z] for blk in blocks for b in blk])


def report(
    label: str,
    blocks: list[list[BeadOut]],
    prev: np.ndarray | None,
    cls_beads: np.ndarray,
    cls_block: np.ndarray,
    c_obs: np.ndarray,
    bin_starts: np.ndarray,
    binsize: int,
    track: np.ndarray,
    lab: np.ndarray,
    radius: float,
    bond: float,
) -> np.ndarray:
    xyz = coords(blocks)
    beads = [b for blk in blocks for b in blk]
    c_sim = contacts.simulated_contacts(xyz, np.array([(b.start + b.end) // 2 for b in beads]), bin_starts, binsize, radius)
    sad = split_saddle(c_sim, track, lab)
    cen = np.array([np.mean([[b.x, b.y, b.z] for b in blk], axis=0) for blk in blocks])
    rg = np.array([np.sqrt(np.mean(np.sum((np.array([[b.x, b.y, b.z] for b in blk]) - c) ** 2, axis=1))) for blk, c in zip(blocks, cen, strict=True)])
    n = len(blocks)
    iu = np.triu_indices(n, k=1)
    d = np.linalg.norm(cen[iu[0]] - cen[iu[1]], axis=1) / (rg[iu[0]] + rg[iu[1]])
    like = (cls_block[iu[0]] == cls_block[iu[1]]) & (cls_block[iu[0]] != 0)
    unlike = (cls_block[iu[0]] * cls_block[iu[1]]) < 0
    touching = cross_block_contacts(blocks, bond)[0]
    print(
        f"  {label:<18} saddle within {sad['within'][0]:.2f} cross {sad['cross'][0]:.2f}"
        f" (cross AA {sad['cross'][1]:.2f} BB {sad['cross'][2]:.2f} AB {sad['cross'][3]:.2f})"
        f" | block pair distance over radii sum: like {np.median(d[like]):.2f} unlike {np.median(d[unlike]):.2f},"
        f" overlapping pairs like {int((d[like] < 1).sum())}/{int(like.sum())} unlike {int((d[unlike] < 1).sum())}/{int(unlike.sum())}"
        f" | Rg {smetrics.radius_of_gyration(xyz):.1f} | beads touching another block {touching}"
    )
    if prev is not None:
        move = np.linalg.norm(xyz - prev, axis=1)
        kinds = np.array([b.kind == "anchor" for b in beads])
        print(
            f"  {'':<18} moved, in bonds: A {np.mean(move[cls_beads > 0]) / bond:.2f}"
            f" B {np.mean(move[cls_beads < 0]) / bond:.2f} none {np.mean(move[cls_beads == 0]) / bond:.2f}"
            f" | anchors {np.mean(move[kinds]) / bond:.2f} subanchors {np.mean(move[~kinds]) / bond:.2f}"
            f" | beads moved over one bond {int((move > bond).sum())}"
        )
    return xyz


def main() -> None:
    config, mcool, track_path, region = sys.argv[1:5]
    s = Settings()
    s.load_ini(config)
    for k in ("mc_executor_smooth", "mc_executor_estimate_dist", "mc_executor_arcs"):
        setattr(s, k, "threaded")
    if len(sys.argv) > 5 and float(sys.argv[5]) > 0:
        s.relax_ev_radius = float(sys.argv[5])
    if len(sys.argv) > 6 and sys.argv[6] == "keep":
        s.relax_keep_compartments = True
    print(f"relax radius {s.relax_ev_radius or '1.5 bonds'}, keep compartments {getattr(s, 'relax_keep_compartments', False)}")
    chrom = region.split(":")[0]
    chrs, reg = parse_chrs_arg(region)
    data = ContactData.from_files(s, chrs, reg)
    cache = Path(config).with_suffix(".placement.pkl")
    if cache.is_file():
        blocks = pickle.loads(cache.read_bytes())
    else:
        state = build_state(s, data, chrs, reg)
        dag, ib_sink = build_coarse_dag(state, 0)
        outputs = pick_executor(s).run(dag)
        blocks = [
            _beads(outputs[ib_node_id(ibs.ib_id, StageKind.SMOOTH)])
            for ibs in ib_sink
            if ibs.chr_ == chrom
        ]
        blocks.sort(key=lambda blk: blk[0].start)
        cache.write_bytes(pickle.dumps(blocks))
    beads = [b for blk in blocks for b in blk]
    binsize = 100_000
    c_obs, bin_starts = contacts.observed_hic(mcool, region, binsize, balance=True)
    track = _track_on_bins(track_path, chrom, bin_starts)
    lab = np.full(len(bin_starts), -1, dtype=np.int64)
    for k, blk in enumerate(blocks):
        lo = int(((blk[0].start + blk[0].end) // 2 - bin_starts[0]) // binsize)
        hi = int(((blk[-1].start + blk[-1].end) // 2 - bin_starts[0]) // binsize)
        lab[max(lo, 0) : min(hi, len(lab) - 1) + 1] = k
    classes = _block_compartments(blocks, data.compartments.get(chrom, []))
    cls_beads = np.concatenate(classes)
    cls_block = np.array([int(np.sign(np.sum(np.sign(c)))) for c in classes])
    radius = float(np.median(smetrics.bond_lengths(coords(blocks))))
    bond = radius
    exp = split_saddle(c_obs, track, lab)
    print(f"{len(beads):,} beads, {len(blocks)} blocks, classes A {int((cls_block > 0).sum())} B {int((cls_block < 0).sum())} none {int((cls_block == 0).sum())}; A beads {int((cls_beads > 0).sum())} B beads {int((cls_beads < 0).sum())}")
    print(f"  {'experiment':<18} saddle within {exp['within'][0]:.2f} cross {exp['cross'][0]:.2f} (cross AA {exp['cross'][1]:.2f} BB {exp['cross'][2]:.2f} AB {exp['cross'][3]:.2f})")
    args = (cls_beads, cls_block, c_obs, bin_starts, binsize, track, lab, radius, bond)
    xyz0 = report("after placement", blocks, None, *args)
    stitched = stitch_blocks(blocks, s)
    xyz1 = report("after stitch", stitched, xyz0, *args)
    relaxed = relax_blocks(stitched, s)
    report("after relax", relaxed, xyz1, *args)
    relaxed_only = relax_blocks(blocks, s)
    report("relax, no stitch", relaxed_only, xyz0, *args)


if __name__ == "__main__":
    main()
