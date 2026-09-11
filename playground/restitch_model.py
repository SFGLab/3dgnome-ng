"""Run the boundary stitch and the cross block relaxation on a finished model.

    python playground/restitch_model.py MODEL.cif CONFIG.ini [--out OUT.cif] [--no-relax]

The pipeline runs both passes at the end of a reconstruction, so a model produced before a
change to either one cannot be compared against a model produced after it without paying for
the whole run again. Both passes take a chromosome's per block bead lists and return the same,
and neither reads anything a finished model does not carry, so they can be replayed on a cif.

The one thing a cif does not carry is which block a bead came from. It is recovered from the
densification rule. Between two consecutive anchors of one block, densify inserts
`round(span / target_bp_per_subanchor) - 1` subanchors, so a consecutive anchor pair further
apart than one and a half times that target always has a subanchor between it. A consecutive
anchor pair wider than that with nothing between it can therefore only be a block boundary.
Anchors closer than the target are ambiguous and are left inside the block they follow, which
merges a block that begins within a target's width of the previous one's end. That costs a
boundary spring rather than inventing one.

Reports the boundary distances against the structure's own within block curve before and
after, which is the objective the stitch minimises, and the cross block bead contacts, which
is the one the relaxation minimises.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.io import write_cif
from gnome3d.pipeline.relax import cross_block_contacts, relax_blocks
from gnome3d.pipeline.stitch import stitch_blocks, within_block_curve
from gnome3d.settings import Settings
from gnome3d.types import BeadOut


def read_cif(path: str) -> list[BeadOut]:
    """The beads of a model written by `io.write_cif`, in genomic order."""
    beads: list[BeadOut] = []
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        beads.append(
            BeadOut(int(f[-3]), int(f[-2]), float(f[10]), float(f[11]), float(f[12]), f[-1])  # type: ignore[arg-type]
        )
    return sorted(beads, key=lambda b: b.start)


def split_blocks(beads: list[BeadOut], target_bp: int) -> list[list[BeadOut]]:
    """The per block bead lists, recovered from the densification rule. See the module doc."""
    mid = np.array([(b.start + b.end) // 2 for b in beads], dtype=np.int64)
    anchor = np.array([b.kind == "anchor" for b in beads], dtype=np.bool_)
    wide = np.diff(mid) > 1.5 * target_bp
    cut = np.where(anchor[:-1] & anchor[1:] & wide)[0] + 1
    edges = np.concatenate(([0], cut, [len(beads)]))
    return [beads[edges[i] : edges[i + 1]] for i in range(edges.size - 1)]


def boundary_report(blocks: list[list[BeadOut]], label: str) -> None:
    """Each boundary distance over what the structure's own interior realises at that gap."""
    curve = within_block_curve(blocks)
    if curve is None:
        print(f"  {label}: no usable within block curve")
        return
    ratio: list[float] = []
    dist: list[float] = []
    for a, b in zip(blocks[:-1], blocks[1:], strict=True):
        la = [x for x in a if x.kind == "anchor"]
        fb = [x for x in b if x.kind == "anchor"]
        if not la or not fb:
            continue
        u, v = la[-1], fb[0]
        d = float(np.linalg.norm(np.array([u.x, u.y, u.z]) - np.array([v.x, v.y, v.z])))
        gap = abs(((v.start + v.end) // 2) - ((u.start + u.end) // 2))
        dist.append(d)
        ratio.append(d / curve(gap))
    r = np.array(ratio)
    print(
        f"  {label}: {r.size} boundaries, realised over target"
        f" median {np.median(r):.2f}, q95 {np.quantile(r, 0.95):.2f}, max {r.max():.2f}"
        f" | longest boundary {max(dist):.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--stitch-iter", type=int, default=0, help="override boundary_stitch_max_iter")
    ap.add_argument(
        "--relax-executor",
        default="",
        help="override mc_executor_smooth, which is what the relaxation picks its kernel from. "
        "The pass is one chain, so batch puts it on the JAX kernel at that kernel's worst case",
    )
    ap.add_argument(
        "--relax-window",
        type=int,
        default=-2,
        help="override relax_local_window. -1 lets every subanchor move, which on a whole "
        "chromosome is hours. A small window keeps only the beads that touch another block "
        "and that many chain neighbours either side",
    )
    a = ap.parse_args()

    s = Settings()
    s.load_ini(a.config)
    if a.stitch_iter > 0:
        s.boundary_stitch_max_iter = a.stitch_iter
    if a.relax_window > -2:
        s.relax_local_window = a.relax_window
    if a.relax_executor:
        s.mc_executor_smooth = a.relax_executor
    beads = read_cif(a.model)
    blocks = split_blocks(beads, int(s.target_bp_per_subanchor))
    sizes = np.array([len(b) for b in blocks])
    print(f"{len(beads):,} beads, {len(blocks)} blocks, median {np.median(sizes):.0f} beads")

    bond = float(
        np.median(
            np.linalg.norm(
                np.diff(np.array([[b.x, b.y, b.z] for b in beads]), axis=0),
                axis=1,
            )
        )
    )
    print(f"median chain bond {bond:.3f}")
    boundary_report(blocks, "before ")
    print(f"  before : cross block contacts within one bond {cross_block_contacts(blocks, bond)[0]}")

    if not a.no_stitch:
        t = time.time()
        blocks = stitch_blocks(blocks, s)
        print(f"\nstitch {time.time() - t:.1f}s")
        boundary_report(blocks, "stitched")
        print(
            "  stitched: cross block contacts within one bond "
            f"{cross_block_contacts(blocks, bond)[0]}"
        )

    if not a.no_relax:
        t = time.time()
        blocks = relax_blocks(blocks, s)
        print(f"\nrelax {time.time() - t:.1f}s")
        boundary_report(blocks, "relaxed ")
        print(
            "  relaxed : cross block contacts within one bond "
            f"{cross_block_contacts(blocks, bond)[0]}"
        )

    out = a.out or str(Path(a.model).with_suffix("")) + ".restitched.cif"
    flat = sorted((b for blk in blocks for b in blk), key=lambda b: b.start)
    write_cif(out, flat, entry_id=Path(out).stem)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
