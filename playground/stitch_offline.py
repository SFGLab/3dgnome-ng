"""Apply the boundary stitch pass to a finished structure and write the result.

The pass normally runs inside `reconstruct` after every chain of a chromosome is done. This
applies the same function to an existing cif so its effect can be measured on a structure
that already exists, without paying for the reconstruction again. Beads are grouped into
blocks by rebuilding the cluster tree and assigning each bead to the block of the anchor whose
genomic range contains or immediately precedes it, which is how densify placed it.

    python playground/stitch_offline.py <config> <data_dir> <chrom> <in.cif> <out.cif>
"""

import bisect
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.pipeline.stitch import stitch_blocks  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

cfg, data_dir, chrom, cif_in, cif_out = sys.argv[1:6]
s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
s.data_dir = data_dir
s.use_boundary_stitch = True

chrs, bed = parse_chrs_arg(chrom)
state = build_state(s, ContactData.from_files(s, chrs, bed), chrs, bed)
cl = state.clusters
anchors = sorted((i for i, c in enumerate(cl) if c.level == Level.ANCHOR), key=lambda i: cl[i].start)
a_start = [cl[i].start for i in anchors]
a_block = [cl[i].parent for i in anchors]

header: list[str] = []
beads: list[BeadOut] = []
for ln in open(cif_in):
    if not ln.startswith("ATOM"):
        header.append(ln)
        continue
    r = ln.split()
    beads.append(BeadOut(int(r[16]), int(r[17]), float(r[10]), float(r[11]), float(r[12]), r[18]))  # type: ignore[arg-type]

blocks: dict[int, list[BeadOut]] = {}
for b in beads:
    k = bisect.bisect_right(a_start, b.start) - 1
    blocks.setdefault(a_block[max(k, 0)], []).append(b)
block_list = list(blocks.values())
print(f"[stitch] {len(beads)} beads into {len(block_list)} blocks")

out = stitch_blocks(block_list, s)
moved = sorted((b for bl in out for b in bl), key=lambda b: b.start)
before = np.array([[b.x, b.y, b.z] for b in beads])
after = np.array([[b.x, b.y, b.z] for b in moved])
print(f"[stitch] mean bead displacement {np.linalg.norm(after - before, axis=1).mean():.2f}")

with open(cif_out, "w") as fh:
    fh.writelines(header)
    for i, b in enumerate(moved, 1):
        comp = "ALA" if b.kind == "anchor" else "GLY"
        fh.write(f"ATOM {i} C CA . {comp} A 1 {i} ? {b.x} {b.y} {b.z} 1.00 99.99 C {b.start} {b.end} {b.kind}\n")
print(f"[stitch] wrote {cif_out}")
