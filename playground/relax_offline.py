"""Apply the cross block relaxation to a finished structure and write the result.

Mirrors stitch_offline.py. Beads are grouped into blocks by rebuilding the cluster tree, the
relaxation runs over the whole chromosome with anchors fixed, and the cross block contact count
is printed before and after, which is the gate the pass exists for.

    python playground/relax_offline.py <config> <data_dir> <chrom> <in.cif> <out.cif>
"""

import bisect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
from gnome3d.pipeline.relax import cross_block_contacts, relax_blocks  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

cfg, data_dir, chrom, cif_in, cif_out = sys.argv[1:6]
s = Settings()
if not s.load_ini(cfg):
    raise SystemExit(f"cannot load {cfg}")
s.data_dir = data_dir
s.use_cross_block_relax = True
s.mc_executor_smooth = "serial"

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
pos = np.array([[b.x, b.y, b.z] for b in beads])
bond = float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
print(f"[relax] {len(beads)} beads in {len(block_list)} blocks, bond {bond:.2f}, contacts before {cross_block_contacts(block_list, bond)}", flush=True)
t = time.time()
out = relax_blocks(block_list, s)
print(f"[relax] {time.time() - t:.0f}s, contacts after {cross_block_contacts(out, bond)}", flush=True)
moved = sorted((b for bl in out for b in bl), key=lambda b: b.start)
with open(cif_out, "w") as fh:
    fh.writelines(header)
    for i, b in enumerate(moved, 1):
        comp = "ALA" if b.kind == "anchor" else "GLY"
        fh.write(f"ATOM {i} C CA . {comp} A 1 {i} ? {b.x} {b.y} {b.z} 1.00 99.99 C {b.start} {b.end} {b.kind}\n")
print(f"[relax] wrote {cif_out}")
