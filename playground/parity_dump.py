"""Reconstruct one region and dump its bead coordinates, for the byte exactness parity gate.

    python playground/parity_dump.py <tree_root> <out.npz> [region]

Run it once from a worktree at the previous commit and once from the working tree, with every
new flag off, and compare the two files exactly. Both runs use the same config and executor
settings, since batch grouping selects which chain gets which RNG stream.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)

import numpy as np  # noqa: E402

from gnome3d.simulate import run_region  # noqa: E402

region = sys.argv[3] if len(sys.argv) > 3 else "chr1:18288319-20307135"
beads = run_region("data/GM12878/config.ini", region, 1, data_dir="data/GM12878")[0]
arr = np.array([(b.start, b.end, b.x, b.y, b.z) for b in beads], dtype=np.float64)
np.savez(sys.argv[2], beads=arr)
print(f"{root}: {len(beads)} beads -> {sys.argv[2]}")
