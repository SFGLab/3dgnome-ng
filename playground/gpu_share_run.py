"""Reconstruct one region with the production config and save the beads, the wall time and the
GPU memory pool of the process. Several of these run at once on one card to check that
instances share it.

    python playground/gpu_share_run.py <tree_root> <out.npz> [region] [n_structures]
"""

import os
import sys
import time

root = sys.argv[1]
sys.path.insert(0, root)
os.chdir(root)

import numpy as np  # noqa: E402

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from validation.core.config import settings_for_cell  # noqa: E402
from validation.core.ensemble import run_ensemble  # noqa: E402

region = sys.argv[3] if len(sys.argv) > 3 else "chr1:11500000-12600000"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 3

t0 = time.time()
s = settings_for_cell("GM12878", "data")
chrs, bed = parse_chrs_arg(region)
data = ContactData.from_files(s, chrs, bed)
ens = run_ensemble(s, data, chrs, bed, n)
wall = time.time() - t0

import jax  # noqa: E402

stats = jax.devices()[0].memory_stats() or {}
pool = stats.get("bytes_limit", 0)
peak = stats.get("peak_bytes_in_use", 0)
beads = np.array([[(b.start, b.end, b.x, b.y, b.z) for b in e] for e in ens], dtype=np.float64)
np.savez(sys.argv[2], beads=beads, wall=wall, pool=pool, peak=peak, fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""))
print(f"{sys.argv[2]}: {beads.shape[0]} structures of {beads.shape[1]} beads in {wall:.0f} s, pool {pool / 2**30:.1f} GiB, peak in use {peak / 2**30:.2f} GiB")
