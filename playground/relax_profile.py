"""Where the cross block relaxation spends its time.

The MC step cost is two local score evaluations. The chain term looks at a bead's two
neighbours, a constant. The excluded volume term scans every bead in the structure, so the step
cost grows with the structure. This times both scorers directly, on prefixes of a real
structure, and counts how many beads are actually inside the excluded volume radius, which is
what a neighbour list would visit instead.

    python playground/relax_profile.py <cif>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba.terms import _local_excl_nb, local_smooth_nb  # noqa: E402

cif = sys.argv[1]
REPS = 2000

rows = [ln.split() for ln in open(cif) if ln.startswith("ATOM")]
P = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float64)
S = np.array([int(r[16]) for r in rows])
P = P[np.argsort(S)]
bond = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
r0 = 1.5 * bond
print(f"{len(P):,} beads, median bond {bond:.2f}, excluded volume radius {r0:.2f}\n")

# warm the JIT
dtn = np.full(len(P) - 1, bond)
_local_excl_nb(P[:100], 5, r0, 10.0, 1)
local_smooth_nb(P[:100], dtn[:99], 5, 100, 10.0, 10.0, 10.0, 1.0, 1.0)

print(f"{'beads':>8s} {'chain us':>9s} {'EV us':>9s} {'EV share':>9s} {'step us':>8s} {'in radius':>10s} {'scanned/useful':>15s}")
rng = np.random.default_rng(0)
per_bead = []
for n in (2_000, 5_000, 10_000, 20_000, len(P)):
    if n > len(P):
        continue
    pos = np.ascontiguousarray(P[:n])
    d = np.ascontiguousarray(dtn[: n - 1])
    ps = rng.integers(1, n - 1, REPS)

    t = time.perf_counter()
    for p in ps:
        local_smooth_nb(pos, d, int(p), n, 10.0, 10.0, 10.0, 1.0, 1.0)
    t_chain = (time.perf_counter() - t) / REPS * 1e6

    t = time.perf_counter()
    for p in ps:
        _local_excl_nb(pos, int(p), r0, 10.0, 1)
    t_ev = (time.perf_counter() - t) / REPS * 1e6

    nb = np.array([len(x) for x in KDTree(pos).query_ball_point(pos[::37], r0)]) - 1
    step = 2 * (t_chain + t_ev)
    per_bead.append(t_ev / n * 1000)
    print(
        f"{n:>8,d} {t_chain:>9.2f} {t_ev:>9.2f} {100 * t_ev / (t_chain + t_ev):>8.0f}% "
        f"{step:>8.1f} {nb.mean():>10.1f} {n / max(nb.mean(), 1):>14,.0f}x",
        flush=True,
    )

print(f"\nexcluded volume cost per 1000 beads: {np.mean(per_bead):.2f} us, spread {np.std(per_bead):.2f}")
print("A flat figure here means the scan is linear in the structure, so the step cost is set by")
print("how big the chromosome is rather than by how many beads are close enough to matter.")

n = len(P)
step_us = 2 * (t_chain + t_ev)  # type: ignore[possibly-undefined]
print(f"\nAt {n:,} beads a step costs about {step_us:.0f} us, so the 19,788 s run was roughly")
print(f"{19788 / (step_us * 1e-6) / 1e6:.0f}M steps, about {19788 / (step_us * 1e-6) / n:.0f} moves per bead.")
