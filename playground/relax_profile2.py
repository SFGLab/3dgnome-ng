"""Where the cross block relax actually spends its time, on a real genome scale structure.

On a trio run the pass took an hour and fifty five minutes per structure whatever the input,
once to move two beads out of 129,457. Two things are worth separating before fixing it.

Which kernel it runs. It picks from `mc_executor_smooth`, and production sets that to `batch`,
so a pass that is one single chain runs on the JAX kernel, which is that kernel's worst case:
one chain is latency bound at a few tens of microseconds a step where numba with its cell grid
is a couple. The eden log gives the JAX side directly, 310,050,000 steps in about seven thousand
seconds.

And how many beads it lets move. Every bead that is not an anchor is movable, so a hundred and
seventeen thousand subanchors are proposed for when a few dozen are touching anything.

    python playground/relax_profile2.py <cif>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba.smooth import mc_smooth_numba  # noqa: E402
from gnome3d.pipeline.relax import _relax_settings  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

# eden, structure 1: 310,050,000 steps between the smooth ending and the relax line, about
# 6,996 s apart, on the JAX kernel with one chain.
EDEN_STEPS, EDEN_SECONDS = 310_050_000, 6996.0


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [ln.split() for ln in open(path) if ln.startswith("ATOM")]
    pos = np.array([[float(r[10]), float(r[11]), float(r[12])] for r in rows], dtype=np.float32)
    anchor = np.array([r[5] == "ALA" for r in rows], dtype=np.bool_)
    mid = np.array([int(r[16]) for r in rows])
    o = np.argsort(mid)
    return np.ascontiguousarray(pos[o]), np.ascontiguousarray(anchor[o])


def main() -> None:
    pos, anchor = load(sys.argv[1])
    n = pos.shape[0]
    bonds = np.linalg.norm(np.diff(pos.astype(np.float64), axis=0), axis=1)
    dtn = bonds.astype(np.float32)
    bond = float(np.median(bonds))
    base = Settings()
    base.mc_executor_smooth = "serial"
    r = _relax_settings(base, bond)
    step = float(base.relax_noise) * bond
    print(f"{n:,} beads, {int(anchor.sum()):,} anchors, median bond {bond:.3f}\n")

    print(f"  eden, JAX, one chain: {EDEN_STEPS:,} steps in {EDEN_SECONDS:,.0f}s "
          f"= {EDEN_SECONDS / EDEN_STEPS * 1e6:.1f} us/step\n")

    steps = 200_000
    r.mc_stop_steps_smooth = steps
    r.mc_stop_improvement_smooth = 0.0  # exactly one round
    r.mc_stop_successes_smooth = 10**9
    print(f"  {'movable beads':>14s} {'sec':>8s} {'us/step':>9s} {'vs eden JAX':>12s}")
    for label, frac in (("all subanchors", 1.0), ("a tenth", 0.1), ("a hundredth", 0.01)):
        fixed = anchor.copy()
        if frac < 1.0:
            idx = np.flatnonzero(~anchor)
            keep = idx[:: int(1 / frac)]
            fixed = np.ones(n, dtype=np.bool_)
            fixed[keep] = False
        p = pos.copy()
        mc_smooth_numba(p[:64].copy(), dtn[:63].copy(), fixed[:64].copy(), step, r)  # warm
        p = pos.copy()
        t = time.perf_counter()
        mc_smooth_numba(p, dtn, fixed, step, r)
        w = time.perf_counter() - t
        us = w / steps * 1e6
        print(
            f"  {int((~fixed).sum()):>14,d} {w:>7.1f}s {us:>9.2f} "
            f"{EDEN_SECONDS / EDEN_STEPS * 1e6 / us:>11.1f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
