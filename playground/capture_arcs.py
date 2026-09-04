"""Capture the real arcs problems, then exit hard.

An arcs problem is its expected-distance matrix and its noised starting anchors, and both exist
the moment the stage begins, so there is no reason to let a reconstruction run. The stage is
threaded, so raising inside one worker does not stop the others; the exit has to be hard.

Synthetic arcs matrices have given the wrong answer twice here, once on the cell grid's density
and once on the temperature ladder, so anything measuring the arcs energy should start from these.

    python playground/capture_arcs.py <config.ini> <region> <n_blocks> <out.pkl>
"""
import os, sys, pickle, threading
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from gnome3d.mc import numba as mc_numba

CAP, LOCK, TARGET, OUT = [], threading.Lock(), int(sys.argv[3]), sys.argv[4]

def spy(pos, exp_dist_mat, step_size, settings, *a, **k):
    with LOCK:
        CAP.append((np.asarray(pos, np.float32).copy(),
                    np.asarray(exp_dist_mat, np.float64).copy(), float(step_size)))
        print(f"  captured block N={pos.shape[0]}  ({len(CAP)}/{TARGET})", flush=True)
        if len(CAP) >= TARGET:
            keep = sorted(CAP, key=lambda c: -c[0].shape[0])[:4]
            with open(OUT, "wb") as fh:
                pickle.dump(keep, fh)
            print(f"saved {len(keep)} blocks, N = {[c[0].shape[0] for c in keep]}", flush=True)
            sys.stdout.flush()
            os._exit(0)
    return np.float64(0.0)

mc_numba.mc_arcs_numba = spy
from gnome3d import simulate
simulate.run_region(sys.argv[1], sys.argv[2], 1)
