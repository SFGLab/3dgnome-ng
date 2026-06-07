"""GPU speedup bench for the spatial-checkerboard arcs MC (f32, production dtype).

Quality is already validated (E_chk/E_seq=1.00 on real IBs, CPU). This measures the
WALL-CLOCK win on the GPU before we wire it into production. Run on the CUDA box:

    GNOME3D_ARCS_PROFILE=1 .venv/bin/python /tmp/arcs_checker_gpu_bench.py

For each large IB it runs the checkerboard to a few budgets (warm then timed) and prints
wall + final energy + effective bead-moves/s. Compare wall against your sequential GPU
baseline (e.g. N=1555 single chain ~3900s at K=1) and energy against a production arcs run.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

# NOTE: do NOT enable x64 — we want f32 (the production/GPU dtype).
from gnome3d.settings import Settings

# import the validated f32-capable kernel (run_checker passes through pos.dtype). It lives
# next to this file; it enables x64 at import, but dtype follows input so f32 inputs => f32.
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cj", os.path.join(_HERE, "arcs_checker_jax.py"))
cj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cj)


def main() -> None:
    print(f"[device] backend={jax.default_backend()} devices={jax.devices()}")
    ibs = pickle.load(open("/tmp/arcs_conv_ibs.pkl", "rb"))
    by_n = {t[1].shape[0]: t for t in ibs}
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    sk, qk = float(s.spring_stretch_arcs), float(s.spring_squeeze_arcs)
    T0, dt, js, jc = float(s.max_temp), float(s.dt_temp), float(s.jump_scale), float(s.jump_coef)

    sizes = [n for n in (664, 1146, 1227, 1555) if n in by_n]
    budgets = (10_000_000, 40_000_000, 80_000_000)  # bead-moves (sweeps = budget // N)
    print(f"{'N':>5} {'budget':>12} {'sweeps':>8} {'wall_s':>8} {'final_E':>11} {'Mmoves/s':>9}")
    for N in sizes:
        pos0, exp, step = by_n[N]
        pos = np.asarray(pos0, np.float32)
        exp = np.asarray(exp, np.float32)
        step = float(step)
        prm = cj.cbnb.arc_params(s, pos.astype(np.float64), exp.astype(np.float64))
        num = cj._numeric_prm(prm)
        d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1e30)
        cell = float(4.0 * np.median(d.min(1)))
        posj = jnp.asarray(pos)
        expj = jnp.asarray(exp)
        for B in budgets:
            nsw = max(1, B // N)
            # warm (compile this shape), then timed
            p, _ = cj.run_checker(posj, expj, nsw, step, T0, dt, js, jc, sk, qk,
                                  *num, cell, jax.random.PRNGKey(0), 50)
            p.block_until_ready()
            t = time.perf_counter()
            p, E = cj.run_checker(posj, expj, nsw, step, T0, dt, js, jc, sk, qk,
                                  *num, cell, jax.random.PRNGKey(0), 50)
            p.block_until_ready()
            wall = time.perf_counter() - t
            moves = nsw * N  # one proposal per anchor per sweep
            print(f"{N:>5} {B:>12,} {nsw:>8} {wall:>8.1f} {float(E):>11.1f} {moves/max(wall,1e-9)/1e6:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
