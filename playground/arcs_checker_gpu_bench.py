"""GPU speedup bench for the spatial-checkerboard arcs MC (f32, production dtype).

Quality is already validated (E_chk/E_seq=1.00 on real IBs, CPU). This measures the
WALL-CLOCK win on the GPU before we wire it into production. Run on the CUDA box:

    .venv/bin/python playground/arcs_checker_gpu_bench.py

For each large IB it runs, at a fixed bead-move budget: the FULL (N,N) kernel (baseline)
then the COLOR-GATHER (maxc,N) kernel at maxc=N//4 (safe) and N//5 (tight). Prints wall,
final energy (gather must equal full), speedup vs full, bead-moves/s, and max_cnt +
overflow guard (gather silently drops a color's anchors if max_cnt>maxc -> energy diverges).
Compare wall against your sequential GPU baseline (N=1555 single chain ~3900s at K=1).
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
    B = 80_000_000  # bead-moves (~near convergence); sweeps = B // N
    key = jax.random.PRNGKey(0)

    def timed(fn):
        """warm (compile) then time one call; returns (wall_s, *outputs_as_python)."""
        out = fn()
        jax.block_until_ready(out)
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        return time.perf_counter() - t, out

    print(f"\nbudget={B:,} bead-moves per IB.  full = run_checker (N,N) delta; "
          f"gather = run_checker_gather (maxc,N) delta.")
    print(f"{'N':>5} {'kernel':>8} {'maxc':>5} {'wall_s':>8} {'final_E':>11} "
          f"{'speedup':>8} {'Mmoves/s':>9} {'max_cnt':>8} {'overflow':>9}")
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
        posj, expj = jnp.asarray(pos), jnp.asarray(exp)
        nsw = max(1, B // N)
        moves = nsw * N

        # full (N,N) baseline
        w_full, (p, E) = timed(lambda: cj.run_checker(
            posj, expj, nsw, step, T0, dt, js, jc, sk, qk, *num, cell, key, 50))
        print(f"{N:>5} {'full':>8} {'-':>5} {w_full:>8.1f} {float(E):>11.1f} "
              f"{1.0:>7.1f}x {moves/max(w_full,1e-9)/1e6:>9.1f} {'-':>8} {'-':>9}", flush=True)

        # color-gather (maxc, N): N//4 is the safe margin, N//5 is tighter
        for maxc in (N // 4, N // 5):
            w, (p, E, mx) = timed(lambda mc=maxc: cj.run_checker_gather(
                posj, expj, nsw, step, T0, dt, js, jc, sk, qk, *num, cell, key, 50, mc))
            mx = int(mx)
            print(f"{N:>5} {'gather':>8} {maxc:>5} {w:>8.1f} {float(E):>11.1f} "
                  f"{w_full/max(w,1e-9):>7.1f}x {moves/max(w,1e-9)/1e6:>9.1f} {mx:>8} "
                  f"{('OVERFLOW' if mx > maxc else 'ok'):>9}", flush=True)


if __name__ == "__main__":
    main()
