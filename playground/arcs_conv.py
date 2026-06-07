"""Lever #1 investigation: arcs MC convergence / schedule.

Captures real chr1 large IBs (pos0 seed + exp_dist), runs the numba arcs MC to FULL
convergence (config.ini schedule) while recording the per-batch score curve, then
reports what fraction of the total energy improvement is reached at what fraction of
the steps.  Because the schedule is deterministic, an earlier stop just truncates the
SAME trajectory -> one full run gives the exact quality of any earlier cutoff.
"""

from __future__ import annotations

import logging
import os
import pickle
import time

import numpy as np

import gnome3d.mc.numba as mc_numba
import gnome3d.mc.numba.arcs as nbarcs
from gnome3d.settings import Settings
from gnome3d.util import seed_rng

CACHE = "/tmp/arcs_conv_ibs.pkl"  # (pos0, exp, step) per IB — distinct from the bench cache
DATA_DIR = "data/GM12878"
DRYRUN_CFG = "data/GM12878/config_dryrun.ini"
REAL_CFG = "data/GM12878/config.ini"


def capture() -> list[tuple[np.ndarray, np.ndarray, float]]:
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            out = pickle.load(f)
        print(f"[capture] loaded {len(out)} IBs from {CACHE}", flush=True)
        return out

    print("[capture] chr1 dryrun to capture (pos0, exp, step)...", flush=True)
    import gnome3d.mc.jax as mc_jax
    import gnome3d.mc.numba.smooth as nbsmooth

    grabbed: list[tuple[np.ndarray, np.ndarray, float]] = []
    _orig = nbarcs.mc_arcs_numba

    def warc(pos, exp, step, s):  # noqa: ANN001, ANN202
        grabbed.append((np.asarray(pos).copy(), np.asarray(exp).copy(), float(step)))  # pos0 BEFORE mutate
        return _orig(pos, exp, step, s)

    nbarcs.mc_arcs_numba = warc
    mc_numba.mc_arcs_numba = warc
    mc_jax.mc_smooth_jax_batch = lambda probs, s: [(0.0, np.asarray(p["pos"])) for p in probs]

    def nosm(pos, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        return 0.0

    mc_numba.mc_smooth_numba = nosm
    nbsmooth.mc_smooth_numba = nosm

    from gnome3d.simulate import run_genome

    nb_log = logging.getLogger("gnome3d.mc.numba")
    prev = nb_log.level
    nb_log.setLevel(logging.WARNING)  # silence per-step flood during capture
    try:
        run_genome(DRYRUN_CFG, "chr1", 1, data_dir=DATA_DIR)
    finally:
        nb_log.setLevel(prev)
        nbarcs.mc_arcs_numba = _orig
        mc_numba.mc_arcs_numba = _orig

    # dedup by (N, rounded exp sum); keep the LARGE ones
    seen: set[tuple[int, float]] = set()
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for pos, exp, step in grabbed:
        key = (exp.shape[0], round(float(exp.sum()), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append((pos.astype(np.float32), exp.astype(np.float32), float(step)))
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    print(f"[capture] {len(out)} unique IBs -> {CACHE}", flush=True)
    return out


class Curve(logging.Handler):
    """Capture the numba arcs per-batch (cumulative_step, score) from the DEBUG log."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[int, float, int]] = []

    def emit(self, rec: logging.LogRecord) -> None:
        a = rec.args
        if not (isinstance(a, tuple) and len(a) >= 4 and isinstance(a[0], str)):
            return
        s0 = a[0].replace(",", "").strip()
        if not s0.isdigit():
            return
        try:
            self.rows.append((int(s0), float(a[1]), int(a[3])))  # step, score, n_ok
        except (ValueError, TypeError):
            pass


def study(ibs: list[tuple[np.ndarray, np.ndarray, float]], s: Settings) -> None:
    lg = logging.getLogger("gnome3d.mc.numba")
    lg.setLevel(logging.DEBUG)
    cap = Curve()
    lg.addHandler(cap)

    print(f"\nschedule: max_temp={s.max_temp} dt_temp={s.dt_temp} "
          f"stop_improve={s.mc_stop_improvement} stop_succ={s.mc_stop_successes} "
          f"stop_steps={s.mc_stop_steps}", flush=True)
    print(f"\n{'N':>5} {'wall_s':>7} {'batches':>8} {'steps':>13} {'final_E':>10} "
          f"{'steps@90%':>12} {'@95%':>10} {'@99%':>10} {'@99.9%':>10}", flush=True)

    for pos0, exp, step in ibs:
        n = exp.shape[0]
        cap.rows = []
        pos = pos0.copy()
        seed_rng(0)
        mc_numba.seed_numba(0)
        t0 = time.perf_counter()
        final = mc_numba.mc_arcs_numba(pos, exp, float(step), s)
        wall = time.perf_counter() - t0
        rows = cap.rows
        if not rows:
            print(f"{n:>5}  (no curve captured)", flush=True)
            continue
        total_steps = rows[-1][0]
        s_start = rows[0][1]
        improvement = s_start - final
        # step at which X% of the total improvement is reached
        cuts = {}
        for frac in (0.90, 0.95, 0.99, 0.999):
            target = s_start - frac * improvement
            cuts[frac] = next((st for st, sc, _ in rows if sc <= target), total_steps)

        def pct(st: int) -> str:
            return f"{st:,}({100*st/total_steps:.0f}%)"

        print(f"{n:>5} {wall:>7.0f} {len(rows):>8} {total_steps:>13,} {final:>10.1f} "
              f"{pct(cuts[0.90]):>12} {pct(cuts[0.95]):>10} {pct(cuts[0.99]):>10} "
              f"{pct(cuts[0.999]):>10}", flush=True)
        # also dump a coarse curve (score at ~10 checkpoints) for shape
        idxs = [int(k * (len(rows) - 1) / 10) for k in range(11)]
        curve = "  ".join(f"{rows[i][0]//1000}k:{rows[i][1]:.0f}" for i in idxs)
        print(f"      curve: {curve}", flush=True)


def main() -> None:
    ibs = capture()
    ibs.sort(key=lambda x: x[1].shape[0])
    big = [t for t in ibs if t[1].shape[0] >= 400]
    print(f"[study] large IBs (N>=400): {[t[1].shape[0] for t in big]}", flush=True)
    s = Settings()
    s.load_ini(REAL_CFG)
    study(big, s)


if __name__ == "__main__":
    main()
