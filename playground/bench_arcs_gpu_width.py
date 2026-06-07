"""Profile the arcs region-batch MC on a CUDA GPU.

ONE script to answer "why is arcs slow on GPU and what's the lever".  It drives the
REAL `mc_arcs_jax_batch` path (with the instrumentation added to gnome3d/mc/jax/arcs.py)
on REAL GM12878 chr1 arc problems, and prints:

  (a) per-batch convergence SPREAD  -> is the batch wall set by ONE slow chain?
      (conv p50/p90/max, never-converged, "wasted" frozen-but-stepped chain-iters)
  (b) COMPILE vs EXECUTE split       -> how much wall is XLA compilation?
  (c) recompiles / distinct (K,B)    -> dump_arcs_profile() aggregate
  (d) WIDTH scaling (K-scan)         -> is it latency-bound with spare GPU width?
      (replicate the largest IB at K=1,2,4,...; per-outer-iter wall flat == spare width)
  (e) which IBs (by N) dominate      -> the inventory table + padding tax B^2/N^2

Run on the CUDA box:

    GNOME3D_ARCS_PROFILE=1 .venv/bin/python playground/bench_arcs_gpu_width.py

The env var MUST be set (this script sets it itself before importing, but exporting it
is harmless).  First run captures real chr1 arc problems via a fast dryrun and pickles
them to /tmp; subsequent runs load the pickle (delete it to recapture).

NOTE on fidelity: exp_dist matrices come from the heatmap->distance estimate stage
(computed BEFORE arcs MC), so capturing them with config_dryrun.ini (fewer MC steps)
yields the SAME matrices config.ini would use; only the MC step budget differs, and we
profile with the real config.ini budget (50000).  See the arcs-gpu instrumentation notes.
"""

from __future__ import annotations

import logging
import os
import pickle
import time

import numpy as np

import gnome3d.log as glog
import gnome3d.mc.jax.arcs as A
from gnome3d.mc.jax.util import jax_bucket_for, jax_device_budget_bytes
from gnome3d.settings import Settings

# Force the compile/run/cost profiling on (production reads GNOME3D_ARCS_PROFILE at
# import time; here we flip the module flag directly so import order doesn't matter).
os.environ.setdefault("GNOME3D_ARCS_PROFILE", "1")
A._ARCS_PROFILE = True

CACHE = "/tmp/arcs_chr1_ibs.pkl"
DATA_DIR = "data/GM12878"
DRYRUN_CFG = "data/GM12878/config_dryrun.ini"
REAL_CFG = "data/GM12878/config.ini"


# ----------------------------------------------------------------------------------
def device_report() -> bool:
    """Print the JAX device + memory budget.  Returns True if a GPU is present."""
    import jax

    backend = jax.default_backend()
    devs = jax.devices()
    print(f"[device] jax backend={backend!r} devices={devs}")
    budget = jax_device_budget_bytes()
    if budget:
        print(f"[device] arcs device budget = {budget / 1e9:.1f} GB")
    on_gpu = backend == "gpu" or any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devs)
    if not on_gpu:
        print("[device] !!! NOT on a GPU — numbers below are CPU and NOT representative !!!")
    return on_gpu


# ----------------------------------------------------------------------------------
def capture_chr1_ibs() -> list[tuple[np.ndarray, float]]:
    """Capture real chr1 arc problems (exp_dist, step) by monkeypatching the numba
    arcs kernel during a fast dryrun (smooth no-op'd).  Cached to a pickle."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            probs = pickle.load(f)
        print(f"[capture] loaded {len(probs)} cached IBs from {CACHE}")
        return probs

    print("[capture] no cache — running chr1 dryrun to capture arc problems...")
    import gnome3d.mc.jax as mc_jax
    import gnome3d.mc.numba as mc_numba
    import gnome3d.mc.numba.arcs as nbarcs
    import gnome3d.mc.numba.smooth as nbsmooth

    captured: list[tuple[np.ndarray, float]] = []
    _orig = nbarcs.mc_arcs_numba

    def warc(pos, exp, step, s):  # noqa: ANN001, ANN202
        captured.append((np.asarray(exp).copy(), float(step)))
        return _orig(pos, exp, step, s)

    nbarcs.mc_arcs_numba = warc
    mc_numba.mc_arcs_numba = warc
    # no-op smooth so the dryrun finishes fast (we only want arcs exp_dist)
    mc_jax.mc_smooth_jax_batch = lambda probs, s: [
        (0.0, np.asarray(p["pos"])) for p in probs
    ]

    def nosm(pos, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        return 0.0

    mc_numba.mc_smooth_numba = nosm
    nbsmooth.mc_smooth_numba = nosm

    from gnome3d.simulate import run_genome

    # The dryrun config sets output_level=2 and run_genome re-runs log.setup(2),
    # which would flood the console with [mc.numba] per-batch "step N score=..."
    # DEBUG lines (thousands per slow IB).  Pin the mc.numba child logger to WARNING
    # for the capture: child levels survive log.setup() (it only touches the gnome3d
    # root + handlers), so the per-step spam is dropped at the source either way.
    nb_log = logging.getLogger("gnome3d.mc.numba")
    prev_level = nb_log.level
    nb_log.setLevel(logging.WARNING)
    try:
        run_genome(DRYRUN_CFG, "chr1", 1, data_dir=DATA_DIR)
    finally:
        nb_log.setLevel(prev_level)
        nbarcs.mc_arcs_numba = _orig
        mc_numba.mc_arcs_numba = _orig

    # dedup by (N, rounded exp sum)
    seen: set[tuple[int, float]] = set()
    probs: list[tuple[np.ndarray, float]] = []
    for exp, step in captured:
        key = (exp.shape[0], round(float(exp.sum()), 3))
        if key in seen:
            continue
        seen.add(key)
        probs.append((exp.astype(np.float32), step))
    with open(CACHE, "wb") as f:
        pickle.dump(probs, f)
    print(f"[capture] captured {len(probs)} unique IBs -> {CACHE}")
    return probs


def seed_pos(exp: np.ndarray, step: float, seed: int) -> np.ndarray:
    """A small-noise seed like the pipeline's restart noise (arcs noises all anchors)."""
    n = exp.shape[0]
    return (np.random.default_rng(seed).standard_normal((n, 3)) * float(step) * 0.5).astype(np.float32)


# ----------------------------------------------------------------------------------
def inventory(probs: list[tuple[np.ndarray, float]], s: Settings) -> None:
    """Print the IB size inventory + bucket padding tax (e for question (e))."""
    print("\n=== (e) IB inventory (sorted by N) — padding tax = B^2/N^2 ===")
    print(f"{'N':>6} {'B':>6} {'pad B^2/N^2':>12}")
    rows = sorted(probs, key=lambda x: x[0].shape[0])
    for exp, _ in rows:
        n = exp.shape[0]
        b = jax_bucket_for(n) if s.mc_executor_jax_bucket_shapes else n
        print(f"{n:>6} {b:>6} {(b * b) / max(n * n, 1):>12.2f}")
    sizes = [e.shape[0] for e, _ in rows]
    print(f"[inventory] {len(sizes)} IBs  min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")


# ----------------------------------------------------------------------------------
def largest_batch(probs: list[tuple[np.ndarray, float]], s: Settings, n_big: int = 8) -> None:
    """Run the n_big largest IBs once (cold) so the per-batch log shows compile-vs-run
    + conv spread for the buckets that dominate arcs wall (questions a, b, c)."""
    print(f"\n=== (a,b,c) {n_big} largest IBs, COLD batch (real {s.mc_stop_steps}-step MC) ===")
    big = sorted(probs, key=lambda x: x[0].shape[0])[-n_big:]
    problems = [
        {"pos": seed_pos(exp, step, i), "exp_dist": exp, "step_size": step}
        for i, (exp, step) in enumerate(big)
    ]
    # cold: clear compiled kernels + seen-shape set so compile cost is measured
    A._kernel_cache.clear()
    A._arcs_seen_shapes.clear()
    t0 = time.perf_counter()
    res = A.mc_arcs_jax_batch(problems, s)
    print(f"[largest] {n_big} IBs total wall {time.perf_counter()-t0:.1f}s")
    print(f"{'N':>6} {'score':>12}")
    for (exp, _), (score, _pos) in zip(big, res, strict=True):
        print(f"{exp.shape[0]:>6} {score:>12.3f}")


# ----------------------------------------------------------------------------------
def width_scan(probs: list[tuple[np.ndarray, float]], s: Settings) -> None:
    """Replicate the SINGLE largest IB into K copies and sweep K to see whether the
    PER-ITER wall stays flat (spare GPU width) or grows (compute/mem bound). (d)

    FAST: forces the kernel to stop after ~1 outer iter (A._ARCS_FORCE_SCORE_EPS)
    instead of running each launch to full convergence (~thousands of iters for a big
    IB - that's what made this take ~days).  Per-iter compute is constant across
    iters, so one iter is a faithful per-iter-wall sample; we take the min of a few."""
    exp, step = max(probs, key=lambda x: x[0].shape[0])
    n = exp.shape[0]
    b = jax_bucket_for(n) if s.mc_executor_jax_bucket_shapes else n

    # isolate this section's aggregate from largest_batch's (dump_arcs_profile after)
    with A._arcs_profile_lock:
        A._arcs_profile.clear()

    # cap K to the device memory budget (auto basis), then scan powers of two.
    # 1-iter launches are cheap, so we can probe the full budget edge (max_k).
    s.mc_executor_jax_batch_width_arcs = "auto"
    max_k, basis = A._resolve_arcs_max_k(b, s)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= max(1, max_k)]
    if max_k > 1 and max_k not in ks:
        ks.append(max_k)
    n_steps = int(s.mc_stop_steps)
    print(f"\n=== (d) WIDTH K-scan on largest IB: N={n} B={b}  (max_k={max_k} {basis}) ===")
    print(f"    forced ~1 outer iter/launch; wall_ms = one batch of K chains x {n_steps} steps")
    print(f"{'K':>5} {'wall_ms':>9} {'us/step':>9} {'vs K=1':>8} {'Mstep/s':>9}")

    base = None
    warned = False
    A._ARCS_FORCE_SCORE_EPS = 1e30  # converge after 1 outer iter (per-iter-wall probe)
    try:
        for k in ks:
            s.mc_executor_jax_batch_width_arcs = str(k)  # force a single chunk of K
            problems = [
                {"pos": seed_pos(exp, step, j), "exp_dist": exp, "step_size": step}
                for j in range(k)
            ]
            A.mc_arcs_jax_batch(problems, s)  # warm (compile) this (K,B)
            samples: list[float] = []
            it = 0
            for _ in range(3):
                A.mc_arcs_jax_batch(problems, s)
                with A._arcs_profile_lock:
                    d = dict(A._last_batch_diag)
                samples.append(d["run_ms"] if d["run_ms"] > 0 else d["elapsed_s"] * 1e3)
                it = d["iter_f"]
            if it != 1 and not warned:
                print(f"    [warn] K={k} ran {it} iters (force-eps not applied); wall_ms is {it}x per-iter")
                warned = True
            wall_ms = min(samples) / max(it, 1)
            us_step = wall_ms * 1e3 / max(n_steps, 1)  # per-chain step latency (chains parallel)
            mstep_s = (k * n_steps) / max(wall_ms / 1e3, 1e-9) / 1e6  # total throughput
            if base is None:
                base = wall_ms
            print(f"{k:>5} {wall_ms:>9.1f} {us_step:>9.2f} {wall_ms / max(base, 1e-9):>7.2f}x {mstep_s:>9.1f}")
    finally:
        A._ARCS_FORCE_SCORE_EPS = None

    print(
        "[width] wall_ms ~flat across K  => latency-bound, spare GPU width "
        "(batch the large IBs + restarts wide for ~free)."
    )
    print(
        "[width] wall_ms grows ~linearly => compute/memory bound, width won't help "
        "(lever is fewer steps / intra-chain parallelism)."
    )


# ----------------------------------------------------------------------------------
def main() -> None:
    glog.setup(0)  # STATUS-level console: the arcs kernel diag lines are STATUS
    print("=" * 84)
    on_gpu = device_report()
    allow_cpu = os.environ.get("GNOME3D_ALLOW_CPU", "").strip().lower() in ("1", "true", "yes", "on")
    if not on_gpu and not allow_cpu:
        print("[main] not on a GPU — refusing to run the full real-budget MC on CPU "
              "(slow & not representative).")
        print("[main] set GNOME3D_ALLOW_CPU=1 to override.")
        return

    probs = capture_chr1_ibs()
    if not probs:
        print("[main] no IBs captured — aborting")
        return

    s = Settings()
    s.load_ini(REAL_CFG)  # real MC budget (stop_condition_steps -> mc_stop_steps=50000)
    s.mc_executor_jax_bucket_shapes = True
    s.mc_executor_jax_precompile_buckets = False  # precompile path is K=1 only; keep off
    print(f"[settings] mc_stop_steps={s.mc_stop_steps} steps_arcs={s.steps_arcs} "
          f"bucket_shapes={s.mc_executor_jax_bucket_shapes}")

    inventory(probs, s)
    # (a,b,c) runs the largest IBs to FULL convergence (~the slowest IB's wall, can be
    # ~1-2h) - skip it on re-runs when you only want the fast (d) width curve.
    if os.environ.get("GNOME3D_BENCH_SKIP_LARGEST", "").strip().lower() in ("1", "true", "yes", "on"):
        print("\n[main] skipping (a,b,c) largest-IB full-convergence batch "
              "(GNOME3D_BENCH_SKIP_LARGEST set)")
    else:
        largest_batch(probs, s)
        print("\n=== (c) per-(K,B) aggregate for the largest-IB batch ===")
        A.dump_arcs_profile()
    width_scan(probs, s)  # clears _arcs_profile at its start; fast (forced 1-iter)
    print("\n=== (c) per-(K,B) aggregate for the width scan ===")
    A.dump_arcs_profile()
    print("=" * 84)


if __name__ == "__main__":
    main()
