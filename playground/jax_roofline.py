"""Is a JAX MC kernel memory bound, compute bound, or latency bound, at the real shapes.

Running a whole reconstruction to learn something about one kernel costs twenty minutes and
answers only what the wall time happens to expose. This compiles the kernel at a production
shape, asks XLA what it will read and how many operations it will do, runs it, and divides.

The verdict comes from where the achieved rates sit against the device's two ceilings. Near peak
bandwidth and far from peak arithmetic means memory bound, and the lever is reading less. Near
peak arithmetic means compute bound. Far from both means latency bound, and neither lever helps,
only more work in flight.

The shapes are the ones a real chr1 run launches, so the answer applies to production rather
than to a benchmark.

    python playground/jax_roofline.py [--rounds N]

Needs a GPU, so it runs on the workstation. `nsys profile -t cuda --stats=true` over the same
script gives the per kernel breakdown when this says something needs decomposing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.jax import smooth as jsmooth  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

# RTX 4060 Ti. fp32 fused multiply add peak and memory bandwidth off the spec sheet.
PEAK_TFLOPS = 22.06
PEAK_GBS = 288.0

# The launches a real chr1:1-60000000 run makes, from its own log.
SHAPES: list[tuple[str, int, int, bool, bool]] = [
    # name,                       K,   B,     heat,  orn
    ("smooth heat+orn (real)", 3, 16384, True, True),
    ("smooth heat+orn small", 2, 4096, True, True),
    ("smooth orn only", 6, 2048, False, True),
    ("estimate dry (real)", 80, 16384, False, False),
    ("estimate dry, K=32", 32, 16384, False, False),
]


def settings(steps: int = 50_000) -> Settings:
    s = Settings()
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 0.1
    s.exclusion_radius_smooth = 0.0
    s.use_confinement = True
    s.confinement_apply_to_smooth = True
    s.max_temp_smooth = 2.0
    s.mc_stop_steps_smooth = steps
    # Exactly one round of `steps`. The plateau test is `score > improvement * ms_score and
    # n_ok < successes`, so an improvement of 0 and an unreachable success count make it true
    # on the first round and the loop exits there. Setting a huge improvement instead makes it
    # never true and the loop runs to _MAX_ITERS, which is ten thousand rounds.
    s.mc_stop_improvement_smooth = 0.0
    s.mc_stop_successes_smooth = 10**9
    s.mc_executor_jax_bucket_shapes = False
    return s


def problems(k: int, b: int, heat: bool, orn: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    out: list[dict[str, Any]] = []
    for i in range(k):
        pos = np.cumsum(rng.normal(size=(b, 3)), axis=0).astype(np.float32)
        fixed = np.zeros(b, dtype=bool)
        # Roughly one anchor in four, matching the real 4096-of-16384 launches.
        fixed[:: max(b // max(b // 4, 1), 1)] = True
        fixed[0] = fixed[-1] = True
        p: dict[str, Any] = {
            "pos": pos,
            "dtn": np.linalg.norm(np.diff(pos, axis=0), axis=1).astype(np.float32),
            "fixed": fixed,
            "step_size": 0.5,
            "seed": i + 1,
        }
        if heat:
            p["heat_dist"] = np.zeros((b, b), dtype=np.float32)
        if orn:
            n_anchor = int(fixed.sum())
            p["char_orientations"] = np.zeros(n_anchor, dtype=np.int8)
            p["anchor_neighbors"] = {j: [max(0, j - 1)] for j in range(n_anchor)}
            p["anchor_neighbor_weights"] = {j: [1.0] for j in range(n_anchor)}
        out.append(p)
    return out


def capture(k: int, b: int, heat: bool, orn: bool, s: Settings) -> tuple[Any, tuple[Any, ...]]:
    """Run one launch, keeping the jitted kernel and the arguments it was handed."""
    seen: dict[str, Any] = {}
    real_build = jsmooth._build_smooth_kernel  # noqa: SLF001

    def spy_build(*a: Any, **kw: Any) -> Any:
        bundle = real_build(*a, **kw)
        kernel = bundle[8]

        def spy_kernel(*args: Any) -> Any:
            seen.setdefault("args", args)
            return kernel(*args)

        spy_kernel.lower = kernel.lower  # type: ignore[attr-defined]
        return (*bundle[:8], spy_kernel, *bundle[9:])

    jsmooth._build_smooth_kernel = spy_build  # noqa: SLF001
    try:
        jsmooth.mc_smooth_jax_batch(problems(k, b, heat, orn), s)
    finally:
        jsmooth._build_smooth_kernel = real_build  # noqa: SLF001
    return real_build, seen["args"]


def analytic() -> None:
    """The roofline from what the kernel provably touches, since XLA cannot cost the loops.

    Each MC step moves one bead and scans every other for the excluded volume, so a chain reads
    its whole position array once per step. That is the dominant traffic; the heat term adds one
    (B,) row and the chain term is O(1).
    """
    print(f"\n{'shape':>24s} {'K':>4s} {'B':>6s} {'us/step':>8s} {'MB/step':>8s} "
          f"{'GB/s':>8s} {'GFLOP/s':>9s} {'%dram':>7s} {'%flop':>7s}")
    # us/step measured from real production launches, see design/kernel-performance.md.
    for name, k, b, us in (
        ("smooth heat+orn (real)", 3, 16384, 39.02),
        ("smooth heat+orn small", 2, 4096, 23.89),
        ("smooth orn only", 6, 2048, 18.17),
        ("estimate dry (real)", 80, 16384, 23.86),
        ("estimate dry, K=32", 32, 16384, 14.24),
    ):
        mb = k * b * 3 * 4 / 1e6
        gbs = mb / 1e3 / (us / 1e6) / 1e3 * 1e3
        gflops = k * b * 10 / (us / 1e6) / 1e9
        print(f"{name:>24s} {k:>4d} {b:>6d} {us:>8.2f} {mb:>8.2f} {gbs:>8.0f} {gflops:>9.0f} "
              f"{gbs / PEAK_GBS * 100:>6.0f}% {gflops / (PEAK_TFLOPS * 1000) * 100:>6.1f}%")
    print("  A rate above 100% of DRAM means the working set is being served from L2 (32 MB on")
    print("  this card), so DRAM bandwidth is not the constraint. Far below both ceilings means")
    print("  the limit is the dependent chain of sequential steps, not bandwidth or arithmetic.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()
    s = settings(args.steps)

    print(f"peak {PEAK_TFLOPS:.1f} TFLOP/s fp32, {PEAK_GBS:.0f} GB/s")
    print(f"one round of {args.steps:,} steps per launch\n")
    hdr = (
        f"{'shape':>24s} {'K':>4s} {'B':>6s} {'ms':>8s} {'GFLOP/s':>9s} {'%peak':>6s} "
        f"{'GB/s':>8s} {'%peak':>6s} {'intensity':>10s}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, k, b, heat, orn in SHAPES:
        try:
            kernel_build, kargs = capture(k, b, heat, orn, s)
            bundle = kernel_build(
                int(s.mc_stop_steps_smooth), int(s.exclusion_skip_neighbors), heat, orn, 1, False
            )
            compiled = bundle[8].lower(*kargs).compile()
            cost = compiled.cost_analysis()
            if isinstance(cost, list):
                cost = cost[0]
            flops = float(cost.get("flops", 0.0))
            byts = float(cost.get("bytes accessed", 0.0))

            compiled(*kargs)[0].block_until_ready()
            t = time.perf_counter()
            compiled(*kargs)[0].block_until_ready()
            wall = time.perf_counter() - t

            gflops = flops / wall / 1e9
            gbs = byts / wall / 1e9
            pf, pb = gflops / (PEAK_TFLOPS * 10), gbs / PEAK_GBS * 100
            intensity = flops / byts if byts else 0.0
            verdict = (
                "MEMORY bound"
                if pb > 60
                else "COMPUTE bound"
                if pf > 60
                else f"LATENCY bound ({max(pf, pb):.0f}% of the nearer ceiling)"
            )
            print(
                f"{name:>24s} {k:>4d} {b:>6d} {wall * 1e3:>8.1f} {gflops:>9.1f} {pf:>5.1f}% "
                f"{gbs:>8.1f} {pb:>5.1f}% {intensity:>10.2f}  {verdict}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - one shape failing must not stop the sweep
            print(f"{name:>24s} {k:>4d} {b:>6d}  FAILED: {type(e).__name__}: {e}", flush=True)

    analytic()
    print("\nXLA's cost analysis does NOT see inside the loops: the MC steps run in a")
    print("while_loop over a fori_loop, and static costing does not unroll either, so the flops")
    print("and bytes it reports are a small fraction of the real work and the percentages above")
    print("are lower bounds, not the roofline. Checked against a hand count: one step on 3 chains")
    print("of 16,384 beads is about 0.5 MFLOP, so 20,000 steps is ~9.8 GFLOP, and cost analysis")
    print("reported 0.077. Use the analytic column below instead.")


if __name__ == "__main__":
    main()
