"""GPU compile-time probe for the arcs checker kernel.

Times ONLY the XLA compile (.lower().compile(), no MC run) of the production `kernel_chunk`
for the exact shape that hit the 25-min compile on chrX: K=1024 (the shrinking pads 719 -> the
next power of two), B=256, n_sweeps=195, maxc=28.

Use it to confirm whether the cell's `jnp.sort` was the slow-compile culprit, by running it in
both git states (old cell has the sort, the working copy has the subsampled-mean cell):

    python playground/arcs_checker_compile_probe.py          # working copy (mean cell)
    git stash && python playground/arcs_checker_compile_probe.py && git stash pop   # old (sort)

The two kernels lower to different HLO, so each compiles cold (the persistent cache won't
short-circuit the comparison).  Pass K / B / n_sweeps / maxc as argv to probe other shapes.
"""

import sys
import time

import jax
import numpy as np

import gnome3d.mc.jax.arcs_checker as ck

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
B = int(sys.argv[2]) if len(sys.argv) > 2 else 256
n_sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 195
maxc = int(sys.argv[4]) if len(sys.argv) > 4 else 28
excl_skip = 1

kernel_chunk, _init_energy = ck._build_checker_kernel(n_sweeps, excl_skip, maxc)

sds = jax.ShapeDtypeStruct
f = lambda *s: sds(s, np.float32)
i = lambda *s: sds(s, np.int32)

# carry = (pos, score, T, ms, conv, conv_iter)
carry = (f(K, B, 3), f(K), f(K), f(K), sds((K,), np.bool_), i(K))
# problem = (exp, step, r0, cx, cy, cz, R, n_active, succ, chain_id)
problem = (f(K, B, B), f(K), f(K), f(K), f(K), f(K), f(K), i(K), f(K), i(K))
# scalars = (dt, js, jc, stretch, squeeze, excl_w, conf_w, stop_improvement, score_eps, stop_ratio)
scalars = tuple(f() for _ in range(10))
base_key = jax.random.PRNGKey(0)

print(f"AOT-compiling arcs checker kernel_chunk: K={K} B={B} n_sweeps={n_sweeps} maxc={maxc}", flush=True)
t = time.perf_counter()
lowered = kernel_chunk.lower(carry, problem, scalars, base_key, np.int32(32), np.int32(0))
t_lower = time.perf_counter() - t
compiled = lowered.compile()  # the XLA compile (the part that took 25 min)
t_total = time.perf_counter() - t
print(f"  lower={t_lower:.1f}s  COMPILE(total)={t_total:.1f}s")
print(f"  (compiled OK: {compiled is not None})")
