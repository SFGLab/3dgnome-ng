"""Validate the BatchExecutor path against SerialExecutor.

With the SMOOTH batch runner registered (mc_smooth_jax_batch), BatchExecutor
runs smooth on JAX (restart fan-out + best-per-IB) while arcs/densify/heat fall
back to their numba serial runners (no batch runner yet) — a valid mixed run.

Checks vs the all-numba SerialExecutor:
  1. bead (start,end,kind) identical   — densify is RNG-free, layout must match.
  2. gyration in the same ballpark      — JAX smooth seeds differently, so the
                                          fold is statistically equivalent, not
                                          bit-identical (same bar as the existing
                                          region-batch validation).
  3. determinism                        — BatchExecutor rerun is identical
                                          (mc_smooth_jax_batch keys PRNGKey(0)
                                          at empty scope).

Runs on CPU-JAX locally (slow but valid) or GPU.
    python playground/refactor/validate_batch.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.pipeline.executor import BatchExecutor, SerialExecutor  # noqa: E402
from gnome3d.reconstruct import reconstruct  # noqa: E402


def _gyration(beads) -> float:
    xyz = np.array([(b.x, b.y, b.z) for b in beads], dtype=np.float64)
    return float(np.linalg.norm(xyz - xyz.mean(axis=0), axis=1).mean())


def _layout(beads):
    return sorted((b.start, b.end, b.kind) for b in beads)


def main() -> int:
    s, bed, data, _ = load_region()
    chrs = [bed.chr]
    ok = True

    serial = reconstruct(s, data, chrs, bed, executor=SerialExecutor())[bed.chr]
    batch = reconstruct(s, data, chrs, bed, executor=BatchExecutor())[bed.chr]
    rerun = reconstruct(s, data, chrs, bed, executor=BatchExecutor())[bed.chr]

    same_layout = len(serial) == len(batch) and _layout(serial) == _layout(batch)
    ok &= same_layout
    print(f"  {'ok ' if same_layout else 'FAIL'} layout identical (serial {len(serial)} vs batch {len(batch)})")

    g_s, g_b = _gyration(serial), _gyration(batch)
    rel = abs(g_s - g_b) / max(g_s, 1e-9)
    structural = rel < 0.5
    ok &= structural
    print(f"  {'ok ' if structural else 'FAIL'} gyration ballpark (serial {g_s:.1f} vs batch {g_b:.1f}, rel {rel:.0%})")

    deterministic = len(batch) == len(rerun) and all(a == b for a, b in zip(batch, rerun, strict=True))
    ok &= deterministic
    print(f"  {'ok ' if deterministic else 'FAIL'} BatchExecutor deterministic (rerun identical)")

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
