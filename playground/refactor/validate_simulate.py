"""Validate the `simulate` entry point routes through the task-DAG pipeline on
the numba backend and that ensembles vary.

  1. executor selection — numba -> SerialExecutor, jax -> None (legacy Solver).
  2. single structure  — simulate() produces the chr's beads, layout matching
                          the Solver path.
  3. ensemble varies   — n_structures=2 gives the SAME genomic layout but
                          DIFFERENT 3D coordinates (per-member seed offset), and
                          each member is itself reproducible.

    python playground/refactor/validate_simulate.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from _common import load_region  # noqa: E402

from gnome3d.reconstruct import pick_executor, reconstruct  # noqa: E402
from gnome3d.simulate import simulate  # noqa: E402


def layout(beads):
    return sorted((b.start, b.end, b.kind) for b in beads)


def main() -> int:
    s, bed, data, _ = load_region()
    chrs = [bed.chr]
    ok = True

    # 1) executor selection
    from gnome3d.pipeline.executor import BatchExecutor, SerialExecutor

    sel_numba = pick_executor(s)
    s_jax = type(s)()
    s_jax.load_ini("data/GM12878/config_dryrun.ini")
    s_jax.mc_backend = "jax"
    routing = isinstance(sel_numba, SerialExecutor) and isinstance(pick_executor(s_jax), BatchExecutor)
    ok &= routing
    print(f"  {'ok ' if routing else 'FAIL'} routing (numba->Serial, jax->Batch)")

    # 2) simulate(n=1) matches a direct reconstruct (entry point adds nothing but
    #    the per-structure loop + empty-chr filtering)
    new_beads = simulate(s, data, chrs, n_structures=1, region=bed)[0][bed.chr]
    direct = reconstruct(s, data, chrs, bed, executor=pick_executor(s))[bed.chr]
    same_layout = layout(new_beads) == layout(direct) and len(new_beads) == len(direct)
    ok &= same_layout
    print(f"  {'ok ' if same_layout else 'FAIL'} simulate == direct reconstruct ({len(new_beads)} beads)")

    # 3) ensemble: same layout, different coords, each reproducible
    ens = simulate(s, data, chrs, n_structures=2, region=bed)
    a, b = ens[0][bed.chr], ens[1][bed.chr]
    same_layout2 = layout(a) == layout(b)
    coords_a = np.array([(x.x, x.y, x.z) for x in a])
    coords_b = np.array([(x.x, x.y, x.z) for x in b])
    differ = not np.array_equal(coords_a, coords_b)
    # reproducible: rerun member 0 identical
    repro = all(
        p == q for p, q in zip(a, simulate(s, data, chrs, 1, bed)[0][bed.chr], strict=True)
    )
    ens_ok = same_layout2 and differ and repro
    ok &= ens_ok
    print(
        f"  {'ok ' if ens_ok else 'FAIL'} ensemble (same layout={same_layout2}, "
        f"coords differ={differ}, member reproducible={repro})"
    )

    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
