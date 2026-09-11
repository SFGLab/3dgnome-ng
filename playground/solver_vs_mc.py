"""Compare solving the arcs stage against annealing it, on the production energy.

Three questions in one pass, on real captured blocks, with the settings loaded from a real
config so the energy is the one production minimises rather than a reconstruction of it. An
earlier version of this comparison used a hand built settings object which left the excluded
volume off, and every production config has it on for arcs, so those numbers were internally
consistent and measured the wrong energy.

Speed and energy: does the solver reach the annealer's minimum, and in how much less time.
Spread: does an ensemble survive, measured as the mean relative difference between the distance
matrices of every pair of structures, which needs no superposition.
Geometry: are the structures the same kind of structure, measured as where arc linked anchors
end up relative to the distance their arc asked for.

    python playground/solver_vs_mc.py <config.ini> <arcs_real.pkl> [n_starts]
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.mc.numba import seed_numba  # noqa: E402
from gnome3d.mc.numba.arcs import mc_arcs_numba  # noqa: E402
from gnome3d.mc.numba.arcs_solver import solve_arcs  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

ITERS = (200, 2000)


def rg(p: np.ndarray) -> float:
    return float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))


def spread(structs: list[np.ndarray]) -> float:
    def dm(p: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        return d[np.triu_indices(p.shape[0], 1)]

    ds = [dm(p) for p in structs]
    out = [
        float(np.abs(ds[i] - ds[j]).mean() / max(ds[i].mean(), 1e-9))
        for i in range(len(ds))
        for j in range(i + 1, len(ds))
    ]
    return float(np.mean(out)) if out else 0.0


def geometry(structs: list[np.ndarray], exp: np.ndarray, cutoff: float) -> tuple[float, ...]:
    ii, jj = np.where(np.triu(exp > 1e-6, 1))
    tgt = exp[ii, jj]
    r, ins, nn = [], [], []
    for p in structs:
        r.append(np.linalg.norm(p[ii] - p[jj], axis=1) / tgt)
        full = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(full, np.inf)
        ins.append((full < cutoff).mean())
        nn.append(float(np.min(full, axis=1).mean()))
    a = np.concatenate(r)
    return (
        float(np.median(a)),
        float(np.percentile(a, 10)),
        float(np.percentile(a, 90)),
        float(np.mean(ins)),
        float(np.mean(nn)),
    )


def main() -> None:
    s = Settings()
    if not s.load_ini(sys.argv[1]):
        sys.exit(f"could not read {sys.argv[1]}")
    s.arcs_solver = "mc"  # the arm is chosen explicitly below, not by the config
    print(
        f"settings from {Path(sys.argv[1]).name}: excluded volume on arcs "
        f"{bool(s.use_excluded_volume) and bool(s.exclusion_apply_to_arcs)}, "
        f"confinement {bool(s.use_confinement) and bool(s.confinement_apply_to_arcs)}, "
        f"repulsion cutoff factor {s.arcs_repulsion_cutoff_factor}"
    )

    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    rng = np.random.default_rng(0)
    for pos0, exp, step in pickle.load(open(sys.argv[2], "rb"))[:2]:
        n = pos0.shape[0]
        mask = exp > 1e-6
        cutoff = float(s.arcs_repulsion_cutoff_factor) * float(exp[mask].mean())
        starts = [
            np.ascontiguousarray(pos0 + rng.normal(0.0, step, size=pos0.shape).astype(np.float32))
            for _ in range(n_starts)
        ]
        print(f"\nREAL block N={n}, {int(mask.sum() // 2)} arc pairs, {n_starts} starts")
        print(
            f"  {'arm':>14s} {'sec':>8s} {'energy':>11s} {'vs MC':>7s} {'Rg':>7s} "
            f"{'spread':>8s} {'d/tgt':>7s} {'p10':>6s} {'p90':>6s} {'in cut':>7s} {'near':>6s}"
        )
        base_e = base_w = None
        for label, fn in [("MC", None)] + [(f"L-BFGS {i}", i) for i in ITERS]:
            E, S = [], []
            t = time.perf_counter()
            for k, st in enumerate(starts):
                p = st.copy()
                if fn is None:
                    seed_numba(100 + k)
                    np.random.seed(100 + k)
                    E.append(mc_arcs_numba(p, exp, step, s))
                    S.append(p.astype(np.float64))
                else:
                    e, q = solve_arcs(p, exp, s, iters=fn)
                    E.append(e)
                    S.append(q.astype(np.float64))
            w = time.perf_counter() - t
            e = float(np.mean(E))
            if base_e is None:
                base_e, base_w = e, w
            med, p10, p90, ins, nn = geometry(S, exp, cutoff)
            print(
                f"  {label:>14s} {w:>7.1f}s {e:>11,.1f} {e / base_e:>6.3f}x "
                f"{np.mean([rg(p) for p in S]):>7.3f} {spread(S):>7.2%} "
                f"{med:>7.2f} {p10:>6.2f} {p90:>6.2f} {ins:>6.1%} {nn:>6.3f}"
                + (f"   {base_w / w:.1f}x" if fn is not None else ""),
                flush=True,
            )


if __name__ == "__main__":
    main()
