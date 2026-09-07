"""Unit checks for the arcs confinement radius derived from the law.

    python harness/test_confinement_radius.py

With `confinement_packing_factor_arcs` at zero the arcs stage gives each block the sphere the
law says a chain of its span fills, and hands it to the kernels as an explicit radius so all of
them see one value. With the factor positive nothing changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.pipeline.ib.arcs import _run, settings_for_block  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def base() -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.confinement_radius_arcs = 0.0
    s.arcs_repulsion_cutoff_factor = 3.0
    return s


def test_settings_for_block() -> None:
    genomic = np.array([1_000_000, 1_400_000, 2_100_000, 2_500_000], dtype=np.int64)
    s = base()
    s.confinement_packing_factor_arcs = 1.5
    check("a positive factor leaves the settings alone", settings_for_block(s, genomic) is s)
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    s.use_confinement = False
    check("confinement off leaves them alone", settings_for_block(s, genomic) is s)
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    s.confinement_radius_arcs = 4.0
    check("an explicit radius wins", settings_for_block(s, genomic) is s)
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    t = settings_for_block(s, genomic)
    want = s.polymer_law().confinement_radius(1_500_000)
    check(
        "factor zero derives the radius from the block span",
        t is not s and abs(t.confinement_radius_arcs - want) < 1e-12,
        f"{t.confinement_radius_arcs:.4f}",
    )
    check("the original is untouched", s.confinement_radius_arcs == 0.0)
    check("everything else is shared", t.polymer is s.polymer and t.steps_arcs == s.steps_arcs)
    u = settings_for_block(s, [int(g) for g in genomic])  # type: ignore[arg-type]
    check("the pipeline's plain list works too", abs(u.confinement_radius_arcs - want) < 1e-12)
    triples = [(int(g) - 5_000, int(g) + 5_000, int(g)) for g in genomic]
    v = settings_for_block(s, triples)
    want3 = s.polymer_law().confinement_radius(1_510_000)
    check(
        "the state's (start, end, midpoint) triples span first start to last end",
        abs(v.confinement_radius_arcs - want3) < 1e-12,
        f"{v.confinement_radius_arcs:.4f} vs {want3:.4f}",
    )


def test_the_kernel_sees_it() -> None:
    """A block of repelling anchors ends at the wall, so where it ends says which radius the
    kernel was given. The derived run must match a run handed the same radius explicitly."""
    rng = np.random.default_rng(3)
    n = 16
    genomic = np.sort(rng.integers(0, 3_000_000, n)).astype(np.int64)
    exp = np.full((n, n), -0.5)
    np.fill_diagonal(exp, 0.0)
    pos0 = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)

    def run(s: Settings) -> np.ndarray:
        s.steps_arcs = 1
        s.arcs_solver = "lbfgs"
        s.arcs_repulsion_cutoff_factor = 50.0  # long range, so the block reaches the wall
        _score, pos = _run(
            {
                "anchor_pos": pos0,
                "exp_dist": exp,
                "step_size": 0.05,
                "settings": s,
                "seed": 7,
                "anchor_genomic": genomic,
            }
        )
        return np.asarray(pos, dtype=np.float64)

    s = base()
    s.confinement_packing_factor_arcs = 0.0
    derived = run(s)
    want = s.polymer_law().confinement_radius(int(genomic[-1] - genomic[0]))
    e = base()
    e.confinement_radius_arcs = want
    explicit = run(e)
    check(
        "derived and explicit at the same radius agree exactly", np.array_equal(derived, explicit)
    )
    radius = np.linalg.norm(derived - derived.mean(0), axis=1).mean()
    check(
        "the block fills the law's sphere",
        0.7 * want < radius < 1.5 * want,
        f"{radius:.2f} vs {want:.2f}",
    )
    f = base()
    f.confinement_packing_factor_arcs = 1.5
    formula = run(f)
    check("the old formula gives a different block", not np.array_equal(derived, formula))


def main() -> int:
    print("arcs confinement radius checks")
    test_settings_for_block()
    test_the_kernel_sees_it()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
