"""Unit checks for the arcs confinement radius derived from the law."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.pipeline.ib.arcs import _run, settings_for_block
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings


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
    assert settings_for_block(s, genomic) is s, "a positive factor leaves the settings alone"
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    s.use_confinement = False
    assert settings_for_block(s, genomic) is s, "confinement off leaves them alone"
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    s.confinement_radius_arcs = 4.0
    assert settings_for_block(s, genomic) is s, "an explicit radius wins"
    s = base()
    s.confinement_packing_factor_arcs = 0.0
    t = settings_for_block(s, genomic)
    want = s.polymer_law().confinement_radius(1_500_000)
    assert t is not s and abs(t.confinement_radius_arcs - want) < 1e-12, (
        f"factor zero derives the radius from the block span: {t.confinement_radius_arcs:.4f}"
    )
    assert s.confinement_radius_arcs == 0.0, "the original is untouched"
    assert t.polymer is s.polymer and t.steps_arcs == s.steps_arcs, "everything else is shared"
    u = settings_for_block(s, [int(g) for g in genomic])  # type: ignore[arg-type]
    assert abs(u.confinement_radius_arcs - want) < 1e-12, "the pipeline's plain list works too"
    triples = [(int(g) - 5_000, int(g) + 5_000, int(g)) for g in genomic]
    v = settings_for_block(s, triples)
    want3 = s.polymer_law().confinement_radius(1_510_000)
    assert abs(v.confinement_radius_arcs - want3) < 1e-12, (
        f"the state's (start, end, midpoint) triples span first start to last end: {v.confinement_radius_arcs:.4f} vs {want3:.4f}"
    )


def test_the_kernel_sees_it() -> None:
    """A block of repelling anchors ends at the wall. The derived run must equal a run given
    the same radius.
    """
    rng = np.random.default_rng(3)
    n = 16
    genomic = np.sort(rng.integers(0, 3_000_000, n)).astype(np.int64)
    exp = np.full((n, n), -0.5)
    np.fill_diagonal(exp, 0.0)
    pos0 = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)

    def run(s: Settings) -> np.ndarray:
        s.steps_arcs = 1
        s.arcs_solver = "lbfgs"
        s.arcs_repulsion_cutoff_factor = 50.0  # long range, the block reaches the wall
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
    assert np.array_equal(derived, explicit), (
        "derived and explicit at the same radius agree exactly"
    )
    radius = np.linalg.norm(derived - derived.mean(0), axis=1).mean()
    assert 0.7 * want < radius < 1.5 * want, (
        f"the block fills the law's sphere: {radius:.2f} vs {want:.2f}"
    )
    f = base()
    f.confinement_packing_factor_arcs = 1.5
    formula = run(f)
    assert not np.array_equal(derived, formula), "the old formula gives a different block"
