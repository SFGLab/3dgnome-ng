"""Unit checks for the contact term of the block layout."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d.mc import numba as mc_numba
from gnome3d.pipeline.coarse.build import block_heatmap_distances
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings


def base() -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=0.3, s0_bp=1000, q_half=1.0)
    s.use_excluded_volume = False
    s.use_confinement = False
    s.max_temp_ib = 1.0  # cold, a descent, see test_hot_layout_never_settles
    return s


def test_block_distances() -> None:
    s = base()
    blocks = [
        (0, 200_000, 100_000),
        (300_000, 500_000, 400_000),
        (600_000, 800_000, 700_000),
        (900_000, 1_100_000, 1_000_000),
    ]
    contacts = []
    for _ in range(10):
        contacts.append(("chr1", 100_000, "chr1", 400_000, 1.0))  # blocks 0 and 1, strong
    contacts.append(("chr1", 400_000, "chr1", 700_000, 1.0))  # blocks 1 and 2, weak
    contacts.append(("chr1", 700_000, "chr1", 1_000_000, 1.0))  # blocks 2 and 3, weak
    d = block_heatmap_distances(s, contacts, "chr1", blocks)
    assert d.shape == (4, 4), "one row and column per block"
    assert np.allclose(d, d.T), "symmetric"
    assert d[0, 2] <= 0.0 and d[0, 3] <= 0.0 and d[1, 3] <= 0.0, (
        "no target where there is no contact"
    )
    assert 0.0 < d[0, 1] < d[1, 2], (
        f"the strong pair sits closer than the weak pairs at the same separation: {d[0, 1]:.3f} < {d[1, 2]:.3f}"
    )
    bg = s.polymer_law().background(300_000)
    assert d[1, 2] > bg > d[0, 1], (
        f"weak pairs sit beyond the background, the strong one inside it: {d[1, 2]:.2f} > {bg:.2f} > {d[0, 1]:.2f}"
    )
    assert (np.diag(d) <= 0.0).all(), "the diagonal carries no target"


def run(s: Settings, heat: np.ndarray | None) -> np.ndarray:
    rng = np.random.default_rng(5)
    pos = rng.normal(0.0, 3.0, size=(6, 3)).astype(np.float32)
    dtn = np.full(5, 2.0, dtype=np.float32)
    mc_numba.seed_numba(11)
    mc_numba.mc_ib_numba(pos, dtn, 0.5, s, heat_dist=heat)
    return pos.astype(np.float64)


def test_weight_zero_is_byte_exact() -> None:
    s = base()
    s.heatmap_weight_ib = 0.0
    heat = np.zeros((6, 6))
    heat[0, 5] = heat[5, 0] = 1.0
    a = run(s, None)
    b = run(s, heat)
    assert np.array_equal(a, b), "a heat map at weight zero changes nothing"


def test_the_term_pulls() -> None:
    s = base()
    heat = np.zeros((6, 6))
    heat[0, 5] = heat[5, 0] = 1.0
    s.heatmap_weight_ib = 0.0
    off = run(s, heat)
    s.heatmap_weight_ib = 5.0
    on = run(s, heat)
    d_off = np.linalg.norm(off[0] - off[5])
    d_on = np.linalg.norm(on[0] - on[5])
    assert d_on < d_off, f"a contact target pulls its pair in: {d_on:.2f} < {d_off:.2f}"


def test_hot_layout_never_settles() -> None:
    """At `max_temp_ib` 20 with `jump_coef` 20 the rule accepts about a third of uphill moves
    and the chain does not settle. Cold, the same run settles.
    """
    s = base()
    s.heatmap_weight_ib = 0.0
    dtn = np.full(5, 2.0)
    s.max_temp_ib = 20.0
    hot = np.linalg.norm(np.diff(run(s, None), axis=0), axis=1).mean()
    s.max_temp_ib = 1.0
    cold = np.linalg.norm(np.diff(run(s, None), axis=0), axis=1).mean()
    assert hot > 3.0 * dtn.mean(), f"hot, the bonds end far from target: {hot:.1f} vs 2"
    assert abs(cold - 2.0) < 0.05, f"cold, the bonds end on target: {cold:.3f}"
