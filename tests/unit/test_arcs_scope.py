"""Unit checks for solving the anchors of a chromosome together."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from gnome3d import skeleton
from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg
from gnome3d.pipeline.coarse import build_state
from gnome3d.pipeline.ib.arcs import _run, settings_for_block
from gnome3d.settings import Settings


def test_arcs_confinement_weight() -> None:
    """The arcs stage's own confinement weight."""
    s = Settings()
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.confinement_weight = 0.1
    g = np.arange(10, dtype=np.int64) * 50_000
    assert settings_for_block(s, g).confinement_weight == 0.1, "zero means the shared weight"
    s.confinement_weight_arcs = 10.0
    assert settings_for_block(s, g).confinement_weight == 10.0, (
        "a positive value replaces it for the arcs stage"
    )
    assert s.confinement_weight == 0.1, "and leaves the shared weight alone"
    assert Settings().confinement_weight_arcs == 0.0, "off by default"


def test_pass_through() -> None:
    """The per block stage passes anchors through at chromosome scope."""
    s = Settings()
    s.arcs_scope = "chromosome"
    pos = np.random.default_rng(0).normal(0, 1, (12, 3)).astype(np.float32)
    exp = np.full((12, 12), -1.0)
    np.fill_diagonal(exp, 0.0)
    score, out = _run(
        {
            "anchor_pos": pos,
            "exp_dist": exp,
            "step_size": 0.01,
            "settings": s,
            "seed": 3,
            "anchor_genomic": np.arange(12, dtype=np.int64) * 50_000,
        }  # type: ignore[arg-type]
    )
    assert np.array_equal(out, pos), f"positions come back unchanged: score {score}"
    assert Settings().arcs_scope == "block", "the default scope is block"


def test_joint_solve_moves_every_block() -> None:
    """The joint solve places every block's anchors."""
    s = Settings()
    s.load_ini("data/GM12878/config.ini")
    s.data_dir = "data/GM12878"
    s.arcs_solver = "lbfgs"
    s.mc_executor_arcs = "threaded"
    s.arcs_scope = "chromosome"
    s.arcs_start = "walk"
    chrs, region = parse_chrs_arg("chr1:1-12000000")
    data = ContactData.from_files(s, chrs, region)
    state = build_state(s, data, chrs, region)
    seeds = skeleton.gather_all_ib_seeds(state, 0)
    multi = [sd for sd in seeds if sd.seed.anchor_seed_pos.shape[0] > 1]
    assert len(multi) >= 2, f"the region has several blocks: {len(multi)} blocks"
    spread = [float(np.std(sd.seed.anchor_seed_pos, axis=0).max()) for sd in multi]
    assert all(v > 0.05 for v in spread), (
        f"every block's anchors are spread out rather than at the centroid: min spread {min(spread):.3f}"
    )
    cens = np.array([sd.seed.anchor_seed_pos.mean(axis=0) for sd in multi])
    assert float(np.linalg.norm(cens[0] - cens[-1])) > 0.5, "blocks sit at distinct places"
    s.arcs_scope = "block"
    state2 = build_state(s, data, chrs, region)
    seeds2 = skeleton.gather_all_ib_seeds(state2, 0)
    at_centroid = [
        float(np.std(sd.seed.anchor_seed_pos, axis=0).max())
        for sd in seeds2
        if sd.seed.anchor_seed_pos.shape[0] > 1
    ]
    assert all(v == 0.0 for v in at_centroid), (
        "at block scope the anchors still seed at the centroid"
    )
