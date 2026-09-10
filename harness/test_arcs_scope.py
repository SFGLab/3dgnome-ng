"""Unit checks for solving a chromosome's anchors together.

    python harness/test_arcs_scope.py

With `[simulation_arcs] scope = chromosome` every anchor of a chromosome is solved as one
problem, from each block's placed centroid, so loops, the contact background and the
compartment term act across blocks. The per block arcs stage then passes its anchors through
unchanged and the chains, the stitch and the relaxation run as they do at block scope. The
arcs stage also gets its own confinement weight, since the sphere that holds a chromosome
sized solve is stronger than the one a block's chain wants.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d import skeleton  # noqa: E402
from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse import build_state  # noqa: E402
from gnome3d.pipeline.ib.arcs import _run, settings_for_block  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def test_arcs_confinement_weight() -> None:
    print("\n[weight] the arcs stage's own confinement weight")
    s = Settings()
    s.use_confinement = True
    s.confinement_apply_to_arcs = True
    s.confinement_weight = 0.1
    g = np.arange(10, dtype=np.int64) * 50_000
    check("zero means the shared weight", settings_for_block(s, g).confinement_weight == 0.1)
    s.confinement_weight_arcs = 10.0
    check(
        "a positive value replaces it for the arcs stage",
        settings_for_block(s, g).confinement_weight == 10.0,
    )
    check("and leaves the shared weight alone", s.confinement_weight == 0.1)
    check("off by default", Settings().confinement_weight_arcs == 0.0)


def test_pass_through() -> None:
    print("\n[scope] the per block stage passes anchors through at chromosome scope")
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
    check("positions come back unchanged", np.array_equal(out, pos), f"score {score}")
    check("the default scope is block", Settings().arcs_scope == "block")


def test_joint_solve_moves_every_block() -> None:
    print("\n[scope] the joint solve places every block's anchors")
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
    check("the region has several blocks", len(multi) >= 2, f"{len(multi)} blocks")
    spread = [float(np.std(sd.seed.anchor_seed_pos, axis=0).max()) for sd in multi]
    check(
        "every block's anchors are spread out rather than at the centroid",
        all(v > 0.05 for v in spread),
        f"min spread {min(spread):.3f}",
    )
    cens = np.array([sd.seed.anchor_seed_pos.mean(axis=0) for sd in multi])
    check("blocks sit at distinct places", float(np.linalg.norm(cens[0] - cens[-1])) > 0.5)
    s.arcs_scope = "block"
    state2 = build_state(s, data, chrs, region)
    seeds2 = skeleton.gather_all_ib_seeds(state2, 0)
    at_centroid = [
        float(np.std(sd.seed.anchor_seed_pos, axis=0).max())
        for sd in seeds2
        if sd.seed.anchor_seed_pos.shape[0] > 1
    ]
    check(
        "at block scope the anchors still seed at the centroid", all(v == 0.0 for v in at_centroid)
    )


def main() -> int:
    print("arcs scope checks")
    test_arcs_confinement_weight()
    test_pass_through()
    test_joint_solve_moves_every_block()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
