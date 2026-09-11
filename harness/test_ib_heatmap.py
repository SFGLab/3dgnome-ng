"""Unit checks for the block layout's contact term.

    python harness/test_ib_heatmap.py

Blocks within a segment were laid out by chain bonds, excluded volume and a sphere, with no
contact data at all, and their arrangement carried no Hi-C. The term bins the run's contacts by
block, converts them to distances with the law as the segment level does, and the block kernel
scores them as its heat term. Weight zero is byte exact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.pipeline.coarse.build import block_heatmap_distances  # noqa: E402
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
    s.use_excluded_volume = False
    s.use_confinement = False
    s.max_temp_ib = 1.0  # cold, a descent; see test_hot_layout_never_settles
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
    check("one row and column per block", d.shape == (4, 4))
    check("symmetric", np.allclose(d, d.T))
    check(
        "no target where there is no contact", d[0, 2] <= 0.0 and d[0, 3] <= 0.0 and d[1, 3] <= 0.0
    )
    check(
        "the strong pair sits closer than the weak pairs at the same separation",
        0.0 < d[0, 1] < d[1, 2],
        f"{d[0, 1]:.3f} < {d[1, 2]:.3f}",
    )
    bg = s.polymer_law().background(300_000)
    check(
        "weak pairs sit beyond the background, the strong one inside it",
        d[1, 2] > bg > d[0, 1],
        f"{d[1, 2]:.2f} > {bg:.2f} > {d[0, 1]:.2f}",
    )
    check("the diagonal carries no target", (np.diag(d) <= 0.0).all())


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
    check("a heat map at weight zero changes nothing", np.array_equal(a, b))


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
    check("a contact target pulls its pair in", d_on < d_off, f"{d_on:.2f} < {d_off:.2f}")


def test_hot_layout_never_settles() -> None:
    """At the schedule production inherited, max_temp_ib 20 against jump_coef 20, the rule accepts
    about a third of uphill moves whatever their size and the chain random walks hot until the
    stop fires. Cold, the same run is a descent and settles."""
    s = base()
    s.heatmap_weight_ib = 0.0
    dtn = np.full(5, 2.0)
    s.max_temp_ib = 20.0
    hot = np.linalg.norm(np.diff(run(s, None), axis=0), axis=1).mean()
    s.max_temp_ib = 1.0
    cold = np.linalg.norm(np.diff(run(s, None), axis=0), axis=1).mean()
    check("hot, the bonds end far from target", hot > 3.0 * dtn.mean(), f"{hot:.1f} vs 2")
    check("cold, the bonds end on target", abs(cold - 2.0) < 0.05, f"{cold:.3f}")


def main() -> int:
    print("block layout contact term checks")
    test_hot_layout_never_settles()
    test_block_distances()
    test_weight_zero_is_byte_exact()
    test_the_term_pulls()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
