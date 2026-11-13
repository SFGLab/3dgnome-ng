"""Unit checks for the law spring of the smooth stage.

    python harness/test_smooth_law.py

`law_target` holds every bead pair closer than the range at the polymer law's distance and
leaves every other pair without a target. The stage feeds it to the subanchor heat term with
`smooth_law_weight` as the weight. On a long gap between two fixed anchors that start as a
compact coil, the spring holds the gap's interior pairs at the law.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.mc import numba as mc_numba  # noqa: E402
from gnome3d.pipeline.ib.smooth import _heat_and_settings, law_target  # noqa: E402
from gnome3d.pipeline.ib.start import coil_start  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []
NU = 0.34
S0 = 1000


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


def settings(**kw: object) -> Settings:
    s = Settings()
    s.polymer = PolymerLaw(nu=NU, s0_bp=S0)
    s.use_excluded_volume = True
    s.exclusion_apply_to_smooth = True
    s.exclusion_weight = 0.1
    s.exclusion_radius_smooth = 0.7
    s.use_confinement = False
    s.use_motif_orientation = False
    s.mc_smooth_chains = 1
    s.max_temp_smooth = 5.0
    s.dt_temp_smooth = 0.9999
    s.mc_stop_steps_smooth = 20000
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def gap_state(n: int, s: Settings, seed: int = 0) -> SimpleNamespace:
    """One gap of `n` beads at one bead per kilobase between two fixed anchors placed at the
    law's distance, the interior on a compact coil."""
    fixed = np.zeros(n, dtype=np.bool_)
    fixed[0] = fixed[-1] = True
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[-1] = [s.polymer.background((n - 1) * S0), 0.0, 0.0]
    for j in range(1, n - 1):
        pos[j] = pos[-1] * (j / (n - 1))
    dtn = np.ones(n - 1, dtype=np.float32)
    pos = coil_start(pos, fixed, dtn, 0.7, np.random.default_rng(seed))
    starts = [i * S0 for i in range(n)]
    return SimpleNamespace(
        settings=s,
        fixed=fixed,
        pos=pos,
        dtn=dtn,
        bead_starts=starts,
        bead_ends=starts,
        heat_dist=None,
    )


def test_target() -> None:
    s = settings(smooth_law_weight=0.0)
    st = gap_state(10, s)
    check("weight zero gives no target", law_target(st) is None)  # type: ignore[arg-type]
    heat, s2 = _heat_and_settings(st)  # type: ignore[arg-type]
    check("and leaves the heat target and the settings as they were", heat is None and s2 is s)

    s = settings(smooth_law_weight=0.5, smooth_law_range_bp=5000)
    st = gap_state(10, s)
    t = law_target(st)  # type: ignore[arg-type]
    assert t is not None
    law = s.polymer
    want = np.float32(law.background(3 * S0))
    check(
        "a pair inside the range holds the law's distance",
        abs(t[2, 5] - want) < 1e-5,
        f"{t[2, 5]} vs {want}",
    )
    check("a pair at or beyond the range has no target", t[2, 7] == 0.0 and t[1, 9] == 0.0)
    check("the diagonal has no target", not np.diag(t).any())
    check("two fixed beads have no target", t[0, 9] == 0.0)
    check("the target is symmetric", np.array_equal(t, t.T))
    check("no target is under one bead", t[t > 0].min() >= 1.0)
    heat, s2 = _heat_and_settings(st)  # type: ignore[arg-type]
    check(
        "the stage runs the heat term at the law weight on a copy",
        heat is not None
        and s2.subanchor_heatmap_dist_weight == 0.5
        and s.subanchor_heatmap_dist_weight == 1.0,
    )
    st.heat_dist = np.ones((10, 10), dtype=np.float32)
    try:
        _heat_and_settings(st)  # type: ignore[arg-type]
        refused = False
    except ValueError:
        refused = True
    check("with a subanchor heatmap already present it refuses", refused)


def interior_ratio(pos: np.ndarray, s: Settings) -> float:
    """Median realised distance over the law for interior pairs 30 to 90 beads apart."""
    n = len(pos)
    i, j = np.triu_indices(n, k=30)
    keep = (j - i < 90) & (i > 0) & (j < n - 1)
    i, j = i[keep], j[keep]
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    law = np.array([s.polymer.background(int(k) * S0) for k in j - i])
    return float(np.median(d / law))


def test_gap_follows_the_law() -> None:
    n = 151
    ratios = {}
    for w in (0.0, 0.01):
        s = settings(smooth_law_weight=w)
        st = gap_state(n, s, seed=7)
        heat, s2 = _heat_and_settings(st)  # type: ignore[arg-type]
        pos = st.pos.copy()
        mc_numba.seed_numba(3)
        mc_numba.mc_smooth_numba(pos, st.dtn, st.fixed, 0.5, s2, None, None, None, heat)
        ratios[w] = interior_ratio(pos, s)
    start = interior_ratio(gap_state(n, settings(), seed=7).pos, settings())
    # The toy gap reaches the law without the spring too, so the collapse seen on real runs,
    # 0.76 of the law at 100 to 200 kb, is measured on real regions and not here.
    check(
        "with the law spring a gap's interior sits within a tenth of the law",
        abs(np.log(ratios[0.01])) < np.log(1.1),
        f"start {start:.2f}, without {ratios[0.0]:.2f}, with {ratios[0.01]:.2f}",
    )


if __name__ == "__main__":
    test_target()
    test_gap_follows_the_law()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
