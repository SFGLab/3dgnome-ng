"""Unit checks for the maximum entropy gap start.

    python tests/unit/test_maxent_start.py

A bridge drawn with its far end from that end's own marginal is a draw from the whole chain, so
its mean pair distances must be the targets. The law's targets are a valid variogram, so the
conditional covariance needs no clipping. Window contacts move a pair's target the way the
heatmap rule does, and the start is a function of its seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.pipeline.ib import maxent as me  # noqa: E402
from gnome3d.polymer import PolymerLaw  # noqa: E402
from gnome3d.settings import Settings  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""))


LAW = PolymerLaw(nu=0.34, s0_bp=1000)


def test_bridge() -> None:
    print("\n[bridge] the conditional law reproduces the targets")
    mids = np.arange(0, 41) * 1000.0 * 2.0
    d = me.law_distances(mids, LAW)
    v = me._MAXWELL * d * d
    c = 0.5 * (v[1:, :1] + v[:1, 1:] - v[1:, 1:])
    lam = np.linalg.eigvalsh(c)
    check("the law's variogram is positive definite", float(lam.min()) > 0, f"min {lam.min():.3g}")
    coef, root = me.bridge(d)
    rng = np.random.default_rng(1)
    n = 20_000
    var_end = float(c[-1, -1])
    dist = np.zeros_like(d)
    for _ in range(n // 1000):
        pb = rng.normal(size=(1000, 3)) * np.sqrt(var_end)
        z = rng.normal(size=(1000, len(coef), 3))
        interior = coef[None, :, None] * pb[:, None, :] + np.einsum("ij,njc->nic", root, z)
        chain = np.concatenate([np.zeros((1000, 1, 3)), interior, pb[:, None, :]], axis=1)
        dist += np.linalg.norm(chain[:, :, None, :] - chain[:, None, :, :], axis=3).sum(0)
    dist /= n
    iu = np.triu_indices(len(d), 1)
    rel = np.abs(dist[iu] / d[iu] - 1.0)
    check(
        "mean pair distances match the targets", float(rel.max()) < 0.03, f"worst {rel.max():.3f}"
    )
    check("the bridge mean runs from one end to the other", coef[0] < coef[-1] and 0 < coef[0] < 1)


def test_windows() -> None:
    print("\n[windows] consecutive beads up to 10 kb")
    st = np.arange(0, 35_000, 1000, dtype=np.int64)
    lab = me.windows(st, st + 1000)
    spans = [int((st[lab == q] + 1000).max() - st[lab == q].min()) for q in range(lab[-1] + 1)]
    check("every window within 10 kb", max(spans) <= 10_000, f"{spans}")
    check("labels ascend by one", bool(np.all(np.diff(lab) >= 0)) and lab[-1] == 3)


def test_factor() -> None:
    print("\n[factor] window contacts scale the target as observed over expected")
    mids = np.array([0.0, 20_000.0, 40_000.0, 60_000.0])
    lens = np.full(4, 10_000.0)
    fit = (np.log10(1e-6), 0.0)
    exp = 1e-6 * 1e8
    counts = np.full((4, 4), exp)
    counts[0, 2] = counts[2, 0] = 4 * exp
    counts[1, 3] = counts[3, 1] = 0.0
    f = me._gap_factor(counts, mids, lens, fit)
    check("a pair at its expectation keeps the law", abs(f[0, 1] - 1.0) < 1e-12)
    check("an enriched pair is pulled in", f[0, 2] < 1.0, f"{f[0, 2]:.3f}")
    check("a depleted pair is pushed out", f[1, 3] > 1.0, f"{f[1, 3]:.3f}")


def test_start() -> None:
    print("\n[start] anchors stay, the draw follows the seed")
    n = 30
    starts = [i * 1000 for i in range(n)]
    ends = [s + 1000 for s in starts]
    fixed = np.zeros(n, dtype=bool)
    fixed[[0, 12, 29]] = True
    pos = np.zeros((n, 3), dtype=np.float32)
    pos[12] = [2.0, 0.0, 0.0]
    pos[29] = [2.0, 3.0, 0.0]
    a = me.maxent_start(pos, fixed, starts, ends, LAW, 2.0, np.random.default_rng(7))
    b = me.maxent_start(pos, fixed, starts, ends, LAW, 2.0, np.random.default_rng(7))
    c = me.maxent_start(pos, fixed, starts, ends, LAW, 2.0, np.random.default_rng(8))
    check("anchors untouched", np.array_equal(a[fixed], pos[fixed]))
    check("same seed same start", np.array_equal(a, b))
    check("another seed another start", not np.array_equal(a, c))
    check("subanchors moved off the anchors", float(np.abs(a[~fixed]).sum()) > 0)


def test_settings() -> None:
    print("\n[settings] the keys are read")
    s = Settings.from_dict(
        {
            "data": {"contact_map": "x.mcool::/resolutions/1000"},
            "simulation_arcs_smooth": {"start": "maxent"},
        }
    )
    check("read", s.smooth_start == "maxent" and s.data_contact_map.endswith("1000"))
    check("default off", Settings().data_contact_map == "" and Settings().smooth_start == "line")


def test_anchor_draw() -> None:
    print("\n[anchors] a draw's mean pair distances are the targets")
    rng = np.random.default_rng(0)
    mids = np.sort(rng.choice(np.arange(0, 2_000_000, 1000), 60, replace=False)).astype(np.float64)
    exp = -0.5 * np.ones((60, 60))
    exp[5, 40] = exp[40, 5] = 2.0
    law = me.law_distances(mids, LAW)
    acc = np.zeros((60, 60))
    n = 400
    for k in range(n):
        x = me.anchor_draw(exp, mids, LAW, np.random.default_rng(k)).astype(np.float64)
        acc += np.linalg.norm(x[:, None] - x[None, :], axis=2)
    mean = acc / n
    iu = np.triu_indices(60, 1)
    r = (mean / np.where(law > 0, law, 1.0))[iu]
    free = np.ones_like(r, dtype=bool)
    free[np.flatnonzero((iu[0] == 5) & (iu[1] == 40))] = False
    check(
        "arcless pairs at the law within ten percent",
        bool(np.abs(np.median(r[free]) - 1.0) < 0.1),
        f"median {np.median(r[free]):.3f}",
    )
    check(
        "the loop pulled in toward its target",
        bool(mean[5, 40] < 0.6 * law[5, 40]),
        f"{mean[5, 40]:.2f} against law {law[5, 40]:.2f}",
    )
    one = me.anchor_draw(exp, mids, LAW, np.random.default_rng(1))
    d1 = np.linalg.norm(one[:, None] - one[None, :], axis=2)[iu] / law[iu]
    check(
        "one draw spreads",
        bool(np.quantile(d1, 0.01) < 0.3),
        f"1 percent quantile of distance over law {np.quantile(d1, 0.01):.3f}",
    )


def test_conditioned() -> None:
    print("\n[conditioned] two fixed ends give the bridge, flanks pull the free beads")
    mids = np.arange(0, 40_000, 1000, dtype=np.float64)
    d = me.law_distances(mids, LAW)
    pos = np.zeros((40, 3))
    pos[-1] = [6.0, 0.0, 0.0]
    fixed = np.zeros(40, dtype=bool)
    fixed[[0, -1]] = True
    coef, root = me.bridge(d)
    draws = np.array(
        [me.conditioned_draw(pos, fixed, d, np.random.default_rng(k)) for k in range(400)]
    )
    mean = draws.mean(axis=0)[1:-1, 0]
    check(
        "mean matches the bridge",
        bool(np.abs(mean - coef * 6.0).max() < 0.4),
        f"{np.abs(mean - coef * 6.0).max():.3f}",
    )
    spread = draws[:, 1:-1, 1].std(axis=0)
    expect = np.sqrt((root**2).sum(axis=1))
    check(
        "spread matches the bridge",
        bool(np.abs(spread / expect - 1).max() < 0.2),
        f"{np.abs(spread / expect - 1).max():.3f}",
    )
    fixed2 = fixed.copy()
    fixed2[:4] = True
    pos2 = pos.copy()
    pos2[1:4] = [[0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0]]
    near = np.array(
        [me.conditioned_draw(pos2, fixed2, d, np.random.default_rng(k)) for k in range(200)]
    ).mean(axis=0)
    check(
        "a free bead beside fixed flank beads follows them",
        bool(near[4, 1] > 2.0),
        f"y {near[4, 1]:.2f}",
    )


def main() -> int:
    test_bridge()
    test_windows()
    test_factor()
    test_start()
    test_settings()
    test_anchor_draw()
    test_conditioned()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  failed: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
