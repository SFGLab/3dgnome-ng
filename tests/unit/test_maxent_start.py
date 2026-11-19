"""Unit checks for the maximum entropy gap start."""

from __future__ import annotations

import numpy as np

from gnome3d.pipeline.ib import maxent as me
from gnome3d.polymer import PolymerLaw
from gnome3d.settings import Settings

LAW = PolymerLaw(nu=0.34, s0_bp=1000)


def test_bridge() -> None:
    """The conditional law reproduces the targets."""
    mids = np.arange(0, 41) * 1000.0 * 2.0
    d = me.law_distances(mids, LAW)
    v = me._MAXWELL * d * d
    c = 0.5 * (v[1:, :1] + v[:1, 1:] - v[1:, 1:])
    lam = np.linalg.eigvalsh(c)
    assert float(lam.min()) > 0, f"the law's variogram is positive definite: min {lam.min():.3g}"
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
    assert float(rel.max()) < 0.03, f"mean pair distances match the targets: worst {rel.max():.3f}"
    assert coef[0] < coef[-1] and 0 < coef[0] < 1, "the bridge mean runs from one end to the other"


def test_windows() -> None:
    """Consecutive beads up to 10 kb."""
    st = np.arange(0, 35_000, 1000, dtype=np.int64)
    lab = me.windows(st, st + 1000)
    spans = [int((st[lab == q] + 1000).max() - st[lab == q].min()) for q in range(lab[-1] + 1)]
    assert max(spans) <= 10_000, f"every window within 10 kb: {spans}"
    assert bool(np.all(np.diff(lab) >= 0)) and lab[-1] == 3, "labels ascend by one"


def test_factor() -> None:
    """Window contacts scale the target as observed over expected."""
    mids = np.array([0.0, 20_000.0, 40_000.0, 60_000.0])
    lens = np.full(4, 10_000.0)
    fit = (np.log10(1e-6), 0.0)
    exp = 1e-6 * 1e8
    counts = np.full((4, 4), exp)
    counts[0, 2] = counts[2, 0] = 4 * exp
    counts[1, 3] = counts[3, 1] = 0.0
    f = me._gap_factor(counts, mids, lens, fit)
    assert abs(f[0, 1] - 1.0) < 1e-12, "a pair at its expectation keeps the law"
    assert f[0, 2] < 1.0, f"an enriched pair is pulled in: {f[0, 2]:.3f}"
    assert f[1, 3] > 1.0, f"a depleted pair is pushed out: {f[1, 3]:.3f}"


def test_start() -> None:
    """Anchors stay, the draw follows the seed."""
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
    assert np.array_equal(a[fixed], pos[fixed]), "anchors untouched"
    assert np.array_equal(a, b), "same seed same start"
    assert not np.array_equal(a, c), "another seed another start"
    assert float(np.abs(a[~fixed]).sum()) > 0, "subanchors moved off the anchors"


def test_settings() -> None:
    """The keys are read."""
    s = Settings.from_dict(
        {
            "data": {"contact_map": "x.mcool::/resolutions/1000"},
            "simulation_arcs_smooth": {"start": "maxent"},
        }
    )
    assert s.smooth_start == "maxent" and s.data_contact_map.endswith("1000"), "read"
    assert Settings().data_contact_map == "" and Settings().smooth_start == "line", "default off"


def test_anchor_draw() -> None:
    """A draw's mean pair distances are the targets."""
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
    assert bool(np.abs(np.median(r[free]) - 1.0) < 0.1), (
        f"arcless pairs at the law within ten percent: median {np.median(r[free]):.3f}"
    )
    assert bool(mean[5, 40] < 0.6 * law[5, 40]), (
        f"the loop pulled in toward its target: {mean[5, 40]:.2f} against law {law[5, 40]:.2f}"
    )
    one = me.anchor_draw(exp, mids, LAW, np.random.default_rng(1))
    d1 = np.linalg.norm(one[:, None] - one[None, :], axis=2)[iu] / law[iu]
    assert bool(np.quantile(d1, 0.01) < 0.3), (
        f"one draw spreads: 1 percent quantile of distance over law {np.quantile(d1, 0.01):.3f}"
    )


def test_conditioned() -> None:
    """Two fixed ends give the bridge, flanks pull the free beads."""
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
    assert bool(np.abs(mean - coef * 6.0).max() < 0.4), (
        f"mean matches the bridge: {np.abs(mean - coef * 6.0).max():.3f}"
    )
    spread = draws[:, 1:-1, 1].std(axis=0)
    expect = np.sqrt((root**2).sum(axis=1))
    assert bool(np.abs(spread / expect - 1).max() < 0.2), (
        f"spread matches the bridge: {np.abs(spread / expect - 1).max():.3f}"
    )
    fixed2 = fixed.copy()
    fixed2[:4] = True
    pos2 = pos.copy()
    pos2[1:4] = [[0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 0.0]]
    near = np.array(
        [me.conditioned_draw(pos2, fixed2, d, np.random.default_rng(k)) for k in range(200)]
    ).mean(axis=0)
    assert bool(near[4, 1] > 2.0), (
        f"a free bead beside fixed flank beads follows them: y {near[4, 1]:.2f}"
    )
