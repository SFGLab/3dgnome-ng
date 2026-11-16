"""Maximum entropy start for a block's gaps.

Under `start = maxent` each gap's subanchors are drawn from the Gaussian chain whose mean pair
distances are the targets, conditioned on the two anchors the arcs stage placed. A pair's target
is the law's background. With `[data] contact_map` it is scaled by the count between the windows
of about 10 kb the two beads are in, against the count a power law in separation expects. The
power law is fitted over the chromosome on the map's 10 kb resolution, so a stretch holding fewer
contacts than the chromosome at large reads as depleted. A map with no 10 kb resolution falls
back to a fit on the call's own windows, which reads every stretch against itself, and says so.
See `design/gap-fill.md`.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from gnome3d import log
from gnome3d.anchor_contacts import overlap_weights
from gnome3d.util import blas_threads

if TYPE_CHECKING:
    from gnome3d.polymer import PolymerLaw
    from gnome3d.types import BoolArray, F32Array, F64Array, I64Array

LOG = log.get("maxent")

_WINDOW_BP = 10_000
# The count under which a pair's ratio is shrunk toward its expectation, the count at which a
# pair with no contact sits three Poisson deviations under it.
_PSEUDO = 9.0
# A Gaussian vector's mean squared length over its squared mean length is 3 pi / 8, so a pair
# whose mean distance is d has a variance per coordinate of pi d^2 / 8.
_MAXWELL = np.pi / 8.0

# The chromosome expectation is read from 20 windows of 5 Mb spread along the chromosome at the
# map's 10 kb resolution, enough for separations to 5 Mb without reading the whole band.
_EXP_RES = 10_000
_EXP_WINDOWS = 20
_EXP_SPAN = 5_000_000

_COUNTS: dict[tuple[str, str, int, int], tuple[F64Array, F64Array]] = {}
_EXPECTED: dict[tuple[str, str], tuple[float, float] | None] = {}
_ANNOUNCED: set[str] = set()
_COOLERS: dict[str, Any] = {}
_LOCK = threading.Lock()


def law_distances(mids: F64Array, law: PolymerLaw) -> F64Array:
    """The law's background for every bead pair, zero on the diagonal."""
    sep = np.maximum(np.abs(mids[:, None] - mids[None, :]), 1.0)
    d = np.maximum(1.0, (sep / max(int(law.s0_bp), 1)) ** float(law.nu))
    np.fill_diagonal(d, 0.0)
    return d


_FIT_ITERS = 500
_FIT_RANK = 256
_FIT: dict[tuple[int, bytes], F64Array] = {}


def _targets(exp_dist: F64Array, mids: F64Array, law: PolymerLaw) -> F64Array:
    """Every anchor pair's target mean distance. A loop's target, the background where it holds
    a pair, the law's distance elsewhere."""
    d = law_distances(mids, law)
    d = np.where(exp_dist > 1e-6, exp_dist, np.where(exp_dist <= -1.0, -exp_dist, d))
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


def _fit_root(target: F64Array, loop: BoolArray) -> F64Array:
    """A root `y` of the covariance per coordinate whose pair variances best meet `target`.

    A pair's variance per coordinate under covariance `y y^T` is the squared distance between
    its two rows of `y`, so the fit is a stress minimisation on those rows, of the squared
    relative misfit to `pi / 8` times the squared target, the loops and every other pair
    weighted equally in total. A set of targets holding loops is not the distance matrix of any
    Gaussian, so the fit is the nearest one rather than an exact one. It starts from the root of
    the double centred squared targets with its negative modes dropped, at up to 256 columns.
    """
    from scipy.optimize import minimize

    n = target.shape[0]
    vt = _MAXWELL * target * target
    off = ~np.eye(n, dtype=bool)
    nl = max(int(loop.sum()), 1)
    w = np.where(loop, (off.sum() - nl) / nl, 1.0)
    w[~off] = 0.0
    w = w / np.where(off, vt * vt, 1.0)
    w /= float(w.sum())
    row = vt.mean(axis=1)
    g = -0.5 * (vt - row[:, None] - row[None, :] + float(row.mean()))
    lam, u = np.linalg.eigh(g)
    m = min(n, _FIT_RANK)
    y0 = u[:, -m:] * np.sqrt(np.clip(lam[-m:], 0.0, None))

    def fun(flat: F64Array) -> tuple[float, F64Array]:
        y = flat.reshape(n, m)
        sq = (y * y).sum(axis=1)
        v = sq[:, None] + sq[None, :] - 2.0 * (y @ y.T)
        r = w * (v - vt)
        # d/dy_i of sum_jk r_jk (v_jk - vt_jk) is 4 sum_j r_ij (y_i - y_j) over both orders.
        grad = 4.0 * (r.sum(axis=1)[:, None] * y - r @ y)
        return float((r * (v - vt)).sum()), grad.reshape(-1)

    with blas_threads():
        res = minimize(
            fun, y0.reshape(-1), jac=True, method="L-BFGS-B", options={"maxiter": _FIT_ITERS}
        )
    y = res.x.reshape(n, m)
    sq = (y * y).sum(axis=1)
    v = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (y @ y.T), 0.0)
    ratio = np.sqrt(v[off] / vt[off])
    lr = np.abs(np.log(np.maximum(ratio, 1e-12)))
    lo = loop[off]
    log.status(
        LOG,
        "maxent anchors: fitted in %d iterations, realised over target median %.3f on loops and %.3f elsewhere, absolute log deviation %.3f and %.3f",
        int(res.nit),
        float(np.median(ratio[lo])) if lo.any() else float("nan"),
        float(np.median(ratio[~lo])),
        float(np.median(lr[lo])) if lo.any() else float("nan"),
        float(np.median(lr[~lo])),
    )
    return y


def anchor_draw(
    exp_dist: F64Array, mids: F64Array, law: PolymerLaw, rng: np.random.Generator
) -> F32Array:
    """One draw of a chromosome's anchors from the maximum entropy Gaussian for the arcs
    targets, centred on the origin.

    Every pair has a target mean distance, a loop's target, the background where it holds the
    pair, the law's distance elsewhere, and the Gaussian is the one whose mean pair distances
    meet them as nearly as a Gaussian can. The fit depends on the targets alone and is kept for the run's
    later conformations. The draw keeps the fluctuations a minimised structure has none of.

    Parameters
    ----------
    exp_dist
        The arcs target matrix. Positive is a loop's target, at most minus one a background
        distance, anything else an arcless pair.
    mids
        The anchors' genomic midpoints.
    law
        The run's polymer law.
    rng
        The draw's generator.
    """
    n = len(mids)
    key = (n, exp_dist.tobytes() + mids.tobytes())
    root = _FIT.get(key)
    if root is None:
        log.status(
            LOG,
            "maxent anchors: %d anchors, %d loops",
            n,
            int(np.triu(exp_dist > 1e-6, 1).sum()),
        )
        loop = exp_dist > 1e-6
        root = _fit_root(_targets(exp_dist, mids, law), loop | loop.T)
        _FIT[key] = root
    x = root @ rng.standard_normal((root.shape[1], 3))
    x -= x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32)


def windows(starts: I64Array, ends: I64Array) -> I64Array:
    """Each bead's window, consecutive beads grouped while a window spans at most 10 kb."""
    lab = np.zeros(len(starts), dtype=np.int64)
    w0 = int(starts[0])
    k = 0
    for i in range(1, len(starts)):
        if int(ends[i]) - w0 > _WINDOW_BP:
            k += 1
            w0 = int(starts[i])
        lab[i] = k
    return lab


def bridge(d: F64Array) -> tuple[F64Array, F64Array]:
    """The conditional law of a chain's interior given its two ends.

    Parameters
    ----------
    d
        Target mean distances between the k beads, the ends first and last.

    Returns
    -------
    coef, root
        An interior bead sits at `a + coef[i] (b - a) + root[i] @ z` with `z` standard normal of
        shape (k - 2, 3), one column per coordinate.
    """
    v = _MAXWELL * d * d
    c = 0.5 * (v[1:, :1] + v[:1, 1:] - v[1:, 1:])
    cl = c[:-1, -1]
    cll = max(float(c[-1, -1]), 1e-12)
    coef = cl / cll
    cov = c[:-1, :-1] - np.outer(cl, cl) / cll
    lam, u = np.linalg.eigh(0.5 * (cov + cov.T))
    root = u * np.sqrt(np.clip(lam, 0.0, None))
    return coef, root


def _cooler(uri: str) -> Any:
    with _LOCK:
        if uri not in _COOLERS:
            import cooler

            _COOLERS[uri] = cooler.Cooler(uri)
        return _COOLERS[uri]


def window_counts(uri: str, chrom: str, w_st: I64Array, w_en: I64Array) -> F64Array:
    """Raw counts between windows, each pixel shared by overlap, the diagonal pixels left out."""
    lo, hi = int(w_st[0]), int(w_en[-1])
    key = (uri, chrom, lo, hi)
    hit = _COUNTS.get(key)
    if hit is not None and len(hit[0]) == len(w_st) and np.array_equal(hit[0], w_st):
        return hit[1]
    clr = _cooler(uri)
    res = int(clr.binsize)
    region = f"{chrom}:{max(lo, 0)}-{hi}"
    with _LOCK:
        m = clr.matrix(balance=False, sparse=True).fetch(region).tocsr()
        bins = clr.bins().fetch(region)["start"].to_numpy().astype(np.int64)
    up = sp.triu(m, 1)
    m = (up + up.T).tocsr()
    w = overlap_weights(bins, res, w_st, w_en)
    out = np.asarray((w.T @ m @ w).todense(), dtype=np.float64)
    _COUNTS[key] = (w_st.astype(np.float64), out)
    return out


def _expectation(sep: F64Array, area: F64Array, obs: F64Array) -> tuple[float, float] | None:
    """A power law in separation for the count per unit window area, fitted in log on eight bins a
    decade weighted by area. None when fewer than two bins hold a count."""
    keep = sep > 0
    if keep.sum() < 2:
        return None
    ls = np.log10(sep[keep])
    b = np.floor(ls * 8).astype(np.int64)
    ub, inv = np.unique(b, return_inverse=True)
    o = np.bincount(inv, obs[keep], minlength=len(ub))
    a = np.bincount(inv, area[keep], minlength=len(ub))
    x = np.bincount(inv, ls * area[keep], minlength=len(ub)) / np.maximum(a, 1e-300)
    ok = o > 0
    if ok.sum() < 2:
        return None
    slope, icpt = np.polyfit(x[ok], np.log10(o[ok] / a[ok]), 1, w=np.sqrt(a[ok]))
    return float(icpt), float(slope)


def chromosome_expectation(uri: str, chrom: str) -> tuple[float, float] | None:
    """The power law for the raw count per unit area against separation over a chromosome, read
    at 10 kb from the same file as `uri`. None when the file has no 10 kb resolution.

    Returns
    -------
    intercept, slope
        Of log10 count per bp squared against log10 separation in bp.
    """
    key = (uri, chrom)
    if key in _EXPECTED:
        return _EXPECTED[key]
    path, _, _ = uri.partition("::")
    coarse = f"{path}::/resolutions/{_EXP_RES}"
    try:
        clr = _cooler(coarse)
    except (KeyError, OSError, ValueError):
        _EXPECTED[key] = None
        return None
    length = int(clr.chromsizes[chrom])
    nd = _EXP_SPAN // _EXP_RES
    sums = np.zeros(nd, dtype=np.float64)
    pairs = np.zeros(nd, dtype=np.float64)
    span = min(_EXP_SPAN, length)
    for lo in np.linspace(0, max(length - span, 0), _EXP_WINDOWS).astype(np.int64):
        with _LOCK:
            m = clr.matrix(balance=False, sparse=True).fetch(f"{chrom}:{lo}-{lo + span}").tocoo()
        n = m.shape[0]
        d = m.col - m.row
        keep = (d >= 1) & (d < nd)
        sums += np.bincount(d[keep], m.data[keep].astype(np.float64), minlength=nd)
        pairs[1 : min(n, nd)] += n - np.arange(1, min(n, nd))
    sep = np.arange(nd, dtype=np.float64) * _EXP_RES
    area = pairs * float(_EXP_RES) ** 2
    ok = pairs > 0
    _EXPECTED[key] = _expectation(sep[ok], area[ok], sums[ok])
    return _EXPECTED[key]


def _gap_factor(
    counts: F64Array, mids: F64Array, lens: F64Array, fit: tuple[float, float]
) -> F64Array:
    """The distance factor between windows, observed over expected to the minus third."""
    sep = np.abs(mids[:, None] - mids[None, :])
    area = np.outer(lens, lens)
    exp = 10.0 ** (fit[0] + fit[1] * np.log10(np.maximum(sep, 1.0))) * area
    f = ((counts + _PSEUDO) / (exp + _PSEUDO)) ** (-1.0 / 3.0)
    np.fill_diagonal(f, 1.0)
    return f


def chain_targets(
    starts: list[int],
    ends: list[int],
    law: PolymerLaw,
    stretch: float,
    contact_map: str = "",
    chrom: str = "",
) -> F64Array:
    """Target mean distances between every two beads of one chain, the law's background scaled
    by the window contacts of `contact_map` as a gap's are, or the law alone without one."""
    st = np.asarray(starts, dtype=np.int64)
    en = np.asarray(ends, dtype=np.int64)
    mids = (st + en) / 2.0
    d = law_distances(mids, law)
    if not (contact_map and chrom):
        return d
    order = np.argsort(st, kind="stable")
    lab_sorted = windows(st[order], en[order])
    lab = np.empty_like(lab_sorted)
    lab[order] = lab_sorted
    k = int(lab_sorted[-1]) + 1
    fit = chromosome_expectation(contact_map, chrom)
    if k < 3 or fit is None:
        return d
    w_st = np.array([st[lab == q].min() for q in range(k)], dtype=np.int64)
    w_en = np.array([en[lab == q].max() for q in range(k)], dtype=np.int64)
    cnt = window_counts(contact_map, chrom, w_st, w_en)
    f = _gap_factor(cnt, (w_st + w_en) / 2.0, np.maximum(w_en - w_st, 1).astype(np.float64), fit)
    out = np.clip(d * f[lab[:, None], lab[None, :]], 1.0, stretch * np.maximum(d, 1.0))
    np.fill_diagonal(out, 0.0)
    return out


def conditioned_draw(
    pos: F64Array, fixed: BoolArray, d: F64Array, rng: np.random.Generator
) -> F64Array:
    """The free beads of a chain drawn from the Gaussian whose mean pair distances are `d`,
    conditioned on every fixed bead where `pos` puts it.

    The covariance per coordinate is taken against the first fixed bead as origin, from the
    squared targets as for a bridge, and the free beads' law given the fixed ones is the usual
    Gaussian conditional. A bridge is the case of two fixed beads at the chain's ends.
    """
    v = _MAXWELL * d * d
    o = int(np.flatnonzero(fixed)[0])
    c = 0.5 * (v[o][:, None] + v[o][None, :] - v)
    fx = np.flatnonzero(fixed)
    fx = fx[fx != o]
    fr = np.flatnonzero(~fixed)
    out = pos.copy()
    if fr.size == 0:
        return out
    cff = c[np.ix_(fx, fx)]
    cuf = c[np.ix_(fr, fx)]
    sol = (
        np.linalg.solve(cff + 1e-9 * np.eye(len(fx)), cuf.T).T
        if fx.size
        else np.zeros((fr.size, 0))
    )
    mean = pos[o] + sol @ (pos[fx] - pos[o]) if fx.size else np.tile(pos[o], (fr.size, 1))
    cov = c[np.ix_(fr, fr)] - sol @ cuf.T
    lam, u = np.linalg.eigh(0.5 * (cov + cov.T))
    root = u * np.sqrt(np.clip(lam, 0.0, None))
    out[fr] = mean + root @ rng.normal(size=(fr.size, 3))
    return out


def maxent_start(
    pos: F32Array,
    fixed: BoolArray,
    starts: list[int],
    ends: list[int],
    law: PolymerLaw,
    stretch: float,
    rng: np.random.Generator,
    contact_map: str = "",
    chrom: str = "",
) -> F32Array:
    """The chain with every gap's subanchors drawn from its maximum entropy bridge.

    Parameters
    ----------
    pos
        Bead positions, anchors where the arcs put them. Not modified.
    fixed
        True for anchors.
    starts, ends
        Each bead's genomic interval.
    law
        The run's distance law.
    stretch
        The largest multiple of the background a contact scaled target may reach.
    rng
        The generator every draw comes from, so the start is a function of its seed.
    contact_map
        A cooler URI at fine resolution, or empty for the law alone.
    chrom
        The block's chromosome in the cooler.
    """
    out = pos.astype(np.float64).copy()
    st = np.asarray(starts, dtype=np.int64)
    en = np.asarray(ends, dtype=np.int64)
    mids = (st + en) / 2.0
    anchors = np.flatnonzero(fixed)
    gaps = [(int(a), int(b)) for a, b in zip(anchors[:-1], anchors[1:], strict=True) if b - a >= 2]
    if not gaps:
        return pos

    per_gap: list[tuple[I64Array, F64Array, F64Array, F64Array] | None] = []
    fit: tuple[float, float] | None = None
    if contact_map and chrom:
        pool_s: list[F64Array] = []
        pool_a: list[F64Array] = []
        pool_o: list[F64Array] = []
        for a, b in gaps:
            lab = windows(st[a : b + 1], en[a : b + 1])
            k = int(lab[-1]) + 1
            if k < 3:
                per_gap.append(None)
                continue
            w_st = np.array([st[a : b + 1][lab == q].min() for q in range(k)], dtype=np.int64)
            w_en = np.array([en[a : b + 1][lab == q].max() for q in range(k)], dtype=np.int64)
            cnt = window_counts(contact_map, chrom, w_st, w_en)
            w_mid = (w_st + w_en) / 2.0
            w_len = np.maximum(w_en - w_st, 1).astype(np.float64)
            per_gap.append((lab, cnt, w_mid, w_len))
            i, j = np.triu_indices(k, 1)
            pool_s.append(np.abs(w_mid[j] - w_mid[i]))
            pool_a.append(w_len[i] * w_len[j])
            pool_o.append(cnt[i, j])
        fit = chromosome_expectation(contact_map, chrom)
        if fit is None and pool_s:
            fit = _expectation(
                np.concatenate(pool_s), np.concatenate(pool_a), np.concatenate(pool_o)
            )
    else:
        per_gap = [None] * len(gaps)
    source = contact_map if contact_map and chrom else ""
    if source not in _ANNOUNCED:
        _ANNOUNCED.add(source)
        if not source:
            how = " alone, no contact map"
        elif _EXPECTED.get((contact_map, chrom)) is not None:
            how = f" scaled by window contacts from {source} against the chromosome at 10 kb"
        else:
            how = (
                f" scaled by window contacts from {source} against each call's own windows,"
                " the map has no 10 kb resolution"
            )
        log.status(LOG, "maxent start: gaps drawn from the law%s", how)

    for (a, b), extra in zip(gaps, per_gap, strict=True):
        bg = law_distances(mids[a : b + 1], law)
        d = bg
        if extra is not None and fit is not None:
            lab, cnt, w_mid, w_len = extra
            f = _gap_factor(cnt, w_mid, w_len, fit)[lab[:, None], lab[None, :]]
            d = np.clip(bg * f, 1.0, stretch * np.maximum(bg, 1.0))
            np.fill_diagonal(d, 0.0)
        coef, root = bridge(d)
        pa, pb = out[a], out[b]
        out[a + 1 : b] = pa + coef[:, None] * (pb - pa) + root @ rng.normal(size=(b - a - 1, 3))
    return out.astype(np.float32)
