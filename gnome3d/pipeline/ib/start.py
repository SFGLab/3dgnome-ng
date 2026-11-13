"""Start positions for a block's subanchors.

The densification puts each gap's subanchors on the straight line between its two anchors, so
the smooth stage starts every coil fully collapsed and has to push it open against a term that
has no force near its radius. The coil start puts them on a random bridge instead: a walk from
one anchor at the bond targets, pulled toward the other so it lands there, each bead placed at
a clear candidate when one of twenty draws is clear of the beads already placed in the gap.
Under the compact rule the clear candidate nearest the gap's midpoint is taken, which keeps
the coil as small as the clearance allows rather than as open as it can be.

Under the law rule the clear candidate is the one whose distances to the beads 2, 4, 8 up to
128 places behind it best match the polymer law in log, so a gap's interior starts at the
law's size instead of packed into a ball, whose interior distances stop growing at its radius.

Only beads of the same gap and its two anchors count for clearance. Pairs across gaps are left
to the smooth stage, whose hard wall only ever lowers their number.
"""

from __future__ import annotations

import numpy as np

from gnome3d.mc.numba.terms import njit
from gnome3d.types import BoolArray, F32Array, F64Array, I64Array

_DRAWS = 20
_LAW_OFFSETS = np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.int64)


@njit(cache=True)
def _bridge_compact_nb(
    a: F64Array, b: F64Array, targets: F64Array, r_avoid: float, normals: F64Array
) -> F64Array:
    """`_bridge` under the compact rule with its draws made beforehand, `normals[j, k]` the
    k-th draw for bead j. The arithmetic is the Python function's, so the result is identical."""
    m = targets.shape[0] - 1
    out = np.zeros((m, 3), dtype=np.float64)
    pts = np.zeros((m + 2, 3), dtype=np.float64)
    pts[0] = a
    pts[1] = b
    n_pts = 2
    cur = a.copy()
    centre = 0.5 * (a + b)
    step = np.zeros(3, dtype=np.float64)
    cand = np.zeros(3, dtype=np.float64)
    best = np.zeros(3, dtype=np.float64)
    for j in range(m):
        remaining = m + 1 - j
        length = targets[j]
        best_gap = -1.0
        best_key = np.inf
        for k in range(normals.shape[1]):
            u0, u1, u2 = normals[j, k, 0], normals[j, k, 1], normals[j, k, 2]
            un = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
            u0, u1, u2 = u0 / un, u1 / un, u2 / un
            step[0] = u0 * length + 0.7 * ((b[0] - cur[0]) / remaining)
            step[1] = u1 * length + 0.7 * ((b[1] - cur[1]) / remaining)
            step[2] = u2 * length + 0.7 * ((b[2] - cur[2]) / remaining)
            sn = np.sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2])
            f = length / sn
            step[0] *= f
            step[1] *= f
            step[2] *= f
            cand[0] = cur[0] + step[0]
            cand[1] = cur[1] + step[1]
            cand[2] = cur[2] + step[2]
            gap = np.inf
            for q in range(n_pts):
                dx = pts[q, 0] - cand[0]
                dy = pts[q, 1] - cand[1]
                dz = pts[q, 2] - cand[2]
                g = np.sqrt(dx * dx + dy * dy + dz * dz)
                if g < gap:
                    gap = g
            if gap >= r_avoid:
                cx = cand[0] - centre[0]
                cy = cand[1] - centre[1]
                cz = cand[2] - centre[2]
                key = np.sqrt(cx * cx + cy * cy + cz * cz)
                if key < best_key:
                    best_key = key
                    best[:] = cand
            elif best_key == np.inf and gap > best_gap:
                best_gap = gap
                best[:] = cand
        cur[:] = best
        out[j] = cur
        pts[n_pts] = cur
        n_pts += 1
    if m > 0:
        dx = b[0] - cur[0]
        dy = b[1] - cur[1]
        dz = b[2] - cur[2]
        dn = np.sqrt(dx * dx + dy * dy + dz * dz)
        t = targets[m]
        rx = b[0] - (cur[0] + dx / dn * t)
        ry = b[1] - (cur[1] + dy / dn * t)
        rz = b[2] - (cur[2] + dz / dn * t)
        for j in range(m):
            f = (j + 1) / (m + 1)
            out[j, 0] += rx * f
            out[j, 1] += ry * f
            out[j, 2] += rz * f
    return out


@njit(cache=True)
def _bridge_law_nb(
    a: F64Array,
    b: F64Array,
    targets: F64Array,
    r_avoid: float,
    normals: F64Array,
    mids: F64Array,
    nu: float,
    s0: float,
    offsets: I64Array,
) -> F64Array:
    """The walk of `_bridge_compact_nb` keyed on the law instead of the gap's midpoint.

    `mids` holds the genomic midpoints of the gap's m + 2 beads, its two anchors at the ends,
    and a bead's law distance to the bead k places behind it is `max(1, (sep / s0)^nu)`.
    """
    m = targets.shape[0] - 1
    out = np.zeros((m, 3), dtype=np.float64)
    pts = np.zeros((m + 2, 3), dtype=np.float64)
    chain = np.zeros((m + 1, 3), dtype=np.float64)
    pts[0] = a
    pts[1] = b
    chain[0] = a
    n_pts = 2
    cur = a.copy()
    step = np.zeros(3, dtype=np.float64)
    cand = np.zeros(3, dtype=np.float64)
    best = np.zeros(3, dtype=np.float64)
    for j in range(m):
        idx = j + 1
        remaining = m + 1 - j
        length = targets[j]
        best_gap = -1.0
        best_key = np.inf
        for k in range(normals.shape[1]):
            u0, u1, u2 = normals[j, k, 0], normals[j, k, 1], normals[j, k, 2]
            un = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
            u0, u1, u2 = u0 / un, u1 / un, u2 / un
            step[0] = u0 * length + 0.7 * ((b[0] - cur[0]) / remaining)
            step[1] = u1 * length + 0.7 * ((b[1] - cur[1]) / remaining)
            step[2] = u2 * length + 0.7 * ((b[2] - cur[2]) / remaining)
            sn = np.sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2])
            f = length / sn
            cand[0] = cur[0] + step[0] * f
            cand[1] = cur[1] + step[1] * f
            cand[2] = cur[2] + step[2] * f
            gap = np.inf
            for q in range(n_pts):
                dx = pts[q, 0] - cand[0]
                dy = pts[q, 1] - cand[1]
                dz = pts[q, 2] - cand[2]
                g = np.sqrt(dx * dx + dy * dy + dz * dz)
                if g < gap:
                    gap = g
            if gap >= r_avoid:
                key = 0.0
                for o in range(offsets.shape[0]):
                    back = offsets[o]
                    if back > idx:
                        break
                    dx = chain[idx - back, 0] - cand[0]
                    dy = chain[idx - back, 1] - cand[1]
                    dz = chain[idx - back, 2] - cand[2]
                    d = np.sqrt(dx * dx + dy * dy + dz * dz)
                    sep = abs(mids[idx] - mids[idx - back])
                    want = (sep / s0) ** nu
                    if want < 1.0:
                        want = 1.0
                    lr = np.log(max(d, 1e-9) / want)
                    key += lr * lr
                if key < best_key:
                    best_key = key
                    best[:] = cand
            elif best_key == np.inf and gap > best_gap:
                best_gap = gap
                best[:] = cand
        cur[:] = best
        out[j] = cur
        chain[idx] = cur
        pts[n_pts] = cur
        n_pts += 1
    if m > 0:
        dx = b[0] - cur[0]
        dy = b[1] - cur[1]
        dz = b[2] - cur[2]
        dn = np.sqrt(dx * dx + dy * dy + dz * dz)
        t = targets[m]
        rx = b[0] - (cur[0] + dx / dn * t)
        ry = b[1] - (cur[1] + dy / dn * t)
        rz = b[2] - (cur[2] + dz / dn * t)
        for j in range(m):
            f = (j + 1) / (m + 1)
            out[j, 0] += rx * f
            out[j, 1] += ry * f
            out[j, 2] += rz * f
    return out


def _bridge(
    a: F64Array, b: F64Array, targets: F64Array, r_avoid: float, rng: np.random.Generator, rule: str
) -> F64Array:
    """Positions of the m subanchors between anchors `a` and `b`, `targets` the m + 1 bonds."""
    m = len(targets) - 1
    placed = [a, b]
    out = np.zeros((m, 3), dtype=np.float64)
    cur = a
    centre = 0.5 * (a + b)
    for j in range(m):
        remaining = m + 1 - j
        length = float(targets[j])
        pts = np.array(placed)
        best: F64Array | None = None
        best_gap = -1.0
        best_key = np.inf
        for _ in range(_DRAWS):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            pull = (b - cur) / remaining
            step = u * length + 0.7 * pull
            step *= length / np.linalg.norm(step)
            cand = cur + step
            gap = float(np.min(np.linalg.norm(pts - cand, axis=1)))
            if gap >= r_avoid:
                if rule == "compact":
                    key = float(np.linalg.norm(cand - centre))
                    if key < best_key:
                        best_key, best = key, cand
                else:
                    best = cand
                    break
            elif best_key == np.inf and gap > best_gap:
                best_gap, best = gap, cand
        assert best is not None
        cur = best
        out[j] = cur
        placed.append(cur)
    # The walk's last bond is whatever is left to `b`. Spread that residual along the bridge so
    # every bond carries a share of it instead of the last one carrying it all.
    end = cur + (b - cur) / np.linalg.norm(b - cur) * float(targets[m]) if m else b
    residual = b - end
    for j in range(m):
        out[j] += residual * ((j + 1) / (m + 1))
    return out


def coil_start(
    pos: F32Array,
    fixed: BoolArray,
    dtn: F32Array,
    r_avoid: float,
    rng: np.random.Generator,
    rule: str = "compact",
    mids: F64Array | None = None,
    nu: float = 0.0,
    s0_bp: int = 1,
) -> F32Array:
    """The chain with every gap's subanchors on a bridge between its anchors.

    Parameters
    ----------
    pos
        Bead positions, anchors where the arcs put them. Not modified.
    fixed
        True for anchors.
    dtn
        Bond targets, `dtn[i]` between beads i and i + 1.
    r_avoid
        Clearance a placed bead keeps from the others of its gap when a draw allows it.
    rng
        The generator every draw comes from, so the start is a function of its seed.
    rule
        `compact` takes the clear candidate nearest the gap's midpoint, `first` the first clear
        one, `law` the clear one whose distances to the beads behind it best match the law.
    mids
        Genomic midpoint in bp of every bead, required under `law`.
    nu
        The law's exponent, under `law`.
    s0_bp
        The separation that is one bead, under `law`.
    """
    out = pos.astype(np.float64).copy()
    anchors = np.flatnonzero(fixed)
    d = dtn.astype(np.float64)
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        if b - a < 2:
            continue
        if rule == "law":
            if mids is None:
                raise ValueError("the law start needs the beads' genomic midpoints")
            normals = rng.normal(size=(b - a - 1, _DRAWS, 3))
            out[a + 1 : b] = _bridge_law_nb(
                out[a],
                out[b],
                d[a:b],
                float(r_avoid),
                normals,
                np.ascontiguousarray(mids[a : b + 1], dtype=np.float64),
                float(nu),
                float(max(s0_bp, 1)),
                _LAW_OFFSETS,
            )
        elif rule == "compact":
            # Every bead takes all of its draws under this rule, so they can be drawn at once
            # in the order the loop would have drawn them, and the walk runs compiled.
            normals = rng.normal(size=(b - a - 1, _DRAWS, 3))
            out[a + 1 : b] = _bridge_compact_nb(out[a], out[b], d[a:b], float(r_avoid), normals)
        else:
            out[a + 1 : b] = _bridge(out[a], out[b], d[a:b], r_avoid, rng, rule)
    return out.astype(np.float32)
