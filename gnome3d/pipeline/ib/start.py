"""Start positions of the subanchors of a block.

Densification puts the subanchors of a gap on the line between its two anchors. The coil start
puts them on a random bridge, a walk from one anchor at the bond targets, pulled toward the
other. Each bead takes a candidate of twenty draws that is clear of the beads placed in its
gap. The compact rule takes the clear candidate nearest the midpoint of the gap. Only the
beads of one gap and its two anchors count for clearance.

The beads before the first anchor of a chain and after the last are a tail. A tail is a walk from
its anchor at the bond targets. Each bead takes the clear candidate nearest the anchor under the
compact rule and the first clear one otherwise. A chain without an anchor is two tails from its
middle bead.
"""

from __future__ import annotations

import numpy as np

from gnome3d.mc.numba.terms import njit
from gnome3d.types import BoolArray, F32Array, F64Array

_DRAWS = 20


@njit(cache=True)
def _bridge_compact_nb(
    a: F64Array, b: F64Array, targets: F64Array, r_avoid: float, normals: F64Array
) -> F64Array:
    """`_bridge` under the compact rule with the draws made beforehand, `normals[j, k]` the k-th
    draw of bead j. The result is identical to `_bridge`.
    """
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


def _bridge(
    a: F64Array, b: F64Array, targets: F64Array, r_avoid: float, rng: np.random.Generator, rule: str
) -> F64Array:
    """Returns the positions of the m subanchors between anchors `a` and `b`. `targets` are the
    m + 1 bonds.
    """
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
    # The residual to `b` is spread along the bridge. The last bond does not carry all of
    # it.
    end = cur + (b - cur) / np.linalg.norm(b - cur) * float(targets[m]) if m else b
    residual = b - end
    for j in range(m):
        out[j] += residual * ((j + 1) / (m + 1))
    return out


@njit(cache=True)
def _tail_nb(
    a: F64Array, targets: F64Array, r_avoid: float, normals: F64Array, compact: bool
) -> F64Array:
    """Returns the positions of the m beads of a tail that leaves the bead at `a`. `targets` are
    the m bonds in the order of the walk and `normals[j, k]` is the k-th draw of bead j.
    """
    m = targets.shape[0]
    out = np.zeros((m, 3), dtype=np.float64)
    pts = np.zeros((m + 1, 3), dtype=np.float64)
    pts[0] = a
    n_pts = 1
    cur = a.copy()
    cand = np.zeros(3, dtype=np.float64)
    best = np.zeros(3, dtype=np.float64)
    for j in range(m):
        length = targets[j]
        best_gap = -1.0
        best_key = np.inf
        for k in range(normals.shape[1]):
            u0, u1, u2 = normals[j, k, 0], normals[j, k, 1], normals[j, k, 2]
            un = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
            cand[0] = cur[0] + u0 / un * length
            cand[1] = cur[1] + u1 / un * length
            cand[2] = cur[2] + u2 / un * length
            gap = np.inf
            for q in range(n_pts):
                dx = pts[q, 0] - cand[0]
                dy = pts[q, 1] - cand[1]
                dz = pts[q, 2] - cand[2]
                g = np.sqrt(dx * dx + dy * dy + dz * dz)
                if g < gap:
                    gap = g
            if gap >= r_avoid:
                cx = cand[0] - a[0]
                cy = cand[1] - a[1]
                cz = cand[2] - a[2]
                key = np.sqrt(cx * cx + cy * cy + cz * cz)
                if key < best_key:
                    best_key = key
                    best[:] = cand
                    if not compact:
                        break
            elif best_key == np.inf and gap > best_gap:
                best_gap = gap
                best[:] = cand
        cur[:] = best
        out[j] = cur
        pts[n_pts] = cur
        n_pts += 1
    return out


def coil_start(
    pos: F32Array,
    fixed: BoolArray,
    dtn: F32Array,
    r_avoid: float,
    rng: np.random.Generator,
    rule: str = "compact",
) -> F32Array:
    """Returns the chain with the subanchors of every gap on a bridge between its anchors and
    every tail on a walk from its anchor.

    Parameters
    ----------
    pos
        Bead positions with the anchors placed. Not modified.
    fixed
        True for anchors.
    dtn
        Bond targets, `dtn[i]` between beads i and i + 1.
    r_avoid
        The clearance a placed bead keeps from the others of its gap when a draw allows it.
    rng
        The generator of every draw.
    rule
        `compact` takes the clear candidate nearest the midpoint of the gap, `first` the first
        clear one.
    """
    out = pos.astype(np.float64).copy()
    anchors = np.flatnonzero(fixed)
    d = dtn.astype(np.float64)
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        if b - a < 2:
            continue
        if rule == "compact":
            # Under this rule every bead takes all of its draws. They are drawn at once in the
            # order of the loop.
            normals = rng.normal(size=(b - a - 1, _DRAWS, 3))
            out[a + 1 : b] = _bridge_compact_nb(out[a], out[b], d[a:b], float(r_avoid), normals)
        else:
            out[a + 1 : b] = _bridge(out[a], out[b], d[a:b], r_avoid, rng, rule)
    n = out.shape[0]
    compact = rule == "compact"
    # The draws of the tails follow the draws of every gap.
    first = int(anchors[0]) if anchors.size else (n - 1) // 2
    last = int(anchors[-1]) if anchors.size else first
    if last < n - 1:
        normals = rng.normal(size=(n - 1 - last, _DRAWS, 3))
        out[last + 1 :] = _tail_nb(out[last], d[last:], float(r_avoid), normals, compact)
    if first > 0:
        normals = rng.normal(size=(first, _DRAWS, 3))
        back = np.ascontiguousarray(d[:first][::-1])
        out[:first] = _tail_nb(out[first], back, float(r_avoid), normals, compact)[::-1]
    return out.astype(np.float32)
