"""Start positions for a block's subanchors.

The densification puts each gap's subanchors on the straight line between its two anchors, so
the smooth stage starts every coil fully collapsed and has to push it open against a term that
has no force near its radius. The coil start puts them on a random bridge instead: a walk from
one anchor at the bond targets, pulled toward the other so it lands there, each bead placed at
a clear candidate when one of twenty draws is clear of the beads already placed in the gap.
Under the compact rule the clear candidate nearest the gap's midpoint is taken, which keeps
the coil as small as the clearance allows rather than as open as it can be.

Only beads of the same gap and its two anchors count for clearance. Pairs across gaps are left
to the smooth stage, whose hard wall only ever lowers their number.
"""

from __future__ import annotations

import numpy as np

from gnome3d.types import BoolArray, F32Array, F64Array

_DRAWS = 20


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
        one.
    """
    out = pos.astype(np.float64).copy()
    anchors = np.flatnonzero(fixed)
    d = dtn.astype(np.float64)
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        if b - a < 2:
            continue
        out[a + 1 : b] = _bridge(out[a], out[b], d[a:b], r_avoid, rng, rule)
    return out.astype(np.float32)
