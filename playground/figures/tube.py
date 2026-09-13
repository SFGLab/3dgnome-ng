"""A chain renderer for matplotlib that keeps strands apart.

matplotlib's 3D axes sort whole collections, not the pieces of a polyline, so two strands of
a chromosome drawn through it merge into one flat tangle. This module does the projection
itself. A chain is smoothed with a Catmull-Rom spline through its beads, projected with a
camera of its own, cut into short pieces, and every piece of every chain in the picture is
sorted by depth and drawn far to near, each with a wider stroke in the background colour
underneath. A nearer strand therefore covers a farther one with a visible rim, the way a
tube would. Depth also darkens and thins a piece, so the back of the structure recedes.

    pieces = [tube_pieces(chain, colours, width) for each chain]
    draw(ax, pieces, background="white")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection


def smooth_chain(pos: np.ndarray, per_bond: int = 4) -> np.ndarray:
    """Catmull-Rom spline through the beads, `per_bond` points per bond, ends included."""
    p = np.asarray(pos, dtype=np.float64)
    if len(p) < 3 or per_bond <= 1:
        return p
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0.0, 1.0, per_bond, endpoint=False)
    t2, t3 = t * t, t * t * t
    b0 = -0.5 * t3 + t2 - 0.5 * t
    b1 = 1.5 * t3 - 2.5 * t2 + 1.0
    b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    b3 = 0.5 * t3 - 0.5 * t2
    p0, p1, p2, p3 = ext[:-3], ext[1:-2], ext[2:-1], ext[3:]
    out = (
        b0[None, :, None] * p0[:, None, :]
        + b1[None, :, None] * p1[:, None, :]
        + b2[None, :, None] * p2[:, None, :]
        + b3[None, :, None] * p3[:, None, :]
    ).reshape(-1, 3)
    return np.vstack([out, p[-1]])


def camera(elev: float, azim: float) -> np.ndarray:
    """Rotation taking world coordinates to view coordinates, the third axis toward the eye."""
    e, a = np.radians(elev), np.radians(azim)
    rz = np.array([[np.cos(a), np.sin(a), 0.0], [-np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(e), np.sin(e)], [0.0, -np.sin(e), np.cos(e)]])
    return rx @ rz


def project(points: np.ndarray, rot: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Screen coordinates and depth toward the eye for world points."""
    v = (points - centre) @ rot.T
    return v[:, :2], v[:, 2]


@dataclass
class Pieces:
    """One chain's drawable chunks: short polylines on screen, depth, colour and width."""

    lines: list[np.ndarray]  # each (k, 2), the stroke
    halos: list[np.ndarray]  # each shorter than its stroke by one piece at the start
    depth: np.ndarray  # (m,)
    colours: np.ndarray  # (m, 4)
    width: float


def tube_pieces(
    chain: np.ndarray,
    colours: np.ndarray,
    rot: np.ndarray,
    centre: np.ndarray,
    width: float,
    per_bond: int = 4,
) -> Pieces:
    """Smooth, project and cut one chain into chunks of one bond each.

    `colours` is one RGBA per bead, carried along the spline by bead index so a band's colour
    runs the whole way between its beads. A chunk's halo covers its own bond; its stroke runs
    one piece further at each end, so the round cap of a neighbouring chunk's halo, which
    intrudes by half its width, is covered again by the stroke drawn after it.
    """
    sm = smooth_chain(chain, per_bond)
    xy, z = project(sm, rot, centre)
    n_bond = max(1, (len(sm) - 1) // max(per_bond, 1))
    lines: list[np.ndarray] = []
    halos: list[np.ndarray] = []
    depth: list[float] = []
    cols: list[np.ndarray] = []
    for b in range(n_bond):
        lo = b * per_bond
        hi = min(lo + per_bond, len(sm) - 1)
        halos.append(xy[lo : hi + 1])
        lines.append(xy[max(lo - 1, 0) : min(hi + 2, len(sm))])
        depth.append(float(z[lo : hi + 1].mean()))
        cols.append(np.asarray(colours)[min(b, len(colours) - 1)])
    return Pieces(lines, halos, np.array(depth), np.array(cols), width)


def draw(
    ax: Axes,
    pieces: list[Pieces],
    background: str | tuple[float, float, float, float] = "white",
    halo: float = 1.6,
    shade: float = 0.45,
    thin: float = 0.35,
) -> None:
    """Draw every chunk of every chain far to near, each halo under its own stroke.

    `halo` is the extra stroke width under each chunk in the background colour, `shade` how
    much the farthest chunks are darkened toward black, `thin` how much thinner they are drawn
    than the nearest.
    """
    if not pieces:
        return
    lines = [ln for p in pieces for ln in p.lines]
    halos = [h for p in pieces for h in p.halos]
    depth = np.concatenate([p.depth for p in pieces])
    cols = np.concatenate([p.colours for p in pieces]).astype(float).copy()
    widths = np.concatenate([np.full(len(p.depth), p.width) for p in pieces])
    order = np.argsort(depth)
    lo, hi = float(depth.min()), float(depth.max())
    near = (depth - lo) / (hi - lo) if hi > lo else np.ones_like(depth)
    cols[:, :3] *= (1.0 - shade * (1.0 - near))[:, None]
    widths = widths * (1.0 - thin * (1.0 - near))
    bg = np.array(matplotlib_colour(background))
    # matplotlib draws a collection's members in order, so one collection sorted by depth is
    # the painter's algorithm; the halo goes in the same collection just under each chunk.
    both: list[np.ndarray] = []
    both_w: list[float] = []
    both_c: list[np.ndarray] = []
    for i in order:
        if halo > 0.0:
            both.append(halos[i])
            both_w.append(float(widths[i]) + halo)
            both_c.append(bg)
        both.append(lines[i])
        both_w.append(float(widths[i]))
        both_c.append(cols[i])
    ax.add_collection(
        LineCollection(both, linewidths=both_w, colors=both_c, capstyle="round", joinstyle="round")
    )
    ax.set_aspect("equal")
    ax.autoscale_view()


def matplotlib_colour(c: str | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    return to_rgba(c)
