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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from gnome3d.geometry import smooth_chain


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


def save_transparent(build: Callable[[str], Figure], path: Path, dpi: int = 300) -> None:
    """Write a transparent PNG of a figure whose halos are in its background colour.

    A halo drawn in the background colour is a cut-out only against that background. The
    figure is built twice, on white and on black, and the alpha of every pixel is read off the
    difference, which is exact through antialiased edges. Where the halo cut, the pixel is the
    background in both renders and comes out fully transparent.
    """
    from PIL import Image

    def rgb(bg: str) -> np.ndarray:
        fig = build(bg)
        fig.set_facecolor(bg)
        fig.set_dpi(dpi)
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.float64) / 255.0
        plt_close(fig)
        return arr

    white, black = rgb("white"), rgb("black")
    alpha = np.clip(1.0 - (white - black).mean(axis=2), 0.0, 1.0)
    colour = np.where(alpha[:, :, None] > 1e-3, black / np.maximum(alpha[:, :, None], 1e-3), 0.0)
    out = np.dstack([np.clip(colour, 0, 1), alpha])
    Image.fromarray((out * 255).round().astype(np.uint8), "RGBA").save(path, dpi=(dpi, dpi))


def plt_close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
