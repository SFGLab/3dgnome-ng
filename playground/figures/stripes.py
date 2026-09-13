"""One conformation coloured in genomic bands, the rest of its ensemble grey behind it.

    python playground/figures/stripes.py ENSEMBLE_DIR OUT.png [--pick 0] [--ref 0] [--band-mb 1.0]
                                         [--title "GM12878 chr1:20-28 Mb"]

Every cif in the directory is one member. The picked member is drawn as a smoothed tube coloured
by genomic position in bands of `--band-mb`, the stripes, with a scale bar in Mb. Every other
member is aligned to it by the Kabsch superposition on the beads they share and drawn as a
thin grey tube behind it, so the ensemble's spread shows around the one that is in colour.
Every piece of every tube is depth sorted with a halo by `tube.py`, so strands stay apart.
Every member is superposed on member `--ref` and the camera looks down that member's least
extended axis, so figures of different picks from one ensemble sit in the same frame. Writes
the PNG, an SVG, and a transparent PNG whose halos are true cut-outs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.tube import camera, draw, project, save_transparent, tube_pieces  # noqa: E402
from playground.validation_battery import _flag, _str_flag, load  # noqa: E402


def kabsch(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    """`moving` superposed on `target`, both (n, 3) in the same bead order."""
    mc = moving.mean(0)
    tc = target.mean(0)
    h = (moving - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (moving - mc) @ r.T + tc


def view_angles(pos: np.ndarray) -> tuple[float, float]:
    """Elevation and azimuth of a camera looking down the structure's least extended axis."""
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    elev = float(np.degrees(np.arcsin(np.clip(n[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(n[1], n[0]))) - 90.0
    return elev, azim


def band_colours(n_bands: int) -> ListedColormap:
    """Distinct colours for neighbouring bands, so each stripe reads against the next."""
    base = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, n_bands))
    return ListedColormap(base)


def build(
    members: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    pick: int,
    ref: int,
    band_mb: float,
    title: str,
    bg: str,
) -> plt.Figure:
    """The figure on background `bg`. Every member is superposed on member `ref` and the camera
    is member `ref`'s, so any pick, and any figure of the same ensemble, sits in one frame."""
    pos_ref, start_ref, _, _ = members[ref]
    aligned = [
        kabsch(p, pos_ref) if len(p) == len(pos_ref) and np.array_equal(st, start_ref) else None
        for p, st, _m, _a in members
    ]
    pos0 = aligned[pick]
    assert pos0 is not None, "the picked member does not share the reference's beads"
    _, _, mid0, _ = members[pick]
    lo = int(np.floor(start_ref.min() / (band_mb * 1e6)) * band_mb * 1e6)  # a round band edge
    band = np.floor((mid0 - lo) / (band_mb * 1e6)).astype(int)
    n_bands = int(band.max()) + 1
    cmap = band_colours(n_bands)
    norm = BoundaryNorm(np.arange(n_bands + 1) - 0.5, n_bands)
    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_axes((0.02, 0.10, 0.96, 0.86))
    ax.set_axis_off()
    elev, azim = view_angles(pos_ref)
    rot = camera(elev, azim)
    centre = pos_ref.mean(0)
    # The other members are the background: drawn first, thin, without halos, so they never
    # cut into the coloured one, and lighter the further back they sit.
    grey = np.tile(np.array([[0.66, 0.66, 0.66, 0.5]]), (len(pos0), 1))
    others = [tube_pieces(q, grey, rot, centre, width=0.7) for k, q in enumerate(aligned) if k != pick and q is not None]
    draw(ax, others, background=bg, halo=0.0, shade=0.0, thin=0.3)
    band_rgba = cmap(norm(band))
    draw(ax, [tube_pieces(pos0, band_rgba, rot, centre, width=3.4)], background=bg, halo=1.6, shade=0.45, thin=0.35)
    # the same box for every pick of this ensemble
    xy_all = np.concatenate([project(q, rot, centre)[0] for q in aligned if q is not None])
    c = xy_all.mean(0)
    r = float(np.abs(xy_all - c).max()) * 1.02
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ink = "#0b0b0b" if bg == "white" else "#f2f2f2"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.7)
    edges = lo + np.arange(n_bands + 1) * band_mb * 1e6
    cb.set_ticks(np.arange(n_bands + 1) - 0.5)
    cb.set_ticklabels([f"{e / 1e6:g}" for e in edges])
    cb.set_label("genomic position, Mb", color=ink)
    cb.ax.tick_params(length=0, colors=ink)
    cb.outline.set_edgecolor(ink)
    if title:
        ax.set_title(f"{title}, one of {len(members)} conformations in colour", fontsize=11, color=ink)
    return fig


def main() -> None:
    pick = int(_flag("--pick", 0))
    ref = int(_flag("--ref", 0))
    band_mb = _flag("--band-mb", 1.0)
    title = _str_flag("--title")
    ens, out = Path(sys.argv[1]), Path(sys.argv[2])
    members = [load(c) for c in sorted(ens.glob("*.cif"))]
    fig = build(members, pick, ref, band_mb, title, "white")
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    transparent = out.with_name(out.stem + "_transparent.png")
    save_transparent(lambda bg: build(members, pick, ref, band_mb, title, bg), transparent, dpi=300)
    print(f"wrote {out}, {out.with_suffix('.svg')} and {transparent}: {len(members)} members")


if __name__ == "__main__":
    main()
