"""One conformation coloured in genomic bands, the rest of its ensemble grey behind it.

    python playground/figures/stripes.py ENSEMBLE_DIR OUT.png [--pick 0] [--band-mb 1.0]
                                         [--title "GM12878 chr1:20-28 Mb"]

Every cif in the directory is one member. The picked member is drawn as a tube coloured by
genomic position in bands of `--band-mb`, the stripes, with a scale bar in Mb. Every other
member is aligned to it by the Kabsch superposition on the beads they share and drawn as a
thin grey translucent line behind it, so the ensemble's spread shows around the one that is
in colour. The camera looks down the picked member's least extended axis. Writes the PNG and
an SVG beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    """Elevation and azimuth that look down the structure's least extended axis."""
    c = pos - pos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    elev = float(np.degrees(np.arcsin(np.clip(n[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(n[1], n[0])))
    return elev, azim


def band_colours(n_bands: int) -> ListedColormap:
    """Distinct colours for neighbouring bands, so each stripe reads against the next."""
    base = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, n_bands))
    return ListedColormap(base)


def main() -> None:
    pick = int(_flag("--pick", 0))
    band_mb = _flag("--band-mb", 1.0)
    title = _str_flag("--title")
    ens, out = Path(sys.argv[1]), Path(sys.argv[2])
    cifs = sorted(ens.glob("*.cif"))
    members = [load(c) for c in cifs]
    pos0, start0, mid0, _ = members[pick]
    lo, hi = int(start0.min()), int(mid0.max())
    band = np.floor((mid0 - lo) / (band_mb * 1e6)).astype(int)
    n_bands = int(band.max()) + 1
    cmap = band_colours(n_bands)
    norm = BoundaryNorm(np.arange(n_bands + 1) - 0.5, n_bands)

    fig = plt.figure(figsize=(9, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    for k, (p, st, _m, _a) in enumerate(members):
        if k == pick:
            continue
        if len(p) != len(pos0) or not np.array_equal(st, start0):
            # members of one run share the bead list; a stray file is skipped rather than misaligned
            continue
        q = kabsch(p, pos0)
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color="#8f8f8f", lw=0.55, alpha=0.32, zorder=1)
    seg = np.stack([pos0[:-1], pos0[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=cmap, norm=norm, linewidths=2.4, zorder=3)
    lc.set_array(band[:-1])
    ax.add_collection3d(lc)
    lim = np.concatenate([pos0] + [kabsch(p, pos0) for k, (p, st, _, _) in enumerate(members) if k != pick and len(p) == len(pos0)])
    c = lim.mean(0)
    r = float(np.abs(lim - c).max()) * 0.92
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    elev, azim = view_angles(pos0)
    ax.view_init(elev=elev, azim=azim)
    cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, shrink=0.7)
    edges = lo + np.arange(n_bands + 1) * band_mb * 1e6
    cb.set_ticks(np.arange(n_bands + 1) - 0.5)
    cb.set_ticklabels([f"{e / 1e6:.0f}" for e in edges])
    cb.set_label("genomic position, Mb")
    cb.ax.tick_params(length=0)
    if title:
        ax.set_title(f"{title}, one of {len(members)} conformations in colour", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".svg"))
    print(f"wrote {out} and {out.with_suffix('.svg')}: {len(members)} members, {n_bands} bands")


if __name__ == "__main__":
    main()
