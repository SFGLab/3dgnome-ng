"""Every member of an ensemble side by side, superposed and viewed from one camera.

    python playground/figures/ensemble_grid.py ENSEMBLE_DIR OUT.png [--ref 0] [--cols 4]
                                               [--title "GM12878 chr1:20-22.5 Mb"]

The other way of showing an ensemble: each member in its own panel, coloured by genomic
position along the chain, every member superposed on member `--ref` and drawn with that
member's camera and box, so the panels are directly comparable. Tubes through `tube.py`.
Writes the PNG and an SVG beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.stripes import kabsch, view_angles  # noqa: E402
from playground.figures.tube import camera, draw, project, tube_pieces  # noqa: E402
from playground.validation_battery import _flag, _str_flag, load  # noqa: E402


def main() -> None:
    ref = int(_flag("--ref", 0))
    cols = int(_flag("--cols", 4))
    title = _str_flag("--title")
    ens, out = Path(sys.argv[1]), Path(sys.argv[2])
    members = [load(c) for c in sorted(ens.glob("*.cif"))]
    pos_ref, start_ref, mid_ref, _ = members[ref]
    aligned = [kabsch(p, pos_ref) for p, _s, _m, _a in members if len(p) == len(pos_ref)]
    elev, azim = view_angles(pos_ref)
    rot = camera(elev, azim)
    centre = pos_ref.mean(0)
    xy_all = np.concatenate([project(q, rot, centre)[0] for q in aligned])
    c = xy_all.mean(0)
    r = float(np.abs(xy_all - c).max()) * 1.02
    cmap = plt.get_cmap("turbo")
    lo, hi = float(mid_ref.min()), float(mid_ref.max())
    norm = Normalize(lo, hi)
    rgba = cmap(norm(mid_ref))
    rows = int(np.ceil(len(aligned) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 0.9), dpi=150)
    for k, ax in enumerate(np.ravel(axes)):
        ax.set_axis_off()
        if k >= len(aligned):
            continue
        draw(ax, [tube_pieces(aligned[k], rgba, rot, centre, width=1.4)], halo=0.9, shade=0.45, thin=0.35)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_title(f"model {k + 1}", fontsize=10, color="#0b0b0b")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=np.ravel(axes).tolist(), orientation="horizontal", fraction=0.025, pad=0.03, shrink=0.5)
    ticks = np.arange(np.ceil(lo / 5e5) * 5e5, hi + 1, 5e5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t / 1e6:g}" for t in ticks])
    cb.set_label("genomic position, Mb")
    cb.ax.tick_params(length=0)
    if title:
        fig.suptitle(f"{title}, {len(aligned)} conformations superposed on model {ref + 1}", fontsize=11)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    print(f"wrote {out}: {len(aligned)} members")


if __name__ == "__main__":
    main()
