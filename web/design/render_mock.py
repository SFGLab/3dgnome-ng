"""Render the structure shown in the viewer of the design mock, `web/design/site.pen`.

    python web/design/render_mock.py design/figures/rnapii-2026-11-12/fig_inputs/maps_GM12878.npz \
        web/design/images/structure.png

Draws the medoid conformation of the TNFRSF8 locus with CTCF and RNAPII loops as a grey tube with
the promoter beads in the site's promoter colour and the enhancer beads in its enhancer colour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.tube import draw, tube_pieces  # noqa: E402

TSS = 12_063_302
ENHANCER = (12_448_194, 12_450_774)
GREY, PROMOTER, ENH = "#a9a8a3", "#eb6834", "#1baf7a"


def main() -> None:
    maps, out = np.load(sys.argv[1]), Path(sys.argv[2])
    xyz, mid = maps["locus_R_xyz"], maps["locus_R_mid"]
    keep = (mid > 11_990_000) & (mid < 12_520_000)
    xyz, mid = xyz[keep][::2], mid[keep][::2]
    cols = np.tile(matplotlib.colors.to_rgba(GREY), (len(mid), 1))
    cols[np.abs(mid - TSS) < 6_000] = matplotlib.colors.to_rgba(PROMOTER)
    cols[(mid > ENHANCER[0] - 6_000) & (mid < ENHANCER[1] + 6_000)] = matplotlib.colors.to_rgba(ENH)
    centre = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - centre)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    draw(ax, [tube_pieces(xyz, cols, vt, centre, 4.5)], background="white")
    ax.set_aspect("equal")
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
