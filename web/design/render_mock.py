r"""Renders the images of the design mock, `web/design/site.pen`.

    python web/design/render_mock.py design/figures/rnapii-2026-11-12/fig_inputs/maps_GM12878.npz \
        web/design/images

Writes `structure.png` and `structure_ctcf.png`, the TNFRSF8 locus with and without RNAPII
loops, `contacts.png`, the Hi-C map of chr1:20-25 Mb over the model map, with the
`method_*.png` panels of the Method page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.tube import draw, tube_pieces  # noqa: E402

TSS = 12_063_302
ENHANCER = (12_448_194, 12_450_774)
GREY, PROMOTER, ENH = "#a9a8a3", "#eb6834", "#1baf7a"
CONTACTS = LinearSegmentedColormap.from_list("contacts", ["#ffffff", "#f6c9a8", "#d9542b", "#7a1f0c"])


def structure(maps: np.lib.npyio.NpzFile, out: Path, arm: str = "R") -> None:
    xyz, mid = maps[f"locus_{arm}_xyz"], maps[f"locus_{arm}_mid"]
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
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")


def _scaled(m: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
    """Returns log10 of a map scaled to the 2nd to 99.5th percentile of its nonzero cells off the
    diagonal, NaN for a bin without a bead. A model map is zero beyond contact range. The scale is
    set on the cells it fills."""
    v = np.log10(m + np.nanmin(m[m > 0]))
    off = m[np.triu_indices_from(m, 1)] > 0
    lo, hi = np.nanpercentile(v[np.triu_indices_from(v, 1)][off], [2, 99.5])
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    if held is not None:
        v[~held, :] = np.nan
        v[:, ~held] = np.nan
    return v


def contacts(maps: np.lib.npyio.NpzFile, out: Path) -> None:
    hic, model = _scaled(maps["hic"]), _scaled(maps["R"], maps["R_held"])
    upper = np.triu(np.ones_like(hic, bool), 1)
    split = np.where(upper, hic, model)
    cmap = CONTACTS.copy()
    cmap.set_bad("#e9e8e4")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(split, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.plot([-0.5, len(hic) - 0.5], [-0.5, len(hic) - 0.5], color="#151514", lw=0.6)
    ax.axis("off")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")


def method_steps(maps: np.lib.npyio.NpzFile, out: Path) -> None:
    """Draws three panels for the Method page from the CTCF and RNAPII locus. They are the anchors
    alone as points joined in genomic order, the full chain and three structures of the ensemble
    superposed on the first by the Kabsch rotation, all on one view."""
    xyz, mid, anc = maps["locus_R_xyz"], maps["locus_R_mid"], maps["locus_R_anchor"]
    keep = (mid > 11_990_000) & (mid < 12_520_000)
    xyz, mid, anc = xyz[keep], mid[keep], anc[keep]
    centre = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - centre)
    grey = np.tile(matplotlib.colors.to_rgba(GREY), (len(mid[::2]), 1))

    fig, ax = plt.subplots(figsize=(4, 4))
    a = (xyz[anc] - centre) @ vt.T
    ax.plot(a[:, 0], a[:, 1], color="#c9c8c3", lw=0.8, zorder=1)
    ax.scatter(a[:, 0], a[:, 1], s=14, color="#151514", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "method_anchors.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    draw(ax, [tube_pieces(xyz[::2], grey, vt, centre, 3.0)], background="white")
    ax.scatter(*((xyz[anc] - centre) @ vt.T)[:, :2].T, s=10, color="#151514", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "method_chain.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    chains = []
    for j in range(3):
        x, m = maps[f"ensemble_R_{j}_xyz"], maps[f"ensemble_R_{j}_mid"]
        chains.append(x[(m > 11_990_000) & (m < 12_520_000)][::3])
    n = min(len(c) for c in chains)
    ref = chains[0][:n] - chains[0][:n].mean(0)
    _, _, v = np.linalg.svd(ref)
    pieces = []
    for x, col in zip(chains, ["#151514", "#2a78d6", "#9fc3ee"], strict=True):
        b = x[:n] - x[:n].mean(0)
        u, _, wt = np.linalg.svd(b.T @ ref)
        d = np.sign(np.linalg.det(u @ wt))
        b = b @ (u @ np.diag([1, 1, d]) @ wt)
        pieces.append(tube_pieces(b, np.tile(matplotlib.colors.to_rgba(col), (n, 1)), v, np.zeros(3), 1.8))
    draw(ax, pieces, background="white")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out / "method_ensemble.png", dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)


def distance_law(out: Path, nu: float = 0.34) -> None:
    """Draws the target distance against genomic separation for a pair without a loop and for loops
    of three strengths, by the formulas of `gnome3d.polymer.PolymerLaw` with one bead at 1 kb and
    the exponent measured on GM12878 Hi-C."""
    sep = np.logspace(3, 7, 200)
    bg = np.maximum(1.0, (sep / 1000.0) ** nu)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(sep, bg, color="#151514", lw=2, label="no loop")
    for q, col in [(0.5, "#9fc3ee"), (2.0, "#5a9be0"), (8.0, "#2a78d6")]:
        ax.plot(sep, 1.0 + (bg - 1.0) / (1.0 + q), color=col, lw=2, label=f"loop, strength {q:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("genomic separation, bp")
    ax.set_ylabel("target distance, bead units")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "method_law.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    maps, out = np.load(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    structure(maps, out / "structure.png")
    structure(maps, out / "structure_ctcf.png", "C")
    contacts(maps, out / "contacts.png")
    method_steps(maps, out)
    distance_law(out)
    print(out)


if __name__ == "__main__":
    main()
