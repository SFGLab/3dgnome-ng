"""The two RNAPII figures and their supplement, `design/2026-11-12-rnapii-figures.md`.

Figure 1: gnome3d matches chromatin structure better than cudaMMC and reads data cudaMMC cannot.
Figure 2: RNAPII loops help the functional reading of the models, not the structure itself.

    python playground/rnapii/figures_two.py <fig_inputs> <readouts> <out_dir>

`fig_inputs` holds `ep_<LINE>.tsv` from `ep_pairs.py`, `maps_GM12878.npz` from `model_maps.py`
and `atlas_GM12878_locus.bed`; `readouts` is the readout directory of the RNAPII comparison.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure, SubFigure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fig_data import ARM_COLOUR, ARM_NAME, agreement, link_draws, trio_paired  # noqa: E402

from gnome3d.geometry import smooth_chain  # noqa: E402

LINES = ("GM12878", "H1ESC", "HFFC6")
LINE_LABEL = {"GM12878": "GM12878", "H1ESC": "H1", "HFFC6": "HFFc6\n(RNAPII loops from BJ)"}
PAIR_LABEL = {
    "GM12878 vs H1ESC": "GM12878 vs H1",
    "GM12878 vs HFFC6": "GM12878 vs HFFc6",
    "H1ESC vs HFFC6": "H1 vs HFFc6",
}
MEASURE_LABEL = {
    "hic": "stratum adjusted correlation\nwith Hi-C at 25 kb",
    "microc": "correlation with\nMicro-C at 5 kb",
}
INK, MUTED, GRID, CONTROL = "#1a1a1a", "#5a5955", "#e4e3df", "#b5b3ad"
LOCUS = {"gene": "TNFRSF8", "tss": 12_063_302, "enhancer": (12_448_194, 12_450_774)}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK,
        "legend.frameon": False,
    }
)


def joy(
    ax: plt.Axes,
    groups: list[tuple[str, list[tuple[str, str, np.ndarray]]]],
    xlabel: str,
    min_n: int = 15,
    clip: tuple[float, float] | None = None,
) -> None:
    """Stacked ridges, one row per group, one filled density per series.

    Each series is a label, a colour and its values. A series with fewer than `min_n` values is
    drawn as points with its count.
    """
    vals = np.concatenate([v[np.isfinite(v)] for _, ser in groups for _, _, v in ser])
    lo, hi = clip or (np.percentile(vals, 0.5), np.percentile(vals, 99.5))
    pad = 0.05 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 400)
    height = 0.9
    for row, (name, series) in enumerate(groups[::-1]):
        base = row * 1.0
        dens = []
        for _, colour, v in series:
            v = v[np.isfinite(v)]
            if len(v) < min_n:
                dens.append((colour, v, None))
            else:
                dens.append((colour, v, gaussian_kde(v)(xs)))
        top = max((d.max() for _, _, d in dens if d is not None), default=1.0)
        for colour, v, d in dens:
            if d is None:
                ax.plot(v, np.full(len(v), base + 0.1), "o", color=colour, ms=3)
                ax.text(hi, base + 0.1, f"n = {len(v)}", ha="right", va="bottom", color=MUTED, fontsize=7)
                continue
            y = base + height * d / top
            if colour != CONTROL:
                ax.fill_between(xs, base, y, color=colour, alpha=0.3, lw=0)
            ax.plot(xs, y, color=colour, lw=1.2)
            med = float(np.median(v))
            ax.plot([med, med], [base, base + height * float(np.interp(med, xs, d)) / top], color=colour, lw=1)
        ax.axhline(base, color=GRID, lw=0.8, zorder=0)
        ax.text(lo - pad - 0.01 * (hi - lo), base + 0.45, name, ha="right", va="center", color=INK, fontsize=8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.05, len(groups) + 0.1)
    ax.set_xlabel(xlabel)


def dumbbells(ax: plt.Axes, df: pd.DataFrame, a: str, b: str, measure: str, xlim: tuple[float, float]) -> None:
    """One row per line from arm `a` to arm `b`; a line without the measure says so."""
    sub = df[df.measure == measure].set_index(["line", "arm"]).value
    for row, line in enumerate(LINES[::-1]):
        if (line, a) not in sub or (line, b) not in sub:
            ax.text(np.mean(xlim), row, "no Micro-C" if measure == "microc" else "no map",
                    ha="center", va="center", color=MUTED, fontsize=7, style="italic")
            continue
        va, vb = sub[(line, a)], sub[(line, b)]
        ax.plot([va, vb], [row, row], color="#9a9893", lw=1.6, zorder=2)
        ax.plot(va, row, "o", color=ARM_COLOUR[a], ms=7, zorder=3)
        ax.plot(vb, row, "o", color=ARM_COLOUR[b], ms=7, zorder=3)
        ax.text(vb, row + 0.28, f"{vb:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(va, row - 0.42, f"{va:.2f}", ha="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(LINES)))
    ax.set_yticklabels([LINE_LABEL[x] for x in LINES[::-1]])
    ax.set_ylim(-0.6, len(LINES) - 0.3)
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel(MEASURE_LABEL[measure] + ", higher is better")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _triangle(ax: plt.Axes, mat: np.ndarray, held: np.ndarray, start: int, binsize: int, depth: int,
              vmin: float, vmax: float, label: str) -> None:
    """The upper triangle of `mat` rotated flat, up to `depth` bins from the diagonal; bins with no
    beads are grey."""
    n = len(mat)
    norm = np.log10(mat / max(np.mean(np.diag(mat, 1)[np.diag(mat, 1) > 0]), 1e-12) + 1e-3)
    gap = ~(held[:, None] & held[None, :])
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    x = (start + (i + j) / 2 * binsize) / 1e6
    y = (j - i) / 2 * binsize / 1e6
    z = np.ma.masked_where(np.triu(np.ones((n, n)), 0) == 0, np.where(gap, np.nan, norm))
    z = np.ma.masked_where(np.subtract.outer(np.arange(n), np.arange(n)) < -depth, z)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#ffffff")
    ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    gz = np.ma.masked_where(~gap | (np.triu(np.ones((n, n)), 0) == 0)
                            | (np.subtract.outer(np.arange(n), np.arange(n)) < -depth), np.ones((n, n)))
    ax.pcolormesh(x, y, gz, cmap=matplotlib.colors.ListedColormap(["#dcdad5"]), shading="flat", rasterized=True)
    ax.set_ylim(0, depth / 2 * binsize / 1e6)
    ax.set_xlim(start / 1e6, (start + n * binsize) / 1e6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(0.01, 0.92, label, transform=ax.transAxes, fontsize=8, color=INK, va="top")


def panel_inputs(ax: plt.Axes, chains: list[np.ndarray]) -> None:
    """Schematic: the inputs each engine reads and the ensemble gnome3d produces."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 6)
    ax.axis("off")
    ax.plot([0.3, 4.6], [1.6, 1.6], color=INK, lw=1.5)
    for s, e, h in ((0.6, 3.0, 1.9), (1.4, 2.6, 0.9)):
        mid = (s + e) / 2
        t = np.linspace(0, np.pi, 50)
        ax.plot(mid + (e - s) / 2 * np.cos(t), 1.6 + h * np.sin(t), color=ARM_COLOUR["C"], lw=2)
    for s, e, h in ((1.0, 2.0, 0.8), (2.4, 4.2, 1.5), (3.0, 3.6, 0.5)):
        mid = (s + e) / 2
        t = np.linspace(0, np.pi, 50)
        ax.plot(mid + (e - s) / 2 * np.cos(t), 1.6 + h * np.sin(t), color=ARM_COLOUR["R"], lw=2)
    xs = np.linspace(0.3, 4.6, 24)
    for k in range(1, 6):
        ax.fill_between(xs, 1.55 - 0.13 * k, 1.55 - 0.13 * (k - 1), color="#d55e3d", alpha=0.45 / k, lw=0)
    ax.add_patch(Rectangle((3.3, 1.5), 0.5, 0.2, color="#6b4fa1"))
    ax.text(0.3, 4.0, "CTCF loops", color=ARM_COLOUR["C"], fontsize=8, fontweight="bold")
    ax.text(2.3, 4.0, "RNAPII loops", color=ARM_COLOUR["R"], fontsize=8, fontweight="bold")
    ax.text(0.3, 0.55, "Hi-C contacts", color="#b0452a", fontsize=8)
    ax.text(3.3, 1.0, "structural\nvariants", color="#6b4fa1", fontsize=7, va="top")
    ax.add_patch(FancyArrowPatch((4.9, 2.4), (6.0, 2.4), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))
    for k, xyz in enumerate(chains):
        pts = xyz[::40] - xyz[::40].mean(0)
        _, _, vt = np.linalg.svd(pts)
        flat = pts @ vt.T[:, :2]
        flat = flat / np.abs(flat).max() * 0.85
        curve = smooth_chain(np.column_stack([flat, np.zeros(len(flat))]), per_bond=6)[:, :2]
        ax.plot(curve[:, 0] + 6.9 + k * 1.05, curve[:, 1] + 2.4, color=ARM_COLOUR["C"], lw=1.6, alpha=0.9,
                solid_capstyle="round")
    ax.text(7.95, 0.9, "structure ensemble", ha="center", fontsize=8, color=INK)
    ax.text(0.3, 5.5, "cudaMMC: CTCF loops", fontsize=8, color=MUTED)
    ax.text(0.3, 5.0, "gnome3d: CTCF loops, RNAPII loops, Hi-C contacts, structural variants", fontsize=8,
            color=INK)


def _render(ax: plt.Axes, xyz: np.ndarray, mid: np.ndarray, rot: np.ndarray, colour: str, title: str) -> None:
    """One locus conformation in projection, chr1:11.99-12.52 Mb, every third bead. The smooth chain
    and boxes around the promoter and the enhancer, the website viewer's rule of the bead stretch
    covering each element, with their 3D distance in the caption."""
    keep = (mid > 11_990_000) & (mid < 12_520_000)
    xyz, mid = xyz[keep][::3], mid[keep][::3]
    p = (xyz - xyz.mean(0)) @ rot
    centres: list[np.ndarray] = []
    curve = smooth_chain(p, per_bond=6)
    depth = curve[:, 2]
    shade = (depth - depth.min()) / max(np.ptp(depth), 1e-9)
    for k in range(len(curve) - 1):
        ax.plot(curve[k : k + 2, 0], curve[k : k + 2, 1], color=colour, lw=0.6 + 1.4 * shade[k], alpha=0.3 + 0.65 * shade[k],
                solid_capstyle="round")
    for (s, e), name, c in (
        ((LOCUS["tss"] - 1000, LOCUS["tss"] + 1000), "promoter", "#111111"),
        (LOCUS["enhancer"], "enhancer", "#7a3fb0"),
    ):
        k = int(np.argmin(np.abs(mid - (s + e) / 2)))
        pts = p[max(0, k - 2) : k + 3, :2]
        lo, hi = pts.min(0) - 0.6, pts.max(0) + 0.6
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=True, facecolor=c, alpha=0.18, edgecolor=c, lw=1.5))
        ax.text(lo[0], hi[1] + 0.3, name, fontsize=7, color=c, va="bottom")
        centres.append(p[k])
    gap = float(np.linalg.norm(centres[0] - centres[1]))
    ax.plot([centres[0][0], centres[1][0]], [centres[0][1], centres[1][1]], ls=":", color=INK, lw=1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.5, -0.04, f"{textwrap.fill(title, 22)}\nenhancer to promoter {gap:.1f} bead units",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK)


def _row_title(sub: SubFigure, letter: str, text: str, width: int = 88) -> None:
    """A panel title across a row, wrapped to the figure width."""
    body = textwrap.fill(text, width, subsequent_indent="    ")
    sub.suptitle(f"{letter}  {body}", x=0.0, ha="left", fontweight="bold", fontsize=9, color=INK)


def _handles(arms: tuple[str, ...], extra: list[tuple[str, str]] | None = None) -> list[plt.Line2D]:
    out = [plt.Line2D([], [], marker="o", ls="", color=ARM_COLOUR[a], label=ARM_NAME[a], ms=6) for a in arms]
    for label, colour in extra or []:
        out.append(plt.Line2D([], [], color=colour, lw=1.5, label=label))
    return out


def _joy_groups(draws: pd.DataFrame, arms: tuple[str, ...]) -> list[tuple[str, list[tuple[str, str, np.ndarray]]]]:
    return [
        (PAIR_LABEL[p], [(a, ARM_COLOUR[a], draws[(draws.pair == p) & (draws.arm == a)].value.to_numpy()) for a in arms])
        for p in PAIR_LABEL
    ]


def _agreement_row(sub: SubFigure, ag: pd.DataFrame, a: str, b: str) -> None:
    axes = sub.subplots(1, 2, sharey=True)
    for ax, measure in zip(axes, ("hic", "microc"), strict=True):
        vals = ag[ag.measure == measure].value
        dumbbells(ax, ag, a, b, measure, (min(0.0, vals.min() - 0.03), vals.max() + 0.05))
    sub.legend(handles=_handles((a, b)), loc="outside lower center", ncol=2, fontsize=7)


def figure_one(inputs: Path, readouts: Path, out: Path) -> Figure:
    """gnome3d against cudaMMC on structure, and the enhancer to promoter distances only RNAPII
    input reaches."""
    ag = agreement(readouts)
    maps = np.load(inputs / "maps_GM12878.npz")
    ep = pd.concat(pd.read_csv(inputs / f"ep_{line}.tsv", sep="\t") for line in LINES)

    fig = plt.figure(figsize=(7.2, 10.0), layout="constrained")
    rows = fig.subfigures(4, 1, height_ratios=[0.8, 1.55, 1.05, 1.45], hspace=0.04)

    _row_title(rows[0], "a", "Input data read by cudaMMC and by gnome3d")
    panel_inputs(rows[0].subplots(), [maps["locus_C_xyz"], maps["locus_R_xyz"], maps["locus_C_xyz"][::-1]])

    _row_title(
        rows[1], "b",
        "Contact maps of GM12878 chr1:20-25 Mb from Hi-C, from ten cudaMMC structures and from ten gnome3d "
        "structures with CTCF loops, on one colour scale",
    )
    axes = rows[1].subplots(3, 1, sharex=True)
    start, binsize = int(maps["bin_starts"][0]), int(maps["bin_starts"][1] - maps["bin_starts"][0])
    logs = []
    for key in ("hic", "E", "C"):
        m = maps[key]
        d1 = np.diag(m, 1)
        logs.append(np.log10(m / np.mean(d1[d1 > 0]) + 1e-3)[np.triu_indices(len(m), 1)])
    allv = np.concatenate(logs)
    vmin, vmax = np.percentile(allv[np.isfinite(allv)], [40, 99.5])
    for ax, (key, label) in zip(axes, (("hic", "Hi-C"), ("E", ARM_NAME["E"]), ("C", ARM_NAME["C"])), strict=True):
        held = np.ones(len(maps["hic"]), bool) if key == "hic" else maps[f"{key}_held"]
        _triangle(ax, maps[key], held, start, binsize, 60, vmin, vmax, label)
    axes[-1].set_xlabel("position on chr1, Mb. Grey bins hold no model beads")

    _row_title(rows[2], "c", "Correlation of simulated contact maps with Hi-C and Micro-C, cudaMMC and gnome3d with CTCF loops")
    _agreement_row(rows[2], ag, "E", "C")

    _row_title(
        rows[3], "d",
        "3D distance between enhancers and promoters joined by an RNAPII loop and by no CTCF loop, chr1, "
        "against pairs at the same genomic separation joined by no loop",
    )
    ax = rows[3].subplots()
    groups = []
    for line in LINES:
        e = ep[ep.line == line]
        only = e[e.set == "rnapii_only"]
        ctrl = e[(e.set == "control") & (e.arm == "R")]
        groups.append(
            (
                LINE_LABEL[line],
                [
                    ("control", CONTROL, ctrl.distance.to_numpy()),
                    ("C", ARM_COLOUR["C"], only[only.arm == "C"].distance.to_numpy()),
                    ("R", ARM_COLOUR["R"], only[only.arm == "R"].distance.to_numpy()),
                ],
            )
        )
    joy(ax, groups, "enhancer to promoter distance in the models, bead units, mean of ten structures", clip=(0, 14))
    rows[3].legend(
        handles=_handles(("C", "R"), [("no loop, matched separation, CTCF and RNAPII loops", CONTROL)]),
        loc="outside lower center", ncol=3, fontsize=7,
    )
    _save(fig, out / "rnapii_figure1")
    return fig


def figure_two(inputs: Path, readouts: Path, out: Path) -> Figure:
    """The link to expression, agreement with contact maps, the trio individuals and one locus,
    gnome3d with and without RNAPII loops."""
    ag = agreement(readouts)
    draws = link_draws(readouts, ("C", "R"))
    trio = trio_paired(readouts / "trio")
    maps = np.load(inputs / "maps_GM12878.npz")

    fig = plt.figure(figsize=(7.2, 10.0), layout="constrained")
    rows = fig.subfigures(3, 1, height_ratios=[1.25, 1.05, 1.5], hspace=0.04)

    _row_title(
        rows[0], "e",
        "Correlation between the change in enhancer to promoter distance and the change in gene expression "
        "across pairs of cell lines, bootstrap distribution over genes",
    )
    ax = rows[0].subplots()
    joy(ax, _joy_groups(draws, ("C", "R")),
        "minus Spearman correlation over genes, 2,000 block bootstrap draws, larger is a stronger link")
    rows[0].legend(handles=_handles(("C", "R")), loc="outside lower center", ncol=2, fontsize=7)

    _row_title(rows[1], "f", "Correlation of simulated contact maps with Hi-C and Micro-C, gnome3d with and without RNAPII loops")
    _agreement_row(rows[1], ag, "C", "R")

    bottom = rows[2].subfigures(1, 2, width_ratios=[0.85, 1.15], wspace=0.03)
    _row_title(bottom[0], "g", "Nine trio individuals, each model set against the individual's own Hi-C", 40)
    titles = {
        "loop_pileup": "pile-up at RNAPII loops,\nmodel relative to Hi-C",
        "hic": "stratum adjusted\ncorrelation with Hi-C",
    }
    for ax, measure in zip(bottom[0].subplots(1, 2), ("loop_pileup", "hic"), strict=True):
        w = trio[trio.measure == measure].pivot(index="person", columns="arm", values="value")
        up = int((w.R > w.C).sum())
        for _, r in w.iterrows():
            ax.plot([0, 1], [r.C, r.R], color="#9a9893", lw=0.9, zorder=1)
        ax.scatter(np.zeros(len(w)), w.C, color=ARM_COLOUR["C"], s=14, zorder=2)
        ax.scatter(np.ones(len(w)), w.R, color=ARM_COLOUR["R"], s=14, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CTCF\nloops", "CTCF and\nRNAPII loops"], fontsize=7)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"{titles[measure]}\nhigher in {up} of {len(w)}", fontsize=7, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)

    _row_title(bottom[1], "h", f"The {LOCUS['gene']} locus in GM12878, medoid of ten structures per model set", 52)
    xyz_c = maps["locus_C_xyz"]
    _, _, vt = np.linalg.svd(xyz_c - xyz_c.mean(0))
    for ax, arm in zip(bottom[1].subplots(1, 2), ("C", "R"), strict=True):
        _render(ax, maps[f"locus_{arm}_xyz"], maps[f"locus_{arm}_mid"], vt.T, ARM_COLOUR[arm], ARM_NAME[arm])
    _save(fig, out / "rnapii_figure2")
    return fig


def figure_supplement(inputs: Path, readouts: Path, out: Path) -> Figure:
    """The merged anchors arm beside the other three on agreement and the link to expression."""
    ag = agreement(readouts)
    arms = ("E", "C", "R", "M")
    draws = link_draws(readouts, arms, n=1000)
    fig = plt.figure(figsize=(7.2, 7.0), layout="constrained")
    rows = fig.subfigures(2, 1, height_ratios=[1.0, 1.4], hspace=0.04)
    _row_title(rows[0], "a", "Correlation of simulated contact maps with Hi-C and Micro-C, all four model sets")
    for k, (ax, measure) in enumerate(zip(rows[0].subplots(1, 2, sharey=True), ("hic", "microc"), strict=True)):
        sub = ag[ag.measure == measure]
        for row, line in enumerate(LINES[::-1]):
            s = sub[sub.line == line].set_index("arm").value
            if s.empty:
                ax.text(0.5, row, "no Micro-C", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        color=MUTED, fontsize=7, style="italic")
                continue
            for j, a in enumerate(arms):
                if a in s:
                    ax.plot(s[a], row + (1.5 - j) * 0.14, "o", color=ARM_COLOUR[a], ms=6)
        ax.set_yticks(range(len(LINES)))
        ax.set_yticklabels([LINE_LABEL[x] for x in LINES[::-1]])
        ax.set_ylim(-0.6, len(LINES) - 0.4)
        ax.set_xlabel(MEASURE_LABEL[measure] + ", higher is better")
        ax.grid(axis="x", color=GRID, lw=0.6)
    _row_title(
        rows[1], "b",
        "Correlation between the change in enhancer to promoter distance and the change in gene expression, "
        "all four model sets",
    )
    joy(rows[1].subplots(), _joy_groups(draws, arms),
        "minus Spearman correlation over genes, 1,000 block bootstrap draws, larger is a stronger link")
    rows[1].legend(handles=_handles(arms), loc="outside lower center", ncol=2, fontsize=7)
    _save(fig, out / "rnapii_supplement")
    return fig


def _save(fig: Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))


def main() -> None:
    inputs, readouts, out = (Path(x) for x in sys.argv[1:4])
    figure_one(inputs, readouts, out)
    figure_two(inputs, readouts, out)
    figure_supplement(inputs, readouts, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
