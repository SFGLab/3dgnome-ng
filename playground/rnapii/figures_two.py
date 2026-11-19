"""The two RNAPII figures and their supplement, `design/2026-11-12-rnapii-figures.md`.

Figure 1: gnome3d matches chromatin structure better than cudaMMC and reads data cudaMMC cannot.
Figure 2: RNAPII loops help the functional reading of the models, not the structure itself.

    python playground/rnapii/figures_two.py <fig_inputs> <readouts> <out_dir> [<trio readouts>]

`fig_inputs` holds `ep_<LINE>.tsv` from `ep_pairs.py`, `maps_GM12878.npz` from `model_maps.py`
and `atlas_GM12878_locus.bed`; `readouts` is the readout directory of the RNAPII comparison.
The trio readouts are `<readouts>/trio` unless given. Micro-C is read as the share of its loops
the models place when `<LINE>_microc_loops.json` is there, and as a map correlation otherwise.
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

from fig_data import ARM_COLOUR, ARM_NAME, agreement, by_chromosome, link_draws, microc_classes, trio_paired  # noqa: E402
from fig_stats import LEGEND, ep_tests, link_difference, mark, paired, stars, unpaired  # noqa: E402

from playground.figures.tube import draw, tube_pieces  # noqa: E402

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
    "microc_loops": "share of Micro-C loops placed,\nabove moved loops",
}
CLASS_LABEL = {"CTCF": "a CTCF loop", "RNAPII alone": "an RNAPII loop\nand no CTCF loop", "none": "no ChIA-PET loop"}
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
    medians: dict[str, str] | None = None,
    notes: dict[str, list[str]] | None = None,
) -> None:
    """Stacked ridges, one row per group, one filled density per series.

    Each series is a label, a colour and its values. A series with fewer than `min_n` values is
    drawn as points with its count. With `medians`, a map from series label to a short name,
    every row lists the median of each series beside its ridges. With `notes`, a map from group
    name to lines of text, a row carries its lines, under the medians when there are medians and
    under the name of the row otherwise. They hold the results of the tests.
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
        if medians:
            x0 = lo + 0.7 * (hi - lo)
            for k, (label, colour, v) in enumerate(series):
                y = base + 0.82 - 0.2 * k
                ax.plot(x0, y, "s", color=colour, ms=4, zorder=4)
                ax.text(x0 + 0.015 * (hi - lo), y, f"median {np.nanmedian(v):.1f}, {medians[label]}",
                        va="center", ha="left", fontsize=6.5, color=INK, zorder=4)
        for k, text in enumerate((notes or {}).get(name, [])):
            if medians:
                ax.text(lo + 0.7 * (hi - lo) + 0.015 * (hi - lo), base + 0.82 - 0.2 * len(series) - 0.15 * k, text,
                        va="center", ha="left", fontsize=6.5, color=MUTED, zorder=4)
            else:
                ax.text(lo - pad - 0.01 * (hi - lo), base + 0.2 - 0.3 * k, text.replace(", ", "\n", 1), ha="right", va="center",
                        color=MUTED, fontsize=6.5, linespacing=1.15)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.05, len(groups) + 0.1)
    ax.set_xlabel(xlabel)


def dumbbells(ax: plt.Axes, df: pd.DataFrame, a: str, b: str, measure: str, xlim: tuple[float, float],
              per_chrom: pd.DataFrame | None = None) -> None:
    """One row per line from arm `a` to arm `b`; a line without the measure says so.

    With `per_chrom`, the readouts one value a chromosome, each row carries the mark of a signed
    rank test of the two arms over the chromosomes both have.
    """
    sub = df[df.measure == measure].set_index(["line", "arm"]).value
    tests = {}
    if per_chrom is not None and len(per_chrom):
        t = per_chrom[per_chrom.measure == measure].pivot_table(index=["line", "chrom"], columns="arm", values="value")
        for line in LINES:
            if line in t.index.get_level_values(0) and a in t.columns and b in t.columns:
                w = t.loc[line].dropna(subset=[a, b])
                if len(w) >= 6:
                    tests[line] = (paired(w[a].to_numpy(), w[b].to_numpy())[0], len(w))
    for row, line in enumerate(LINES[::-1]):
        if (line, a) not in sub or (line, b) not in sub:
            ax.text(np.mean(xlim), row, "no Micro-C" if measure.startswith("microc") else "no map",
                    ha="center", va="center", color=MUTED, fontsize=7, style="italic")
            continue
        va, vb = sub[(line, a)], sub[(line, b)]
        ax.plot([va, vb], [row, row], color="#9a9893", lw=1.6, zorder=2)
        ax.plot(va, row, "o", color=ARM_COLOUR[a], ms=7, zorder=3)
        ax.plot(vb, row, "o", color=ARM_COLOUR[b], ms=7, zorder=3)
        ax.text(vb, row + 0.28, f"{vb:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(va, row - 0.42, f"{va:.2f}", ha="center", fontsize=7, color=MUTED)
        if line in tests:
            # Over the line between the two dots when they are far apart, else at the end of the
            # row the dots are not at.
            text, span = mark(tests[line][0]), xlim[1] - xlim[0]
            if abs(va - vb) > 0.2 * span:
                ax.text((va + vb) / 2, row + 0.2, text, ha="center", va="center", fontsize=6.5, color=MUTED)
            elif min(va, vb) - xlim[0] > 0.3 * span:
                ax.text(0.02, row, text.replace(", ", "\n", 1), transform=ax.get_yaxis_transform(), ha="left", va="center",
                        fontsize=6.5, color=MUTED, linespacing=1.15)
            else:
                ax.text(0.98, row, text.replace(", ", "\n", 1), transform=ax.get_yaxis_transform(), ha="right", va="center",
                        fontsize=6.5, color=MUTED, linespacing=1.15)
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


def panel_inputs(ax: plt.Axes, chains: list[tuple[np.ndarray, np.ndarray]]) -> None:
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
    for k, (xyz, mid) in enumerate(chains):
        inset = ax.inset_axes((6.3 + k * 1.3, 1.2, 1.25, 3.0), transform=ax.transData)
        cols = plt.get_cmap("viridis")((mid - mid.min()) / max(np.ptp(mid), 1))
        _tube(inset, xyz[::3], cols[::3], 1.4)
    ax.text(8.15, 0.55, "ensemble of structures, coloured by genomic position", ha="center", fontsize=8, color=INK)
    ax.text(0.3, 5.5, "cudaMMC: CTCF loops", fontsize=8, color=MUTED)
    ax.text(0.3, 5.0, "gnome3d: CTCF loops, RNAPII loops, Hi-C contacts, structural variants", fontsize=8,
            color=INK)


def _tube(ax: plt.Axes, xyz: np.ndarray, colours: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw one chain with the conference tube renderer, viewed along its smallest principal
    axis. Returns the view rotation and centre so callers can place marks on the same view."""
    centre = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - centre)
    rot = vt.copy()
    draw(ax, [tube_pieces(xyz, colours, rot, centre, width)], background="white")
    ax.axis("off")
    return rot, centre


def _ep_gap(xyz: np.ndarray, mid: np.ndarray) -> float:
    """The 3D distance between the beads nearest the promoter and the enhancer of the locus."""
    a = xyz[int(np.argmin(np.abs(mid - LOCUS["tss"])))]
    b = xyz[int(np.argmin(np.abs(mid - sum(LOCUS["enhancer"]) / 2)))]
    return float(np.linalg.norm(a - b))


def _render(ax: plt.Axes, xyz: np.ndarray, mid: np.ndarray, colour: str, title: str,
            ensemble: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """One conformation of chr1:11.99-12.52 Mb, every third bead, as a tube. The promoter and
    the enhancer are drawn dark and boxed by the bead stretch covering each element, the
    website viewer's rule. The caption gives their 3D distance in this conformation and its
    mean over the conformations of `ensemble`."""
    keep = (mid > 11_990_000) & (mid < 12_520_000)
    full_xyz, full_mid = xyz[keep], mid[keep]
    xyz, mid = full_xyz[::3], full_mid[::3]
    base = np.array(matplotlib.colors.to_rgba(colour))
    light = base.copy()
    light[:3] = 1 - 0.55 * (1 - base[:3])
    cols = np.tile(light, (len(xyz), 1))
    elements = (
        ((LOCUS["tss"] - 1000, LOCUS["tss"] + 1000), "promoter", "#111111"),
        (LOCUS["enhancer"], "enhancer", "#7a3fb0"),
    )
    for (s, e), _, c in elements:
        on = np.abs(mid - (s + e) / 2) <= 4000
        cols[on] = matplotlib.colors.to_rgba(c)
    rot, centre = _tube(ax, xyz, cols, 3.0)
    centres = []
    for (s, e), name, c in elements:
        k = int(np.argmin(np.abs(full_mid - (s + e) / 2)))
        centres.append(full_xyz[k])
        pts = (full_xyz[max(0, k - 4) : k + 5] - centre) @ rot.T
        lo, hi = pts[:, :2].min(0) - 0.8, pts[:, :2].max(0) + 0.8
        ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, edgecolor=c, lw=1.3, zorder=5))
        ax.text(lo[0], hi[1] + 0.3, name, fontsize=7, color=c, va="bottom", zorder=6)
    gap = float(np.linalg.norm(centres[0] - centres[1]))
    mean = float(np.mean([_ep_gap(x, m) for x, m in ensemble]))
    ax.text(0.5, -0.02,
            f"{textwrap.fill(title, 24)}\nenhancer to promoter\n{gap:.1f} bead units here,\n"
            f"{mean:.1f} on average over\n{len(ensemble)} structures",
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


def _microc_measure(ag: pd.DataFrame) -> str:
    """The Micro-C measure the readouts hold, the loops when they are there."""
    return "microc_loops" if (ag.measure == "microc_loops").any() else "microc"


def _agreement_title(ag: pd.DataFrame, who: str, genome: bool = False) -> str:
    where = ", chromosomes 1 to 22" if genome else ""
    if _microc_measure(ag) == "microc_loops":
        return f"Correlation of simulated contact maps with Hi-C, and the share of Micro-C loops the models place, {who}{where}"
    return f"Correlation of simulated contact maps with Hi-C and Micro-C, {who}{where}"


def _agreement_row(sub: SubFigure, ag: pd.DataFrame, a: str, b: str, per_chrom: pd.DataFrame | None = None) -> None:
    axes = sub.subplots(1, 2, sharey=True)
    for ax, measure in zip(axes, ("hic", _microc_measure(ag)), strict=True):
        vals = ag[ag.measure == measure].value
        dumbbells(ax, ag, a, b, measure, (min(0.0, vals.min() - 0.03), vals.max() + 0.05), per_chrom)
    sub.legend(handles=_handles((a, b)), loc="outside lower center", ncol=2, fontsize=7)


def figure_one(inputs: Path, readouts: Path, out: Path) -> Figure:
    """gnome3d against cudaMMC on structure, and the enhancer to promoter distances only RNAPII
    input reaches."""
    ag = agreement(readouts)
    maps = np.load(inputs / "maps_GM12878.npz")
    ep = pd.concat(pd.read_csv(inputs / f"ep_{line}.tsv", sep="\t") for line in LINES)

    fig = plt.figure(figsize=(7.2, 10.8), layout="constrained")
    rows = fig.subfigures(4, 1, height_ratios=[0.8, 1.55, 1.05, 1.9], hspace=0.04)

    _row_title(rows[0], "a", "Input data read by cudaMMC and by gnome3d")
    panel_inputs(rows[0].subplots(), [(maps[f"ensemble_R_{j}_xyz"], maps[f"ensemble_R_{j}_mid"]) for j in range(3)])

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

    _row_title(rows[2], "c", _agreement_title(ag, "cudaMMC and gnome3d with CTCF loops", len(by_chromosome(readouts)) > 0))
    _agreement_row(rows[2], ag, "E", "C", by_chromosome(readouts))

    where = "chromosomes 1 to 22" if "chrom" in ep.columns and ep.chrom.nunique() > 1 else "chr1"
    over = " c: Wilcoxon signed rank test of the two model sets over the chromosomes." if len(by_chromosome(readouts)) else ""
    _row_title(
        rows[3], "d",
        f"3D distance between enhancers and promoters on {where}. Pairs joined by an RNAPII loop and by no CTCF "
        "loop, in models with and without the RNAPII loops, and reference pairs at the same genomic separation "
        "joined by no loop",
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
    notes = {}
    for line in LINES:
        t = ep_tests(ep, line)
        # The marks belong to the last median listed, the models with RNAPII loops.
        notes[LINE_LABEL[line]] = [f"against CTCF loops {stars(t['models'])}, reference {stars(t['reference'])}"]
    joy(
        ax, groups,
        "enhancer to promoter distance in the models, bead units, mean of ten structures.\nA smaller distance is closer",
        clip=(0, 14),
        medians={"control": "reference", "C": "CTCF loops", "R": "CTCF and RNAPII loops"},
        notes=notes,
    )
    rows[3].legend(
        handles=[
            plt.Line2D([], [], color=CONTROL, lw=1.5,
                       label="reference: enhancer and promoter pairs joined by no loop, same genomic separation"),
            plt.Line2D([], [], marker="o", ls="", color=ARM_COLOUR["C"], ms=6,
                       label="pairs joined by an RNAPII loop, in the models built from CTCF loops"),
            plt.Line2D([], [], marker="o", ls="", color=ARM_COLOUR["R"], ms=6,
                       label="the same pairs, in the models built from CTCF and RNAPII loops"),
        ],
        loc="outside lower center", ncol=1, fontsize=7,
    )
    fig.supxlabel(
        textwrap.fill(
            f"{LEGEND}. d: Wilcoxon signed rank test over the pairs of both model sets, "
            f"Mann-Whitney U test against the reference pairs. The marks stand beside the models with RNAPII loops.{over}", 150),
        fontsize=6.5, color=MUTED,
    )
    _save(fig, out / "rnapii_figure1")
    return fig


def figure_two(inputs: Path, readouts: Path, out: Path, trio_dir: Path | None = None) -> Figure:
    """The link to expression, agreement with contact maps, the trio individuals and one locus,
    gnome3d with and without RNAPII loops."""
    ag = agreement(readouts)
    draws = link_draws(readouts, ("C", "R"))
    trio = trio_paired(trio_dir or readouts / "trio")
    maps = np.load(inputs / "maps_GM12878.npz")

    fig = plt.figure(figsize=(7.2, 10.0), layout="constrained")
    rows = fig.subfigures(3, 1, height_ratios=[1.25, 1.05, 1.5], hspace=0.04)

    _row_title(
        rows[0], "e",
        "Correlation between the change in enhancer to promoter distance and the change in gene expression "
        "across pairs of cell lines, bootstrap distribution over genes" + (", chromosomes 1 to 22" if len(by_chromosome(readouts)) else ""),
    )
    ax = rows[0].subplots()
    link = link_difference(readouts, "C", "R", n=10000)
    joy(ax, _joy_groups(draws, ("C", "R")),
        "minus Spearman correlation over genes, 2,000 block bootstrap draws, larger is a stronger link",
        notes={PAIR_LABEL[k]: [f"difference {d:+.3f}, {mark(pv)}"] for k, (d, pv, _) in link.items()})
    rows[0].legend(handles=_handles(("C", "R")), loc="outside lower center", ncol=2, fontsize=7)

    _row_title(rows[1], "f", _agreement_title(ag, "gnome3d with and without RNAPII loops", len(by_chromosome(readouts)) > 0))
    _agreement_row(rows[1], ag, "C", "R", by_chromosome(readouts))

    bottom = rows[2].subfigures(1, 2, width_ratios=[0.85, 1.15], wspace=0.03)
    scope = ", chromosomes 1 to 22" if ((trio_dir or readouts / "trio") / "trio_genome.json").is_file() else ""
    _row_title(
        bottom[0], "g",
        f"Nine trio individuals, change from adding RNAPII loops, each against the individual's own Hi-C{scope}", 40,
    )
    ax = bottom[0].subplots()
    rows_g = (
        ("loop_pileup", "contact enrichment\nat RNAPII loops"),
        ("hic", "correlation with\nthe whole Hi-C map"),
    )
    for row, (measure, label) in enumerate(rows_g[::-1]):
        w = trio[trio.measure == measure].pivot(index="person", columns="arm", values="value")
        change = 100 * (w.R - w.C) / w.C
        jitter = np.linspace(-0.12, 0.12, len(change))
        ax.scatter(change, row + jitter, color=ARM_COLOUR["R"], s=16, zorder=3)
        up = int((change > 0).sum())
        pv, _ = paired(w.C.to_numpy(), w.R.to_numpy())
        ax.text(1.0, row + 0.32, f"higher in {up} of {len(change)}, {mark(pv)}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7, color=INK)
        ax.text(-0.02, row, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(None, ax.get_xlim()[1] + 18)
    ax.set_xlabel("change with RNAPII loops, percent of the\nvalue with CTCF loops alone")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    _row_title(bottom[1], "h", f"The {LOCUS['gene']} locus in GM12878, medoid of ten structures per model set", 52)
    for ax, arm in zip(bottom[1].subplots(1, 2), ("C", "R"), strict=True):
        n = sum(1 for k in maps.files if k.startswith(f"ensemble_{arm}_") and k.endswith("_xyz"))
        ens = [(maps[f"ensemble_{arm}_{j}_xyz"], maps[f"ensemble_{arm}_{j}_mid"]) for j in range(n)]
        _render(ax, maps[f"locus_{arm}_xyz"], maps[f"locus_{arm}_mid"], ARM_COLOUR[arm], ARM_NAME[arm], ens)
    gaps = {
        arm: np.array([_ep_gap(maps[f"ensemble_{arm}_{j}_xyz"], maps[f"ensemble_{arm}_{j}_mid"])
                       for j in range(sum(1 for k in maps.files if k.startswith(f"ensemble_{arm}_") and k.endswith("_xyz")))])
        for arm in ("C", "R")
    }
    bottom[1].supxlabel(
        textwrap.fill(
            "enhancer to promoter distance over the ten structures, with against without RNAPII loops: "
            f"{mark(unpaired(gaps['C'], gaps['R']))}", 70),
        fontsize=7, color=INK,
    )
    fig.supxlabel(
        textwrap.fill(
            f"{LEGEND}. e: block bootstrap of the difference between the two model sets on the genes of both."
            + (" f: Wilcoxon signed rank test of the two model sets over the chromosomes. " if len(by_chromosome(readouts)) else " ")
            + "g: Wilcoxon signed rank test over the nine individuals, exact. h: Mann-Whitney U test.", 150),
        fontsize=6.5, color=MUTED,
    )
    _save(fig, out / "rnapii_figure2")
    return fig


def _forest(ax: plt.Axes, values: dict[str, tuple[float, float, float] | None], arms: tuple[str, ...]) -> None:
    """One row per model set: the value as a dot, with its interval when one is given, and the
    value written beside it. A panel with no values at all says so once."""
    if all(values.get(x) is None for x in arms):
        ax.text(0.5, 0.5, "no Micro-C map for this line", transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=7, style="italic")
    for row, a in enumerate(arms[::-1]):
        v = values.get(a)
        if v is None:
            if all(values.get(x) is None for x in arms):
                continue
            ax.text(0.5, row, "no data", transform=ax.get_yaxis_transform(), ha="center", va="center",
                    color=MUTED, fontsize=7, style="italic")
            continue
        mid, lo, hi = v
        if np.isfinite(lo):
            ax.plot([lo, hi], [row, row], color=ARM_COLOUR[a], lw=2, alpha=0.5, solid_capstyle="round")
        ax.plot(mid, row, "o", color=ARM_COLOUR[a], ms=7, zorder=3)
        ax.text(mid, row + 0.3, f"{mid:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_ylim(-0.6, len(arms) - 0.3)
    ax.set_yticks(range(len(arms)))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))


def figure_supplement(inputs: Path, readouts: Path, out: Path) -> Figure:
    """All four model sets on agreement with Hi-C, with Micro-C and on the link to expression,
    one small panel per cell line or line pair."""
    ag = agreement(readouts)
    arms = ("E", "C", "R", "M")
    draws = link_draws(readouts, arms, n=1000)
    classes = microc_classes(readouts)
    by_class = not classes.empty
    fig = plt.figure(figsize=(7.2, 11.0 if by_class else 8.4), layout="constrained")
    rows = fig.subfigures(4 if by_class else 3, 1, hspace=0.05)
    microc = _microc_measure(ag)
    specs = (
        ("a", "Stratum adjusted correlation of simulated contact maps with Hi-C at 25 kb, per cell line, "
              "higher is better", "hic"),
        ("b", "Share of the loops called on Micro-C that the models place, above the share for the same loops "
              "moved away, per cell line, higher is better" if microc == "microc_loops" else
              "Correlation of simulated contact maps with Micro-C at 5 kb, per cell line, higher is better",
         microc),
    )
    for sub, (letter, title, measure) in zip(rows[:2], specs, strict=True):
        _row_title(sub, letter, title)
        axes = sub.subplots(1, 3, sharey=True)
        m = ag[ag.measure == measure]
        allv = m.value
        for ax, line in zip(axes, LINES, strict=True):
            s_ = m[m.line == line].set_index("arm").value
            _forest(ax, {a: (float(s_[a]), np.nan, np.nan) if a in s_ else None for a in arms}, arms)
            ax.set_title(LINE_LABEL[line].replace("\n", " "), fontsize=8, color=INK)
            ax.set_xlim(allv.min() - 0.03, allv.max() + 0.03)
        axes[0].set_yticklabels([textwrap.fill(ARM_NAME[a], 26) for a in arms[::-1]], fontsize=7)
    _row_title(
        rows[2], "c",
        "Correlation between the change in enhancer to promoter distance and the change in gene expression, "
        "sign reversed so larger is a stronger link, median and 95 percent bootstrap interval",
    )
    axes = rows[2].subplots(1, 3, sharey=True, sharex=True)
    for ax, pair in zip(axes, PAIR_LABEL, strict=True):
        vals = {}
        for a in arms:
            v = draws[(draws.pair == pair) & (draws.arm == a)].value.to_numpy()
            vals[a] = (float(np.median(v)), *(float(x) for x in np.percentile(v, [2.5, 97.5])))
        _forest(ax, vals, arms)
        ax.set_title(PAIR_LABEL[pair], fontsize=8, color=INK)
    axes[0].set_yticklabels([textwrap.fill(ARM_NAME[a], 26) for a in arms[::-1]], fontsize=7)
    if by_class:
        _row_title(
            rows[3], "d",
            "Share of Micro-C loops placed, above moved loops, by the ChIA-PET loop at the Micro-C loop, "
            "gnome3d with and without RNAPII loops",
        )
        lines = [x for x in LINES if (classes.line == x).any()]
        axes = rows[3].subplots(1, len(lines), sharey=True, sharex=True, squeeze=False)[0]
        for ax, line in zip(axes, lines, strict=True):
            c = classes[classes.line == line].set_index(["loop_class", "arm"])
            for row, cls in enumerate(list(CLASS_LABEL)[::-1]):
                va, vb = float(c.value[(cls, "C")]), float(c.value[(cls, "R")])
                ax.plot([va, vb], [row, row], color="#9a9893", lw=1.6, zorder=2)
                ax.plot(va, row, "o", color=ARM_COLOUR["C"], ms=7, zorder=3)
                ax.plot(vb, row, "o", color=ARM_COLOUR["R"], ms=7, zorder=3)
                ax.text(vb, row + 0.28, f"{vb:.2f}", ha="center", fontsize=7, color=INK)
                ax.text(va, row - 0.42, f"{va:.2f}", ha="center", fontsize=7, color=MUTED)
                ax.text(1.0, row, f"{int(c.loops[(cls, 'C')])} loops", transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=7, color=MUTED)
            ax.set_title(LINE_LABEL[line].replace("\n", " "), fontsize=8, color=INK)
            ax.set_ylim(-0.6, len(CLASS_LABEL) - 0.3)
            ax.set_xlim(0, 1.25)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_yticks(range(len(CLASS_LABEL)))
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
        axes[0].set_yticklabels([CLASS_LABEL[k] for k in list(CLASS_LABEL)[::-1]], fontsize=7)
        rows[3].legend(handles=_handles(("C", "R")), loc="outside lower center", ncol=2, fontsize=7)
    _save(fig, out / "rnapii_supplement")
    return fig


def _save(fig: Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))


def main() -> None:
    inputs, readouts, out = (Path(x) for x in sys.argv[1:4])
    trio_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    figure_one(inputs, readouts, out)
    figure_two(inputs, readouts, out, trio_dir)
    figure_supplement(inputs, readouts, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
