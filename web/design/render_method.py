r"""Renders the figures of section 2 of the Method page, blocks and target distances.

    python web/design/render_method.py tables data playground/derived/ps_curve.csv web/design/method_data
    python web/design/render_method.py figures web/design/method_data web/design/images

`tables` reads the GM12878 inputs of production and the contact probability table of three cell
lines from `playground/ps_curve.py`. It writes the small tables the figures plot. `figures`
draws `method_blocks.png`, `method_exponent.png`, `method_default.png`, `method_loops.png` and
`method_hic.png` of section 2 and `method_anchor_terms.png` of section 3 from those tables alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

INK, MUTED, LINE = "#151514", "#8a8984", "#e4e3df"
BLUE, MID, PALE = "#2a78d6", "#5a9be0", "#9fc3ee"
WASH, GAP = "#eaf2fc", "#f1f0ec"
CELL_COLOURS = {"GM12878": "#2a78d6", "H1ESC": "#eb6834", "HFFC6": "#1baf7a"}
CELL_NAMES = {"GM12878": "GM12878", "H1ESC": "H1-hESC", "HFFC6": "HFFc6"}

CELL = "GM12878"
CHROM = "chr1"
REGION = (236_000_000, 242_100_000)
WIDTH = 7.6


# Tables


def tables(data_root: str, ps_curve: str, out: Path) -> None:
    from gnome3d.data import ContactData
    from gnome3d.hierarchy import Level, build_cluster_tree
    from gnome3d.io import load_singletons
    from gnome3d.polymer import fit_arc_strength, fit_contact_exponent
    from gnome3d.production import settings_for_cell

    out.mkdir(parents=True, exist_ok=True)
    fits: dict[str, object] = {"cell": CELL, "chrom": CHROM, "region": list(REGION)}

    settings = settings_for_cell(CELL, data_root)
    data = ContactData.from_files(settings, [CHROM], None)
    loops = pd.DataFrame(
        [(a.genomic_start, a.genomic_end, a.score) for a in data.arcs[CHROM] if a.factor == 0],
        columns=["start", "end", "pets"],
    )
    arc_fit = fit_arc_strength([a for a in data.arcs[CHROM] if a.factor == 0])
    assert arc_fit.ok, arc_fit.reason
    fits["loops"] = {"log_a": arc_fit.log_a, "slope": arc_fit.slope, "n": len(loops)}
    loops.assign(span=loops.end - loops.start)[["span", "pets"]].to_csv(out / "loop_counts.csv", index=False)

    clusters, _, _ = build_cluster_tree(data.anchors, data.arcs, data.breakpoints, [CHROM])
    blocks = pd.DataFrame(
        [(c.start, c.end, len(c.children)) for c in clusters if c.level == Level.INTERACTION_BLOCK],
        columns=["start", "end", "anchors"],
    )
    inside = (blocks.start >= REGION[0]) & (blocks.end <= REGION[1])
    blocks[inside].to_csv(out / "blocks.csv", index=False)
    region_loops = loops[(loops.start >= REGION[0]) & (loops.end <= REGION[1])]
    region_loops.to_csv(out / "block_loops.csv", index=False)
    anchors = pd.DataFrame(
        [(c.start, c.end) for c in clusters if c.level == Level.ANCHOR and c.start >= REGION[0] and c.end <= REGION[1]],
        columns=["start", "end"],
    )
    anchors.to_csv(out / "block_anchors.csv", index=False)

    contacts = load_singletons(str(Path(data_root) / CELL / f"{CELL}_hic_25kb_singletons.bedpe"), {CHROM}, None)
    fit = fit_contact_exponent(contacts)
    assert fit.ok, fit.reason
    a = np.array([c[1] for c in contacts], dtype=np.float64)
    b = np.array([c[3] for c in contacts], dtype=np.float64)
    w = np.array([c[4] for c in contacts], dtype=np.float64)
    step = 25_000
    k = np.rint(np.abs(b - a) / step).astype(np.int64)
    counts = np.bincount(k, weights=w)
    n_bins = int(np.rint((max(a.max(), b.max()) - min(a.min(), b.min())) / step)) + 1
    ks = np.arange(counts.size)
    offered = np.maximum(n_bins - ks, 0)
    keep = (ks >= 1) & (ks * step <= 20_000_000) & (counts > 20) & (offered > 0)
    curve = pd.DataFrame({"s": ks[keep] * step, "p": counts[keep] / offered[keep]})
    band = (curve.s >= fit.lo) & (curve.s <= fit.hi)
    slope, intercept = np.polyfit(np.log(curve.s[band]), np.log(curve.p[band]), 1)
    assert abs(slope - fit.slope) < 1e-9, (slope, fit.slope)
    curve.to_csv(out / "contact_curve.csv", index=False)
    fits["contacts"] = {"nu": fit.nu, "slope": fit.slope, "intercept": float(intercept), "lo": fit.lo, "hi": fit.hi}

    three = pd.read_csv(ps_curve)
    three.to_csv(out / "three_cells.csv", index=False)
    fits["anchor_pairs"] = anchor_tables(data_root, out)
    (out / "fits.json").write_text(json.dumps(fits, indent=1) + "\n")
    print(json.dumps(fits, indent=1))
    print(blocks[inside].to_string())


def anchor_tables(data_root: str, out: Path) -> dict[str, int]:
    """Writes the kind of every anchor pair of the region and the windows of the window term, as
    the joint anchor solve of a job on the region builds them."""
    from gnome3d.data import ContactData
    from gnome3d.hierarchy import Level
    from gnome3d.io import parse_region
    from gnome3d.mc.numba.arcs_solver import gyration_windows
    from gnome3d.pipeline.coarse import build as cb
    from gnome3d.production import settings_for_cell

    s = settings_for_cell(CELL, data_root)
    region = parse_region(f"{CHROM}:{REGION[0]}-{REGION[1]}")
    data = ContactData.from_files(s, [CHROM], region)
    state = cb.build_state(s, data, [CHROM], region)
    cl = state.clusters
    ibs = [c for c in cl if c.level == Level.INTERACTION_BLOCK and len(c.children) > 1]
    active = [a for ib in ibs for a in ib.children]
    block = np.concatenate([np.full(len(ib.children), k) for k, ib in enumerate(ibs)])
    heat, _ = cb.build_contact_heatmaps(state, active, CHROM, with_subanchor=False)
    mat = cb.calc_anchor_expected_distances(state, active, CHROM, heat, None)
    mids = np.array([cl[a].genomic_pos for a in active], dtype=np.int64)
    sep = np.abs(mids[:, None] - mids[None, :])
    rng = int(s.background_range_bp)
    # 1 a loop, 2 the background under the range, 3 a Hi-C target beyond it, 0 no target.
    kind = np.zeros(mat.shape, dtype=np.int64)
    kind[mat > 0] = 1
    kind[(mat <= -1) & (sep < rng)] = 2
    kind[(mat <= -1) & (sep >= rng)] = 3
    np.fill_diagonal(kind, -1)
    assert ((mat == 0) == (kind <= 0)).all()
    pd.DataFrame(kind).to_csv(out / "anchor_pairs.csv", index=False, header=False)
    pd.DataFrame({"mid": mids, "block": block}).to_csv(out / "anchor_mids.csv", index=False)

    law = s.polymer_law()
    member, _, target = gyration_windows(mids, law, rng)
    rows = []
    lo, hi = int(mids.min()), int(mids.max())
    w = rng
    while w <= hi - lo:
        for tiling, off in enumerate((0, w // 2)):
            lab = (mids - lo + off) // w
            for v in np.unique(lab):
                n = int((lab == v).sum())
                rows.append((w, tiling, lo - off + int(v) * w, lo - off + (int(v) + 1) * w, n))
        w *= 2
    win = pd.DataFrame(rows, columns=["size", "tiling", "start", "end", "anchors"])
    assert int((win.anchors >= 4).sum()) == member.shape[0] == len(target)
    win.to_csv(out / "windows.csv", index=False)
    iu = np.triu_indices(len(mids), 1)
    k = kind[iu]
    return {
        "anchors": len(mids), "pairs": int(k.size), "loop": int((k == 1).sum()), "background": int((k == 2).sum()),
        "hic": int((k == 3).sum()), "none": int((k == 0).sum()), "windows": int(member.shape[0]), "range_bp": rng,
    }


# Figures


def style(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    if grid:
        ax.grid(True, which="major", color=LINE, lw=0.6)
        ax.set_axisbelow(True)


def letter(ax: plt.Axes, text: str) -> None:
    ax.text(-0.02, 1.06, text, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, ha="right", va="bottom")


def separation_axis(ax: plt.Axes, ticks: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e6:g} Mb" if t >= 1e6 else f"{t / 1e3:g} kb" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("genomic separation", fontsize=9.5)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(path)


def blocks(tab: Path, out: Path) -> None:
    """Draws the loops of one region as arcs, the count of loops over each position and the blocks."""
    loops = pd.read_csv(tab / "block_loops.csv")
    bl = pd.read_csv(tab / "blocks.csv")
    anchors = pd.read_csv(tab / "block_anchors.csv")
    lo, hi = REGION
    mb = 1e6

    edges = np.unique(np.concatenate([[lo, hi], loops.start, loops.end]))
    cover = np.array([((loops.start <= a) & (loops.end >= b)).sum() for a, b in zip(edges[:-1], edges[1:], strict=True)])
    # The blocks of the engine are the stretches that the loops cover.
    covered = [(a, b) for a, b, c in zip(edges[:-1], edges[1:], cover, strict=True) if c > 0]
    runs: list[list[float]] = []
    for a, b in covered:
        if runs and runs[-1][1] == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    assert len(runs) == len(bl), (len(runs), len(bl))

    fig, (ax_l, ax_c, ax_b) = plt.subplots(
        3, 1, figsize=(WIDTH, 4.3), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.3, 0.75], "hspace": 0.16}
    )
    gaps = [(lo, bl.start.iloc[0])] + list(zip(bl.end.iloc[:-1], bl.start.iloc[1:], strict=True)) + [(bl.end.iloc[-1], hi)]
    for ax in (ax_l, ax_c, ax_b):
        for a, b in gaps:
            ax.axvspan(a / mb, b / mb, color=GAP, lw=0)

    top = (loops.end - loops.start).max() / mb / 2
    for s, e, n in loops.itertuples(index=False):
        w = (e - s) / mb
        ax_l.add_patch(Arc(((s + e) / 2 / mb, 0), w, w, theta1=0, theta2=180, color=BLUE, lw=0.5 + 0.35 * np.log2(n), alpha=0.55))
    ax_l.set_ylim(0, top * 1.08)
    ax_l.set_yticks([])
    ax_l.spines[["top", "right", "left"]].set_visible(False)
    ax_l.set_ylabel("loops", fontsize=9.5)

    ax_c.fill_between(np.repeat(edges, 2)[1:-1] / mb, np.repeat(cover, 2), step=None, color=PALE, lw=0)
    ax_c.plot(np.repeat(edges, 2)[1:-1] / mb, np.repeat(cover, 2), color=BLUE, lw=1)
    ax_c.set_ylim(0, cover.max() * 1.1)
    ax_c.set_ylabel("loops over\neach position", fontsize=9.5)
    style(ax_c, grid=False)

    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([])
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    for i, (s, e, n) in enumerate(bl.itertuples(index=False), start=1):
        ax_b.add_patch(
            FancyBboxPatch((s / mb, 0.42), (e - s) / mb, 0.4, boxstyle="round,pad=0,rounding_size=0.03", fc=INK, ec="none", mutation_aspect=8)
        )
        ax_b.text((s + e) / 2 / mb, 0.2, f"block {i}", ha="center", va="center", fontsize=8.5, color=INK)
    ax_b.vlines(anchors[["start", "end"]].mean(axis=1) / mb, 0.42, 0.82, color="white", lw=0.5)
    for a, b in gaps:
        if (b - a) / mb > 0.5:
            ax_l.text((a + b) / 2 / mb, top * 0.5, "no loop passes\nover this stretch", ha="center", va="center", fontsize=8.5, color=MUTED)
            ax_b.text((a + b) / 2 / mb, 0.62, "no beads", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax_b.set_ylabel("blocks", fontsize=9.5)
    ax_b.set_xlim(lo / mb, hi / mb)
    ax_b.set_xlabel("position on chromosome 1, Mb", fontsize=9.5)
    ax_b.tick_params(labelsize=8.5)
    fig.align_ylabels([ax_l, ax_c, ax_b])
    fig.subplots_adjust(left=0.115, right=0.985, top=0.98, bottom=0.115)
    save(fig, out / "method_blocks.png")


def exponent(tab: Path, out: Path) -> None:
    """Draws the contact probability of one chromosome with the fit of the exponent and the background it gives."""
    curve = pd.read_csv(tab / "contact_curve.csv")
    fit = json.loads((tab / "fits.json").read_text())["contacts"]
    nu, lo, hi = fit["nu"], fit["lo"], fit["hi"]
    band = (curve.s >= lo) & (curve.s <= hi)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    ax.axvspan(lo, hi, color=WASH, lw=0)
    ax.plot(curve.s[~band], curve.p[~band], "o", ms=2.6, color="#bcbbb6", mew=0)
    ax.plot(curve.s[band], curve.p[band], "o", ms=3.2, color=INK, mew=0)
    xs = np.array([lo, hi], dtype=float)
    ax.plot(xs, np.exp(fit["intercept"]) * xs ** fit["slope"], color=BLUE, lw=2)
    ax.set_yscale("log")
    separation_axis(ax, [25e3, 100e3, 1e6, 10e6])
    ax.set_ylabel("contacts per pair of bins", fontsize=9.5)
    ax.text(np.sqrt(lo * hi), 0.04, "range of the fit", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=BLUE)
    ax.text(
        0.97, 0.95, f"slope of the line = $-3\\nu$ = \u2212{-fit['slope']:.2f}\n$\\nu$ = {nu:.2f}",
        transform=ax.transAxes, fontsize=9.5, color=INK, ha="right", va="top", linespacing=1.5,
    )
    style(ax)
    letter(ax, "a")

    sep = np.logspace(3, 7, 200)
    bx.plot(sep, np.maximum(1.0, (sep / 1000.0) ** nu), color=INK, lw=2)
    for s in (1e4, 1e5, 1e6):
        d = (s / 1000.0) ** nu
        bx.plot([s], [d], "o", ms=5, color=BLUE, mew=0)
        bx.annotate(f"{d:.1f} beads", (s, d), textcoords="offset points", xytext=(6, -11), fontsize=8.5, color=BLUE)
    bx.plot([1e3], [1.0], "o", ms=5, color=BLUE, mew=0)
    bx.annotate("1 bead at 1 kb", (1e3, 1.0), textcoords="offset points", xytext=(7, -3), fontsize=8.5, color=BLUE)
    bx.set_yscale("log")
    bx.set_yticks([1, 2, 5, 10, 20])
    bx.set_yticklabels(["1", "2", "5", "10", "20"])
    bx.minorticks_off()
    separation_axis(bx, [1e3, 1e4, 1e5, 1e6, 1e7])
    bx.set_ylabel("background distance $b(s)$, bead units", fontsize=9.5)
    style(bx)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, out / "method_exponent.png")


def default(tab: Path, out: Path) -> None:
    """Draws the contact probability of three cell lines with their exponents and the mean, the default."""
    three = pd.read_csv(tab / "three_cells.csv")
    lo, hi = 2e4, 1e6
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 3.5), gridspec_kw={"width_ratios": [1.45, 1]})
    ax.axvspan(lo, hi, color=WASH, lw=0)
    nus: dict[str, float] = {}
    for i, (cell, d) in enumerate(three.groupby("cell", sort=False)):
        d = d[d.s <= 1e7]
        m = (d.s >= lo) & (d.s <= hi)
        slope, icpt = np.polyfit(np.log10(d.s[m]), np.log10(d.p[m]), 1)
        nus[cell] = -slope / 3
        shift = 10.0 ** (-0.8 * i)
        col = CELL_COLOURS[cell]
        ax.plot(d.s, d.p * shift, "o", ms=3.2, color=col, mew=0, alpha=0.55)
        xs = np.array([lo, hi])
        ax.plot(xs, 10**icpt * xs**slope * shift, color=col, lw=2)
        ax.text(1.25e7, d.p.iloc[-1] * shift, CELL_NAMES[cell], color=col, fontsize=9, va="center")
    ax.set_yscale("log")
    ax.set_yticks([])
    ax.minorticks_off()
    separation_axis(ax, [1e4, 1e5, 1e6, 1e7])
    ax.set_xlim(7e3, 9e7)
    ax.set_ylabel("contact probability, curves shifted apart", fontsize=9.5)
    ax.text(np.sqrt(lo * hi), 0.06, "range of the fit", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=BLUE)
    style(ax)
    letter(ax, "a")

    mean = float(np.mean(list(nus.values())))
    for i, (cell, v) in enumerate(nus.items()):
        bx.plot([v], [i], "o", ms=9, color=CELL_COLOURS[cell], mew=0)
        side = 1 if v > mean else -1
        bx.text(v + side * 0.005, i, f"{v:.3f}", ha="left" if side > 0 else "right", va="center", fontsize=9, color=CELL_COLOURS[cell])
    bx.axvline(mean, color=INK, lw=1.6)
    bx.text(mean, -0.86, f"mean {mean:.3f},\nthe default", ha="center", va="bottom", fontsize=9, color=INK)
    bx.axvline(1 / 3, color=MUTED, lw=1, ls=(0, (4, 3)))
    bx.text(1 / 3, -0.86, "1/3", ha="center", va="bottom", fontsize=9, color=MUTED)
    bx.set_yticks(range(len(nus)))
    bx.set_yticklabels([CELL_NAMES[c] for c in nus])
    bx.set_ylim(2.75, -1.0)
    bx.set_xlim(0.24, 0.35)
    bx.set_xticks([0.25, 0.30, 0.35])
    bx.set_xlabel("exponent $\\nu$", fontsize=9.5)
    bx.tick_params(axis="y", length=0)
    style(bx)
    bx.grid(False, axis="y")
    letter(bx, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, out / "method_default.png")
    print({k: round(v, 3) for k, v in nus.items()}, round(mean, 4))


def loop_targets(tab: Path, out: Path) -> None:
    """Draws the PET count against span with the typical count and the target of a loop of each strength."""
    loops = pd.read_csv(tab / "loop_counts.csv")
    fits = json.loads((tab / "fits.json").read_text())
    f, nu = fits["loops"], fits["contacts"]["nu"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    rng = np.random.default_rng(0)
    jitter = np.exp(rng.uniform(-0.04, 0.04, len(loops)))
    ax.plot(loops.span, loops.pets * jitter, "o", ms=1.8, color=PALE, mew=0, alpha=0.5, rasterized=True)
    xs = np.array([loops.span.min(), loops.span.max()], dtype=float)
    typical = lambda s: np.exp(f["log_a"] + f["slope"] * np.log(s))  # noqa: E731
    ax.plot(xs, typical(xs), color=INK, lw=2, label="typical count $\\bar{n}(s)$")
    ax.plot(xs, 4 * typical(xs), color=BLUE, lw=1.2, ls=(0, (4, 3)))
    box = {"fc": "white", "ec": "none", "pad": 1.2, "alpha": 0.85}
    ax.text(xs[1] * 1.15, 4 * typical(xs[1]), "$q$ = 4", color=BLUE, fontsize=8.5, ha="left", va="center", bbox=box)
    ax.text(xs[1] * 1.15, typical(xs[1]), "$q$ = 1", color=INK, fontsize=8.5, ha="left", va="center", bbox=box)
    ax.set_xlim(xs[0] * 0.8, xs[1] * 2.6)
    ax.set_yscale("log")
    ax.set_yticks([3, 10, 100, 1000, 10000])
    ax.set_yticklabels(["3", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    separation_axis(ax, [1e4, 1e5, 1e6])
    ax.set_xlabel("span of the loop", fontsize=9.5)
    ax.set_ylabel("PET count $n$", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax)
    letter(ax, "a")

    sep = np.logspace(3, 7, 200)
    bg = np.maximum(1.0, (sep / 1000.0) ** nu)
    bx.plot(sep, bg, color=INK, lw=2, label="no loop, $b(s)$")
    for q, col in [(0.5, PALE), (1.0, MID), (4.0, BLUE)]:
        bx.plot(sep, 1.0 + (bg - 1.0) / (1.0 + q), color=col, lw=2, label=f"loop, $q$ = {q:g}")
    bx.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    bx.text(1e7, 1.04, "one bead", color=MUTED, fontsize=8.5, ha="right", va="bottom")
    bx.set_yscale("log")
    bx.set_yticks([1, 2, 5, 10, 20])
    bx.set_yticklabels(["1", "2", "5", "10", "20"])
    bx.minorticks_off()
    bx.set_ylim(0.9, 28)
    separation_axis(bx, [1e3, 1e4, 1e5, 1e6, 1e7])
    bx.set_ylabel("target distance, bead units", fontsize=9.5)
    bx.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(bx)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, out / "method_loops.png")


def hic_targets(tab: Path, out: Path) -> None:
    """Draws the target of a Hi-C cell against its observed over expected count and against separation."""
    nu = json.loads((tab / "fits.json").read_text())["contacts"]["nu"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    r = np.logspace(np.log10(0.25), np.log10(16), 200)
    ax.plot(r, r ** (-1 / 3), color=INK, lw=2)
    for v, text, off in [(1, "the expected count, 1", (8, 6)), (2, "2 times, 0.79", (8, 4)), (8, "8 times, 0.50", (-6, -15))]:
        ax.plot([v], [v ** (-1 / 3)], "o", ms=5.5, color=BLUE, mew=0)
        ax.annotate(text, (v, v ** (-1 / 3)), textcoords="offset points", xytext=off, fontsize=8.5, color=BLUE, ha="left" if off[0] > 0 else "right")
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16])
    ax.set_xticklabels(["1/4", "1/2", "1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_ylim(0.3, 1.7)
    ax.set_xlabel("observed over expected contacts", fontsize=9.5)
    ax.set_ylabel("target over background distance", fontsize=9.5)
    style(ax)
    letter(ax, "a")

    sep = np.logspace(np.log10(25e3), 7, 200)
    bg = (sep / 1000.0) ** nu
    bx.plot(sep, bg, color=INK, lw=2, label="expected count, $b(s)$")
    for v, col in [(2, PALE), (4, MID), (8, BLUE)]:
        bx.plot(sep, bg * v ** (-1 / 3), color=col, lw=2, label=f"{v} times the expected count")
    bx.set_yscale("log")
    bx.set_yticks([2, 5, 10, 20])
    bx.set_yticklabels(["2", "5", "10", "20"])
    bx.minorticks_off()
    bx.set_ylim(1.2, 40)
    separation_axis(bx, [25e3, 1e5, 1e6, 1e7])
    bx.set_ylabel("target distance, bead units", fontsize=9.5)
    bx.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(bx)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.2)
    save(fig, out / "method_hic.png")


def anchor_terms(tab: Path, out: Path) -> None:
    """Draws the kind of target of every anchor pair of the region and the windows of the window term."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    kind = pd.read_csv(tab / "anchor_pairs.csv", header=None).to_numpy()
    anchors = pd.read_csv(tab / "anchor_mids.csv")
    win = pd.read_csv(tab / "windows.csv")
    n_of = json.loads((tab / "fits.json").read_text())["anchor_pairs"]
    n = len(anchors)
    mb = 1e6

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    shown = np.ma.masked_less(kind, 0)
    ax.imshow(shown, cmap=ListedColormap([GAP, BLUE, INK, "#eb6834"]), vmin=0, vmax=3, interpolation="nearest")
    cuts = np.flatnonzero(np.diff(anchors.block)) + 0.5
    for c in cuts:
        ax.axhline(c, color="white", lw=1.6)
        ax.axvline(c, color="white", lw=1.6)
    centres = [float(np.flatnonzero(anchors.block == b).mean()) for b in sorted(anchors.block.unique())]
    ax.set_xticks(centres)
    ax.set_xticklabels([f"block {b + 1}" for b in range(len(centres))], fontsize=8.5)
    ax.set_yticks(centres)
    ax.set_yticklabels([f"block {b + 1}" for b in range(len(centres))], fontsize=8.5, rotation=90, va="center")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    handles = [
        Patch(fc=BLUE, label=f"loop, {n_of['loop']} pairs"),
        Patch(fc=INK, label=f"background, {n_of['background']} pairs"),
        Patch(fc=GAP, ec="#d9d8d3", label=f"no target, {n_of['none']:,} pairs"),
    ]
    if n_of["hic"]:
        handles.insert(2, Patch(fc="#eb6834", label=f"Hi-C target, {n_of['hic']} pairs"))
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, handlelength=1.2)
    ax.set_title(f"{n} anchors, {n_of['pairs']:,} pairs", fontsize=9.5, color=INK, pad=6)
    letter(ax, "a")

    sizes = sorted(win["size"].unique())
    for k, size in enumerate(sizes):
        for tiling in (0, 1):
            y = k + (0.2 if tiling == 0 else -0.2)
            rows = win[(win["size"] == size) & (win.tiling == tiling)]
            for start, end, count in rows[["start", "end", "anchors"]].itertuples(index=False):
                kept = count >= 4
                bx.add_patch(
                    plt.Rectangle(
                        (start / mb, y - 0.15), (end - start) / mb, 0.3,
                        fc=BLUE if kept else "none", ec=BLUE if kept else "#c9c8c3", lw=0.6, alpha=0.75 if kept else 1.0,
                    )
                )
    bx.vlines(anchors.mid / mb, -1.05, -0.65, color=INK, lw=0.6)
    bx.text(REGION[0] / mb - 0.08, -0.85, "anchors", ha="right", va="center", fontsize=8.5)
    bx.set_yticks(range(len(sizes)))
    bx.set_yticklabels([f"{s / 1e6:g} Mb" if s >= 1e6 else f"{s / 1e3:g} kb" for s in sizes], fontsize=8.5)
    bx.set_ylim(-1.15, len(sizes) - 0.4)
    bx.set_xlim(REGION[0] / mb, REGION[1] / mb)
    bx.set_xlabel("position on chromosome 1, Mb", fontsize=9.5)
    bx.set_ylabel("size of the window", fontsize=9.5)
    bx.tick_params(axis="y", length=0)
    bx.tick_params(labelsize=8.5)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.set_title(f"{n_of['windows']} windows with four or more anchors", fontsize=9.5, color=INK, pad=6)
    letter(bx, "b")
    fig.tight_layout(w_pad=2.4, rect=(0, 0.14, 1, 1))
    save(fig, out / "method_anchor_terms.png")


def main() -> None:
    if sys.argv[1] == "tables":
        tables(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        return
    tab, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    blocks(tab, out)
    exponent(tab, out)
    default(tab, out)
    loop_targets(tab, out)
    hic_targets(tab, out)
    anchor_terms(tab, out)


if __name__ == "__main__":
    main()
