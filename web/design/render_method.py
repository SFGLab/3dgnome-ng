r"""Renders the figures of section 2 of the Method page, blocks and target distances.

    python web/design/render_method.py tables data playground/derived/ps_curve.csv web/design/method_data
    python web/design/render_method.py figures web/design/method_data web/design/images

    python web/design/render_method.py uniform-tables data web/design/method_data [CONTACT_MAP]

`tables` reads the GM12878 inputs of production and the contact probability table of three cell
lines from `playground/ps_curve.py`. It writes the small tables the figures plot. `figures`
draws `method_blocks.png`, `method_exponent.png`, `method_default.png`, `method_loops.png` and
`method_hic.png` of section 2 and `method_anchor_terms.png` of section 3 from those tables alone.

`uniform-tables` writes the tables of section 5, the chain of one bead per kilobase: the beads
of two example anchors on both chains, the strength of the loops of the region and the kind of
target of every pair of nodes of one structure. It reads the contact map of the cell line under
`data/_hic/`. `figures` draws `method_uniform_anchor.png`, `method_uniform_loops.png` and
`method_uniform_targets.png` from them.
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


# The example anchors of section 5, by block and place in the block of the region: one anchor of 3 kb
# with three loops, and two anchors that overlap by 2 kb.
OTHER = "#b0afa9"
SINGLE = (1, [3])
OVERLAP = (1, [7, 8])
FLANK = 4_000


def uniform_tables(data_root: str, out: Path, contact_map: str = "") -> None:
    """Writes the tables of the chain of one bead per kilobase as a job on the region builds them.
    `contact_map` is the map of the contact cells, the deep map of the cell line or its cut to the
    region. Without it the job reads the largest map under `data_root`."""
    import gnome3d.skeleton as skeleton
    from gnome3d.data import ContactData
    from gnome3d.hierarchy import Level
    from gnome3d.io import parse_region
    from gnome3d.pipeline.coarse import build as cb
    from gnome3d.pipeline.ib.densify import densify
    from gnome3d.pipeline.ib.nodes import draw_ends
    from gnome3d.pipeline.ib.spans import anchor_spans, grid_edges
    from gnome3d.production import settings_for_cell
    from gnome3d.simulate import simulate

    root = str(Path(data_root).resolve())
    region = parse_region(f"{CHROM}:{REGION[0]}-{REGION[1]}")
    plain = settings_for_cell(CELL, root)
    s = settings_for_cell(CELL, root, uniform=True)
    if contact_map:
        s.data_contact_map = f"{Path(contact_map).resolve()}::/resolutions/5000"
    data = ContactData.from_files(s, [CHROM], region)
    state = cb.build_state(s, data, [CHROM], region)
    cl = state.clusters
    arcs = state.arcs.get(CHROM, [])
    ibs = [c for c in cl if c.level == Level.INTERACTION_BLOCK and len(c.children) > 1]
    law = s.polymer_law()

    def example(block: int, picks: list[int]) -> dict[str, object]:
        ids = list(ibs[block].children)
        genomic = [(cl[a].start, cl[a].end, cl[a].genomic_pos) for a in ids]
        lo_w = min(genomic[k][0] for k in picks) - FLANK
        hi_w = max(genomic[k][1] for k in picks) + FLANK
        _pos, fixed, starts, ends, *_rest = densify(np.zeros((len(ids), 3), dtype=np.float32), genomic, plain)
        default = [
            {"start": int(a), "end": int(b), "anchor": bool(f)}
            for a, b, f in zip(starts, ends, fixed, strict=True)
            if b > lo_w and a < hi_w
        ]
        edges = grid_edges(genomic, int(s.target_bp_per_subanchor))
        lo, hi, _knot = anchor_spans(genomic, edges)
        beads = [
            {
                "start": int(edges[i]),
                "end": int(edges[i + 1]),
                "anchors": [n for n, k in enumerate(picks) if lo[k] <= i < hi[k]],
                "covered": bool(((lo <= i) & (i < hi)).any()),
            }
            for i in range(len(edges) - 1)
            if edges[i + 1] > lo_w and edges[i] < hi_w
        ]
        first = next(i for i in range(len(edges) - 1) if edges[i + 1] > lo_w)
        loops = []
        for n, k in enumerate(picks):
            for arc_i in cl[ids[k]].arcs:
                arc = arcs[arc_i]
                other = arc.end if arc.start == ids[k] else arc.start
                loops.append({"anchor": n, "side": 1 if cl[other].genomic_pos > genomic[k][2] else -1, "pets": int(arc.score)})
        pair = np.array([[picks[lp["anchor"]], picks[lp["anchor"]]] for lp in loops], dtype=np.int32)
        draws = [
            [int(b) - first for b in draw_ends(lo, hi, pair, np.random.default_rng(seed))[:, 0]] for seed in (1, 2)
        ]
        return {
            "window": [int(lo_w), int(hi_w)],
            "anchors": [{"start": int(genomic[k][0]), "end": int(genomic[k][1])} for k in picks],
            "default": default,
            "uniform": beads,
            "loops": loops,
            "draws": draws,
        }

    (out / "uniform_anchors.json").write_text(json.dumps({"single": example(*SINGLE), "overlap": example(*OVERLAP)}, indent=1) + "\n")

    rows = []
    for ib in ibs:
        for a in ib.children:
            for arc_i in cl[a].arcs:
                arc = arcs[arc_i]
                if arc.start != a:
                    continue
                span = abs(cl[arc.end].genomic_pos - cl[arc.start].genomic_pos)
                q = float(law.arc_strength(int(arc.score), int(span), int(arc.factor)))
                rows.append((int(span), int(arc.score), q, q / (q + float(s.loop_dropout_scale))))
    pd.DataFrame(rows, columns=["span", "pets", "strength", "kept"]).to_csv(out / "uniform_loops.csv", index=False)

    # The kinds of the targets of one structure, read where the node solve builds its matrix.
    seen: dict[str, np.ndarray] = {}
    wrapped = {name: getattr(skeleton, name) for name in ("node_targets", "hold_by_cells", "hold_loci")}

    def keep(name: str):  # noqa: ANN202
        def call(*args, **kw):  # noqa: ANN002, ANN003, ANN202
            got = wrapped[name](*args, **kw)
            if name == "node_targets":
                seen["mid"] = np.array(args[0].mid, dtype=np.int64)
                seen["block"] = np.array(args[0].block, dtype=np.int64)
            seen.setdefault(name, np.array(got, dtype=np.float64))
            return got

        return call

    for name in wrapped:
        setattr(skeleton, name, keep(name))
    try:
        simulate(s, data, [CHROM], 1, region=region)
    finally:
        for name, fn in wrapped.items():
            setattr(skeleton, name, fn)
    first, cells, loci = seen["node_targets"], seen["hold_by_cells"], seen["hold_loci"]
    mid = seen["mid"]
    sep = np.abs(mid[:, None] - mid[None, :]).astype(np.float64)
    bg = np.maximum(1.0, (np.maximum(sep, 1.0) / max(int(law.s0_bp), 1)) ** law.nu)
    # 1 a loop, 2 the background under the range, 3 an enriched cell, 4 a depleted cell, 5 the
    # background between two stretches of 100 kb, 0 no target.
    kind = np.zeros(first.shape, dtype=np.int64)
    kind[first > 0] = 1
    kind[(first <= -1) & (first == cells)] = 2
    moved = cells != first
    kind[moved & (-cells < bg)] = 3
    kind[moved & (-cells >= bg)] = 4
    kind[loci != cells] = 5
    np.fill_diagonal(kind, -1)
    pd.DataFrame(kind).to_csv(out / "uniform_pairs.csv", index=False, header=False)
    pd.DataFrame({"mid": mid, "block": seen["block"]}).to_csv(out / "uniform_nodes.csv", index=False)
    iu = np.triu_indices(len(mid), 1)
    k = kind[iu]
    counts = {name: int((k == code).sum()) for code, name in enumerate(("none", "loop", "background", "enriched", "depleted", "stretches"))}
    fits = json.loads((out / "fits.json").read_text())
    fits["uniform_pairs"] = {"nodes": len(mid), "pairs": int(k.size), **counts}
    (out / "fits.json").write_text(json.dumps(fits, indent=1) + "\n")
    print(fits["uniform_pairs"])


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


def _loop_stub(ax: plt.Axes, x: float, y: float, side: int, reach: float, rise: float, colour: str) -> None:
    """Draws the start of a loop that leaves `x` upwards to one side and runs out of the panel."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    verts = [(x, y), (x, y + rise), (x + side * reach * 0.45, y + rise * 1.25), (x + side * reach, y + rise * 1.25)]
    ax.add_patch(PathPatch(MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]), fc="none", ec=colour, lw=1.3))


def uniform_anchor(tab: Path, out: Path) -> None:
    """Draws one anchor and two overlapping anchors as the data give them, as the beads of the
    chain of loop anchors and as the beads of the chain of one bead per kilobase in two structures."""
    from matplotlib.lines import Line2D

    both = json.loads((tab / "uniform_anchors.json").read_text())
    tints = [BLUE, "#eb6834"]
    edge = "#c9c8c3"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.7))
    rows = ["anchor in the data", "loop anchors\nand segments", "one bead per kb,\nstructure 1", "one bead per kb,\nstructure 2"]
    for ax, key, tag in zip(axes, ("single", "overlap"), ("a", "b"), strict=True):
        w0, w1 = both[key]["window"]
        ax.set_xlim(0, (w1 - w0) / 1e3)
        ax.set_ylim(-0.5, 3.6)
        ax.set_yticks([3.0, 2.0, 1.0, 0.0])
        ax.set_yticklabels(rows if tag == "a" else [""] * 4, fontsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel(f"kb from {w0 / 1e6:.3f} Mb on chromosome 1", fontsize=9.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        title = "one anchor of 3 kb with three loops" if key == "single" else "two anchors that overlap"
        ax.set_title(title, fontsize=9.5, color=INK, pad=6)
        letter(ax, tag)
    fig.tight_layout(w_pad=1.6, rect=(0, 0.1, 1, 1))
    fig.canvas.draw()
    # A bead is a circle on the page. Its size comes from the scale of the panel with the longer window.
    per_kb = min(ax.get_window_extent().width * 72 / fig.dpi / ax.get_xlim()[1] for ax in axes)
    size = 0.84 * per_kb
    for ax, key in zip(axes, ("single", "overlap"), strict=True):
        e = both[key]
        w0, w1 = e["window"]
        span = (w1 - w0) / 1e3
        top = size / 2 / (ax.get_window_extent().height * 72 / fig.dpi / 4.1)
        two = len(e["anchors"]) > 1

        def x(v: float, w0: int = w0) -> float:
            return (v - w0) / 1e3

        def centre(b: dict, x=x) -> float:  # noqa: ANN001
            return (x(b["start"]) + x(b["end"])) / 2

        def bead(at: float, y: float, fill: str, line: str, ax: plt.Axes = ax, span: float = span) -> None:
            if 0.0 <= at <= span:
                ax.plot([at], [y], marker="o", ms=size, color=fill, mec=line, mew=0.8, zorder=4, clip_on=False)

        def dna(a: dict, y: float, n: int, ax: plt.Axes = ax, x=x, top: float = top) -> None:  # noqa: ANN001
            ax.add_patch(plt.Rectangle((x(a["start"]), y - top - 0.1 - 0.07 * n), x(a["end"]) - x(a["start"]), 0.045, fc=tints[n], ec="none"))

        # The anchors as the data give them, one bar each.
        for n, a in enumerate(e["anchors"]):
            y = 3.0 + (0.16 if n == 0 else -0.16) * two
            ax.add_patch(plt.Rectangle((x(a["start"]), y - 0.11), x(a["end"]) - x(a["start"]), 0.22, fc=tints[n], ec="none"))
        for i, lp in enumerate(e["loops"]):
            a = e["anchors"][lp["anchor"]]
            y = 3.0 + (0.16 if lp["anchor"] == 0 else -0.16) * two
            _loop_stub(ax, centre(a) + 0.25 * lp["side"], y + 0.11, lp["side"], span, 0.26 + 0.07 * i, tints[lp["anchor"]])
        # The chain of loop anchors: an anchor is one bead, with the DNA it covers as a bar under it.
        picked = {(a["start"], a["end"]): n for n, a in enumerate(e["anchors"])}
        ax.plot([centre(b) for b in e["default"]], [2.0] * len(e["default"]), color=edge, lw=1.6, zorder=2)
        for b in e["default"]:
            n = picked.get((b["start"], b["end"]))
            if n is not None:
                bead(centre(b), 2.0, tints[n], "white")
                dna(b, 2.0, n)
            else:
                bead(centre(b), 2.0, OTHER if b["anchor"] else PALE, "white")
        for i, lp in enumerate(e["loops"]):
            a = e["anchors"][lp["anchor"]]
            _loop_stub(ax, centre(a), 2.0 + top, lp["side"], span, 0.26 + 0.07 * i, tints[lp["anchor"]])
        # The chain of one bead per kilobase in two structures: a loop ends on one bead of its anchor.
        for row, draw in zip((1.0, 0.0), e["draws"], strict=True):
            ax.plot([centre(b) for b in e["uniform"]], [row] * len(e["uniform"]), color=edge, lw=1.6, zorder=2)
            ends = {bead_of: lp["anchor"] for lp, bead_of in zip(e["loops"], draw, strict=True)}
            for k, b in enumerate(e["uniform"]):
                owners = b["anchors"]
                if k in ends:
                    bead(centre(b), row, tints[ends[k]], "white")
                elif owners:
                    bead(centre(b), row, WASH, MID)
                else:
                    bead(centre(b), row, OTHER if b["covered"] else GAP, "white" if b["covered"] else edge)
            for n, a in enumerate(e["anchors"]):
                dna(a, row, n)
            for i, (lp, bead_of) in enumerate(zip(e["loops"], draw, strict=True)):
                _loop_stub(ax, centre(e["uniform"][bead_of]), row + top, lp["side"], span, 0.26 + 0.07 * i, tints[lp["anchor"]])
    key_size = 7.5
    handles = [
        Line2D([], [], marker="o", ls="", ms=key_size, color=BLUE, mec="white", label="bead with the end of a loop"),
        Line2D([], [], marker="o", ls="", ms=key_size, color=WASH, mec=MID, label="bead inside the anchor, no loop end"),
        Line2D([], [], marker="o", ls="", ms=key_size, color=PALE, mec="white", label="bead between two anchors, loop anchors chain"),
        Line2D([], [], marker="o", ls="", ms=key_size, color=GAP, mec=edge, label="bead outside the anchor, one bead per kb"),
        Line2D([], [], marker="o", ls="", ms=key_size, color=OTHER, mec="white", label="bead of another anchor"),
        Line2D([], [], color=BLUE, lw=3, label="DNA of the anchor"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower center", ncol=2, handlelength=1.4, columnspacing=2.2)
    save(fig, out / "method_uniform_anchor.png")


def uniform_nodes(tab: Path, out: Path) -> None:
    """Draws the share of the structures that keep a loop against its strength and the kind of
    target of every pair of nodes of one structure of the region."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    loops = pd.read_csv(tab / "uniform_loops.csv")
    kind = pd.read_csv(tab / "uniform_pairs.csv", header=None).to_numpy()
    nodes = pd.read_csv(tab / "uniform_nodes.csv")
    n_of = json.loads((tab / "fits.json").read_text())["uniform_pairs"]
    n = len(nodes)
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 4.9), gridspec_kw={"width_ratios": [1.0, 1.0]})

    q = np.geomspace(0.05, 30, 200)
    left.plot(q, q / (q + 1), color=INK, lw=1.4, label="q / (q + 1)")
    left.scatter(loops.strength, loops.kept, s=22, color=BLUE, alpha=0.75, lw=0, zorder=3, label=f"the {len(loops)} loops of the region")
    left.axhline(0.5, color="#c9c8c3", lw=0.7, ls=(0, (3, 3)))
    left.axvline(1.0, color="#c9c8c3", lw=0.7, ls=(0, (3, 3)))
    left.set_xscale("log")
    left.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    left.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    left.minorticks_off()
    left.set_ylim(0, 1.02)
    left.set_box_aspect(1.0)
    left.set_anchor("N")
    left.set_xlabel("strength of the loop, q", fontsize=9.5)
    left.set_ylabel("share of the structures\nthat keep the loop", fontsize=9.5)
    left.legend(frameon=False, fontsize=8.5, loc="lower right")
    left.set_title("loops that a structure keeps", fontsize=9.5, color=INK, pad=6)
    style(left)
    letter(left, "a")

    colours = [GAP, BLUE, INK, "#eb6834", "#1baf7a", PALE]
    right.imshow(np.ma.masked_less(kind, 0), cmap=ListedColormap(colours), vmin=0, vmax=5, interpolation="nearest")
    for c in np.flatnonzero(np.diff(nodes.block)) + 0.5:
        right.axhline(c, color="white", lw=1.6)
        right.axvline(c, color="white", lw=1.6)
    blocks_seen = sorted(nodes.block.unique())
    centres = [float(np.flatnonzero(nodes.block == b).mean()) for b in blocks_seen]
    right.set_xticks(centres)
    right.set_xticklabels([str(k + 1) for k in range(len(centres))], fontsize=8.5)
    right.set_yticks(centres)
    right.set_yticklabels([str(k + 1) for k in range(len(centres))], fontsize=8.5)
    right.set_xlabel("block", fontsize=9.5)
    right.set_ylabel("block", fontsize=9.5)
    right.tick_params(length=0)
    for side in right.spines.values():
        side.set_visible(False)
    right.set_xlim(-0.5, n - 0.5)
    right.set_ylim(n - 0.5, -0.5)
    right.set_anchor("N")
    handles = [
        Patch(fc=BLUE, label=f"loop, {n_of['loop']} pairs"),
        Patch(fc=INK, label=f"background under 100 kb, {n_of['background']} pairs"),
        Patch(fc=PALE, label=f"background between two stretches of 100 kb, {n_of['stretches']} pairs"),
    ]
    if n_of["enriched"]:
        handles.append(Patch(fc="#eb6834", label=f"enriched cell, {n_of['enriched']} pairs"))
    if n_of["depleted"]:
        handles.append(Patch(fc="#1baf7a", label=f"depleted cell, {n_of['depleted']} pairs"))
    handles.append(Patch(fc=GAP, ec="#d9d8d3", label=f"no target, {n_of['none']:,} pairs"))
    right.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, handlelength=1.2)
    right.set_title(f"targets of the {n} nodes of one structure, {n_of['pairs']:,} pairs", fontsize=9.5, color=INK, pad=6)
    letter(right, "b")
    fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=2.0)
    save(fig, out / "method_uniform_nodes.png")


def main() -> None:
    if sys.argv[1] == "tables":
        tables(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
        return
    if sys.argv[1] == "uniform-tables":
        uniform_tables(sys.argv[2], Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "")
        return
    tab, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    blocks(tab, out)
    exponent(tab, out)
    default(tab, out)
    loop_targets(tab, out)
    hic_targets(tab, out)
    anchor_terms(tab, out)
    uniform_anchor(tab, out)
    uniform_nodes(tab, out)


if __name__ == "__main__":
    main()
