"""Figures of the trace of the pairs 100 to 500 kb apart with RNAPII loops, design/2026-11-25-rnapii-band.md.

    python playground/rnapii/fig_band_trace.py GENOME_DIR CHR19_DIR OUT_DIR

GENOME_DIR holds `<LINE>_<chrom>.json` of `band_trace.py` over the two arms of the genome runs.
CHR19_DIR holds `trace.json` and `end_pairs.json` of `slurm/ensemble/trace_arcs_ev_score.sh`.
Three figures go to OUT_DIR: `band_problem`, `band_pairs` and `band_arms`, as png and pdf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID = "#1a1a1a", "#5a5955", "#e4e3df"
MAP, CTCF, RNAPII = "#8c8c8c", "#2a78d6", "#eb6834"
LINES = ("GM12878", "H1ESC", "HFFC6")
LINE_LABEL = {"GM12878": "GM12878", "H1ESC": "H1", "HFFC6": "HFFc6"}
BANDS = ((0, "under 100 kb"), (100_000, "100 to 500 kb"), (500_000, "0.5 to 2 Mb"))
CLASSES = ("RNAPII loop", "CTCF loop", "two RNAPII anchors", "one RNAPII anchor", "no RNAPII anchor")
CLASS_LABEL = {
    "RNAPII loop": "joined by an\nRNAPII loop",
    "CTCF loop": "joined by a\nCTCF loop",
    "two RNAPII anchors": "two RNAPII\nanchors, no loop",
    "one RNAPII anchor": "one RNAPII\nanchor",
    "no RNAPII anchor": "no RNAPII\nanchor",
}
SIZES = dict(zip([f"chr{i}" for i in range(1, 23)], [248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622, 133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468], strict=True))
# The arms of chr19 in the order of the figure: key of the scores, label, colour.
ARMS = (
    ("ctcf", "CTCF loops", CTCF),
    ("contactC", "CTCF loops, a kept loop at one bead", CTCF),
    ("contactC_r288", "CTCF loops, a kept loop at one bead, radius of today", CTCF),
    ("s0", "RNAPII loops all dropped, their beads kept", CTCF),
    ("base", "CTCF and RNAPII loops", RNAPII),
    ("noev", "no excluded volume in the node solve", RNAPII),
    ("r133", "its radius at the contact radius", RNAPII),
    ("contact", "a kept loop at one bead", RNAPII),
    ("contact_r288", "a kept loop at one bead, radius of today", RNAPII),
    ("half01", "a kept loop near the contact radius", RNAPII),
    ("ctcf_nodrop", "CTCF loops, every loop kept", CTCF),
    ("rnapii_nodrop", "CTCF and RNAPII loops, every loop kept", RNAPII),
    ("s0n", "every loop kept, RNAPII loops without pull", RNAPII),
)

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


def genome(folder: Path) -> tuple[dict, dict]:
    """Returns the scores per line, band and arm as arrays over chromosomes with their weights,
    and the classes per line summed over chromosomes."""
    scores: dict = {}
    classes: dict = {}
    for line in LINES:
        for chrom, size in SIZES.items():
            d = json.loads((folder / f"{line}_{chrom}.json").read_text())
            for r in d["scc"]:
                if r["bins"] != "of them mapped":
                    continue
                arm = r["arm"].split("_")[-1]
                scores.setdefault((line, r["sep_lo"], arm), []).append((r["scc_pair_bal"], r["scc_pair"], size))
            for c in d["classes"]:
                acc = classes.setdefault((line, c["class"]), {"pairs": 0, "map": 0.0, "C": 0.0, "R": 0.0, "rate_C": 0.0, "rate_R": 0.0})
                acc["pairs"] += c["pairs"]
                acc["map"] += c["pairs"] * c["map_balanced"]
                for arm in ("C", "R"):
                    acc[arm] += c["pairs"] * c[f"{line}_{arm}"]["over_mean"]
                    acc[f"rate_{arm}"] += c["pairs"] * c[f"{line}_{arm}"]["per_1000"]
    return scores, classes


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def problem(scores: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (lo, label) in zip(axes, BANDS, strict=True):
        for k, line in enumerate(LINES):
            for j, (arm, colour) in enumerate((("C", CTCF), ("R", RNAPII))):
                v = np.array(scores[(line, lo, arm)])
                mean = float(np.average(v[:, 0], weights=v[:, 2]))
                x = k + (j - 0.5) * 0.36
                ax.bar(x, mean, 0.32, color=colour)
                ax.text(x, mean + 0.008, f"{mean:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
            c, r = np.array(scores[(line, lo, "C")])[:, 0], np.array(scores[(line, lo, "R")])[:, 0]
            ax.text(k, -0.075, f"higher on\n{int((r > c).sum())} of 22", ha="center", va="top", fontsize=6.5, color=MUTED)
        ax.set_xticks(range(3), [LINE_LABEL[line] for line in LINES])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.set_title(f"pairs {label} apart", loc="left")
        ax.set_ylim(0, 0.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("stratum adjusted correlation\nwith balanced Hi-C at 10 kb")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CTCF), plt.Rectangle((0, 0), 1, 1, color=RNAPII)]
    fig.legend(handles, ["3D-GNOMEᴺᴳ, CTCF loops", "3D-GNOMEᴺᴳ, CTCF and RNAPII loops"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.text(0.5, -0.16, "Chromosomes 1 to 22, weighted by length, on the bins where both models have beads.\nUnder each pair: the chromosomes where the model with RNAPII loops is higher.", ha="center", va="top", fontsize=7, color=MUTED)
    save(fig, out, "band_problem")


def pairs(classes: dict, ends: dict, out: Path) -> None:
    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.05), hspace=0.75, wspace=0.42)
    for k, line in enumerate(LINES):
        ax = fig.add_subplot(grid[0, k])
        for i, cls in enumerate(CLASSES):
            acc = classes[(line, cls)]
            for j, (key, colour) in enumerate((("map", MAP), ("C", CTCF), ("R", RNAPII))):
                v = acc[key] / acc["pairs"]
                ax.bar(i + (j - 1) * 0.27, v, 0.25, color=colour)
        ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xticks(range(len(CLASSES)), [CLASS_LABEL[c] for c in CLASSES], fontsize=5.6, rotation=90)
        ax.tick_params(axis="x", length=0)
        ax.set_title(LINE_LABEL[line], loc="left")
        ax.set_ylim(0, 2.1)
        if k == 0:
            ax.set_ylabel("contacts of a pair of bins over\nthe mean of its separation")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MAP, CTCF, RNAPII)]
    fig.legend(handles, ["balanced Hi-C", "model, CTCF loops", "model, CTCF and RNAPII loops"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.text(0.02, 0.975, "a  Pairs of 10 kb bins 100 to 500 kb apart, by the loops at them, chromosomes 1 to 22", fontsize=9, color=INK, ha="left")

    ax = fig.add_subplot(grid[1, 0:2])
    sets = ("RNAPII loop", "CTCF loop", "two RNAPII ends, no loop", "RNAPII end and chain", "two chain beads")
    labels = ("ends of an\nRNAPII loop", "ends of a\nCTCF loop", "two RNAPII\nends, no loop", "RNAPII end\nand a chain\nbead", "two chain\nbeads")
    shown = [a for a in (("ctcf", "CTCF loops", CTCF, 1.0), ("base", "CTCF and RNAPII loops", RNAPII, 1.0), ("rnapii_nodrop", "the same, every loop kept", RNAPII, 0.45)) if a[0] in ends]
    for i, name in enumerate(sets):
        for j, (key, _, colour, alpha) in enumerate(shown):
            v = 100 * ends[key][name]["within_1.33"]
            x = i + (j - (len(shown) - 1) / 2) * 0.27
            ax.bar(x, v, 0.25, color=colour, alpha=alpha)
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    ax.set_xticks(range(len(sets)), labels, fontsize=6)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("pairs of beads in contact, %")
    ax.set_title("b  Beads 100 to 500 kb apart in contact, chr19 of GM12878", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for _, _, c, a in shown]
    ax.legend(handles, [s[1] for s in shown], loc="upper right", fontsize=6.5)

    ax = fig.add_subplot(grid[1, 2])
    edges = np.arange(0, 12.25, 0.25)
    mids = (edges[:-1] + edges[1:]) / 2
    for key, label, colour, alpha in shown:
        h = np.array(ends[key]["RNAPII loop"]["histogram"], dtype=float)
        ax.plot(mids, h / h.sum() / 0.25, color=colour, alpha=alpha, linewidth=1.4, label=label)
    ax.axvline(1.33, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(1.5, 0.02, "contact\nradius", fontsize=6.5, color=INK, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 12)
    ax.set_xlabel("distance of the ends, bonds")
    ax.set_ylabel("share of pairs per bond")
    ax.set_title("c  Ends of RNAPII loops", loc="left")
    ax.set_ylim(0, None)
    save(fig, out, "band_pairs")


def arms(trace: dict, out: Path) -> None:
    rows = {(r["arm"], r["sep_lo"]): r for r in trace["scc"] if r["bins"] == "of them mapped"}
    shown = [a for a in ARMS if (a[0], 0) in rows]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.34 * len(shown) + 0.9), sharey=True)
    for ax, (lo, label) in zip(axes, BANDS, strict=True):
        for i, (key, _, colour) in enumerate(shown):
            v = rows[(key, lo)]["scc_pair_bal"]
            ax.barh(i, v, 0.7, color=colour, alpha=1.0 if key in ("ctcf", "base") else 0.55)
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", ha="left", fontsize=7, color=INK)
        ax.set_title(f"{label}", loc="left")
        ax.set_xlim(0, 0.45)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    axes[0].set_yticks(range(len(shown)), [a[1] for a in shown])
    axes[0].tick_params(axis="y", length=0)
    fig.text(0.5, -0.02, "stratum adjusted correlation with balanced Hi-C at 10 kb, chr19 of GM12878, on the bins where every arm has beads", ha="center", va="top", fontsize=8, color=INK)
    save(fig, out, "band_arms")


def main() -> None:
    genome_dir, chr19_dir, out = (Path(p) for p in sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True)
    scores, classes = genome(genome_dir)
    problem(scores, out)
    pairs(classes, json.loads((chr19_dir / "end_pairs.json").read_text()), out)
    arms(json.loads((chr19_dir / "trace.json").read_text()), out)


if __name__ == "__main__":
    main()
