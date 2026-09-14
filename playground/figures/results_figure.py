"""The validation figure: every arm on every cell, one panel per measure.

    python playground/figures/results_figure.py TABLE_DIR OUT.png

TABLE_DIR holds `<cell>_battery.txt` as the battery writes it. Arms are drawn in one fixed
order and colour whatever a cell has, so an arm keeps its colour from panel to panel and cell
to cell. Bars are grouped by cell. The exponent panel marks each cell's own fitted exponent,
which is the yardstick the battery reports against. Writes the PNG and an SVG beside it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CELLS = ["GM12878", "H1ESC", "HFFC6"]
# arm suffix -> (label, colour); the order is the drawing order and the legend order
ARMS: list[tuple[str, str, str]] = [
    ("reference", "reference 3dgnome", "#2a78d6"),
    ("prod", "3dgnome-ng, before", "#eb6834"),
    ("prod3", "3dgnome-ng, now", "#1baf7a"),
    ("mm_loops_md", "MultiMM, loops", "#eda100"),
    ("mm_comps_md", "MultiMM, loops + compartments", "#e87ba4"),
]
PANELS: list[tuple[str, str, str]] = [
    ("pearson", "Hi-C Pearson", "higher is better"),
    ("SCC", "stratum adjusted correlation", "higher is better"),
    ("multimm", "MultiMM's ensemble metric", "higher is better"),
    ("exponent", "distance exponent", "the cell's own fit is marked"),
    ("wb-sa", "overlaps within blocks, per 1,000 beads", "lower is better; the reference is off the scale"),
    ("xb", "overlaps across blocks, per 1,000 beads", "lower is better; the reference is off the scale"),
]
# The reference has no excluded volume and its own bead density, so its overlap counts run
# into the thousands per thousand beads and would flatten every other bar. They are reported
# in the note instead of the two overlap panels.
OFF_SCALE = {("wb-sa", "reference"), ("xb", "reference")}


def read_table(path: Path) -> tuple[dict[str, dict[str, float]], float | None]:
    """Rows by arm suffix, and the exponent yardstick the battery printed."""
    rows: dict[str, dict[str, float]] = {}
    cols: list[str] = []
    nu: float | None = None
    names = ["pearson", "spearman", "SCC", "multimm", "exponent", "vs_hic", "e20-100k", "e100k-1M", "Rg", "wb-aa", "wb-sa", "xb"]
    for line in path.read_text().splitlines():
        m = re.search(r"exponent yardstick: nu ([0-9.]+)", line)
        if m:
            nu = float(m.group(1))
        f = line.split()
        if not f:
            continue
        if f[0] == "arm":
            cols = names if "pearson" in f else []  # the saddle table has its own header
            continue
        if cols and len(f) == 2 + len(cols) and f[1].isdigit():
            arm = f[0].split("_", 1)[1] if "_" in f[0] else f[0]
            rows[arm] = {c: float(v.rstrip("x")) for c, v in zip(cols, f[2:], strict=True)}
    return rows, nu


def main() -> None:
    table_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = {c: read_table(table_dir / f"{c}_battery.txt") for c in CELLS}
    present = [a for a in ARMS if any(a[0] in data[c][0] for c in CELLS)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), dpi=150)
    width = 0.8 / len(present)
    for ax, (key, name, note) in zip(axes.ravel(), PANELS, strict=True):
        for j, (suffix, label, colour) in enumerate(present):
            xs, ys = [], []
            for i, c in enumerate(CELLS):
                if suffix in data[c][0] and (key, suffix) not in OFF_SCALE:
                    xs.append(i + (j - (len(present) - 1) / 2) * width)
                    ys.append(data[c][0][suffix][key])
            ax.bar(xs, ys, width=width * 0.92, color=colour, label=label, linewidth=0)
        if key == "exponent":
            for i, c in enumerate(CELLS):
                nu = data[c][1]
                if nu is not None:
                    ax.hlines(nu, i - 0.42, i + 0.42, colors="#0b0b0b", linestyles=(0, (2, 2)), linewidth=1)
        ax.text(0.0, 1.10, name, transform=ax.transAxes, fontsize=11, color="#0b0b0b")
        ax.text(0.0, 1.03, note, transform=ax.transAxes, fontsize=8.5, color="#52514e")
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels(CELLS, fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5, length=0)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(present), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("chr1:1-60 Mb, five structures per arm, against each cell's deep Hi-C", fontsize=11, x=0.02, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), h_pad=3.0)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    print(f"wrote {out}; arms drawn: {[a[1] for a in present]}")


if __name__ == "__main__":
    main()
