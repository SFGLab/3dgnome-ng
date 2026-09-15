"""The GM12878 region figure: the original pipeline's models against ours on four windows.

    python playground/figures/region_figure.py TABLE_DIR OUT.png

TABLE_DIR holds `GM12878_r<k>_battery.txt` for k in 1 to 4, as `slurm/ensemble/reference_regions.sh`
writes them. One panel per measure, bars grouped by region, the arms in the validation figure's
colours. Writes the PNG and an SVG beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground.figures.results_figure import ARMS, read_table  # noqa: E402

REGIONS: list[tuple[str, str]] = [
    ("r1", "0.85-3.5"),
    ("r2", "12.6-13.85"),
    ("r3", "15.8-17.4"),
    ("r4", "18.3-20.2"),
]
PANELS: list[tuple[str, str, str]] = [
    ("pearson", "Hi-C Pearson", "higher is better"),
    ("SCC", "stratum adjusted correlation", "higher is better"),
    ("multimm", "MultiMM's ensemble metric", "higher is better"),
    ("exponent", "distance exponent", "the cell's own fit is marked"),
]


def main() -> None:
    table_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = {}
    for r, _ in REGIONS:
        rows, nu = read_table(table_dir / f"GM12878_{r}_battery.txt")
        # the table names an arm GM12878_r1_prod3, and read_table keeps r1_prod3 of it
        data[r] = ({k.split("_", 1)[1]: v for k, v in rows.items()}, nu)
    present = [a for a in ARMS if any(a[0] in data[r][0] for r, _ in REGIONS)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), dpi=150)
    width = 0.8 / len(present)
    for ax, (key, name, note) in zip(axes, PANELS, strict=True):
        for j, (suffix, label, colour) in enumerate(present):
            xs, ys = [], []
            for i, (r, _) in enumerate(REGIONS):
                if suffix in data[r][0]:
                    xs.append(i + (j - (len(present) - 1) / 2) * width)
                    ys.append(data[r][0][suffix][key])
            ax.bar(xs, ys, width=width * 0.92, color=colour, label=label, linewidth=0)
        if key == "exponent":
            for i, (r, _) in enumerate(REGIONS):
                nu = data[r][1]
                if nu is not None:
                    ax.hlines(nu, i - 0.42, i + 0.42, colors="#0b0b0b", linestyles=(0, (2, 2)), linewidth=1)
            ax.axhline(0.0, color="#c3c2b7", linewidth=0.6)
        ax.text(0.0, 1.10, name, transform=ax.transAxes, fontsize=11, color="#0b0b0b")
        ax.text(0.0, 1.03, note, transform=ax.transAxes, fontsize=8.5, color="#52514e")
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels([lab for _, lab in REGIONS], fontsize=8)
        ax.set_xlabel("chr1 window, Mb", fontsize=8.5, color="#52514e")
        ax.tick_params(axis="y", labelsize=8.5, length=0)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(present), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "GM12878 chr1 windows, scored on the deep Hi-C: the original pipeline's 100 models per window, "
        "our five chr1:1-60 Mb structures cut to the window, and MultiMM's five",
        fontsize=10.5, x=0.02, ha="left", color="#0b0b0b",
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.08, 1, 0.94), w_pad=2.5)
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".svg"), facecolor="white")
    print(f"wrote {out}; arms drawn: {[a[1] for a in present]}")


if __name__ == "__main__":
    main()
