"""Figure and table of the uniform chain arms against production.

    python playground/uniform_arms_figure.py SCORES_DIR OUT_DIR

SCORES_DIR holds the files of `slurm/ensemble/arms_trial.sh`: `<cell>_<region>_pooled.txt` and
`<cell>_<region>_battery_balance_no.txt`. The figure has one panel per metric and one row per
cell and region. The table is `arms.md` beside it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# arm suffix, label, colour, filled
ARMS = [
    ("prod", "default chain", "#52514e", True),
    ("nomotif", "default chain, orientation off", "#52514e", False),
    ("uni", "uniform chain", "#2a78d6", True),
    ("uni_nomotif", "uniform chain, orientation off", "#2a78d6", False),
    ("uni_w01", "uniform chain, span weight 0.1", "#eb6834", True),
    ("uni_w01_nomotif", "uniform chain, span weight 0.1, orientation off", "#eb6834", False),
]
# key, panel title, reference line
PANELS = [
    ("raw_scc", "Pooled SCC, raw Hi-C", None),
    ("bal_scc", "Pooled SCC, balanced Hi-C", None),
    ("raw_sp", "Pooled Spearman, raw Hi-C", None),
    ("bal_p", "Pooled Pearson, balanced Hi-C", None),
    ("exponent", "Distance exponent over the exponent of Hi-C", 1.0),
    ("overlaps", "Overlapping pairs per thousand beads", None),
]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def arm_of(name: str, prefix: str) -> str:
    return name[len(prefix) + 1 :]


def read_scores(scores: Path) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    """Returns {(cell, region): {arm: {metric: value}}}."""
    out: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for pooled in sorted(scores.glob("*_pooled.txt")):
        prefix = pooled.name[: -len("_pooled.txt")]
        m = re.match(r"(.+?)_(chr\w+?)_(\d+)_(\d+)$", prefix)
        if m is None:
            continue
        cell = m.group(1)
        region = f"{m.group(2)}:{int(m.group(3)) / 1e6:g}-{int(m.group(4)) / 1e6:g} Mb"
        arms: dict[str, dict[str, float]] = {}
        for line in pooled.read_text().splitlines():
            f = line.split()
            if len(f) >= 11 and f[0].startswith(prefix):
                arms[arm_of(f[0], prefix)] = {
                    "raw_p": float(f[2]),
                    "raw_sp": float(f[3]),
                    "raw_scc": float(f[4]),
                    "bal_p": float(f[5]),
                    "bal_sp": float(f[6]),
                    "bal_scc": float(f[7]),
                }
        battery = scores / f"{prefix}_battery_balance_no.txt"
        if battery.is_file():
            for line in battery.read_text().splitlines():
                f = line.split()
                if len(f) >= 14 and f[0].startswith(prefix) and arm_of(f[0], prefix) in arms:
                    arms[arm_of(f[0], prefix)].update(
                        {
                            "exponent": float(f[7].rstrip("x")),
                            "rg": float(f[10]),
                            "overlaps": float(f[11]) + float(f[12]) + float(f[13]),
                        }
                    )
        out[(cell, region)] = arms
    return out


def figure(data: dict[tuple[str, str], dict[str, dict[str, float]]], path: Path) -> None:
    rows = list(data)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 2.6 + 1.05 * len(rows)), facecolor=SURFACE)
    step = 0.12
    for ax, (key, title, ref) in zip(axes.ravel(), PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        for r, row in enumerate(rows):
            for a, (arm, _label, colour, filled) in enumerate(ARMS):
                v = data[row].get(arm, {}).get(key)
                if v is None:
                    continue
                y = r + (a - (len(ARMS) - 1) / 2) * step
                ax.plot(
                    [v],
                    [y],
                    marker="o",
                    markersize=8,
                    markerfacecolor=colour if filled else SURFACE,
                    markeredgecolor=colour,
                    markeredgewidth=1.8,
                    linestyle="none",
                    zorder=3,
                )
        if ref is not None:
            ax.axvline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{c}  {g}" for c, g in rows], fontsize=9, color=INK)
        ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for r in range(len(rows) - 1):
            ax.axhline(r + 0.5, color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=8,
            linestyle="none",
            markerfacecolor=colour if filled else SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.8,
            label=label,
        )
        for _arm, label, colour, filled in ARMS
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK
    )
    fig.suptitle(
        "3D-GNOMEᴺᴳ, uniform chain against the default chain, five structures per arm",
        fontsize=13,
        color=INK,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def table(data: dict[tuple[str, str], dict[str, dict[str, float]]], path: Path) -> None:
    cols = [
        ("raw_scc", "SCC raw"),
        ("bal_scc", "SCC balanced"),
        ("raw_sp", "Spearman raw"),
        ("bal_p", "Pearson balanced"),
        ("exponent", "exponent over Hi-C"),
        ("overlaps", "overlaps per thousand beads"),
        ("rg", "Rg"),
    ]
    lines = [
        "| cell | region | arm | " + " | ".join(c for _, c in cols) + " |",
        "|---|---|---|" + "---|" * len(cols),
    ]
    for (cell, region), arms in data.items():
        for arm, label, _colour, _filled in ARMS:
            if arm not in arms:
                continue
            vals = " | ".join(
                f"{arms[arm][k]:.3f}" if k in arms[arm] else "" for k, _ in cols
            )
            lines.append(f"| {cell} | {region} | {label} | {vals} |")
    lines += ["", "Mean over the cells and regions:", ""]
    lines += ["| arm | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
    for arm, label, _colour, _filled in ARMS:
        means = []
        for k, _ in cols:
            v = [a[arm][k] for a in data.values() if arm in a and k in a[arm]]
            means.append(f"{sum(v) / len(v):.3f}" if v else "")
        lines.append(f"| {label} | " + " | ".join(means) + " |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    scores_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = read_scores(scores_dir)
    figure(scores, out_dir / "arms.png")
    table(scores, out_dir / "arms.md")
    print(f"{len(scores)} cell and region rows -> {out_dir / 'arms.png'}, {out_dir / 'arms.md'}")
