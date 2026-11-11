"""Figures of the RNAPII comparison, `design/2026-11-11-rnapii-release.md`, Part 2.

Every readout enters one long table of dataset, kind (line or person), arm, metric, value and
structure count n. `figure_main` draws panels c to f of the main figure, the pile-ups, the
structure and expression readouts as paired dots and the trade-off. `figure_grid` draws the
supplementary grid of every dataset by every metric. Panels a and b, the model snapshots and
the example locus, are made from renders and from the locus `--loci` proposes.

Arms keep one colour each everywhere, C blue, R orange, M aqua, E yellow, and every mark also
carries its arm letter, since aqua and yellow sit under 3:1 contrast on white. An arm with fewer
than ten structures is marked n=k. HFFc6 is labelled with BJ when its prep record says the
RNAPII loops came from BJ.

    python playground/rnapii/figure_rnapii.py <manifest.json> <out_dir>
    python playground/rnapii/figure_rnapii.py --loci <mcool> <region> <loops.bedpe> <C_dir> <R_dir>

The manifest lists the readout files: `battery`, `saddle` and `pileup` entries carry dataset,
kind, path and `arms`, a map from the directory name in the file to the arm letter;
`expression` entries carry dataset, kind and the path of a CSV with columns arm, metric, value
and n; `records` maps a line to its `_xc_prep.json`; `piles` maps `dataset/arm` to an npz and
key, for panel c.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ARMS = ("E", "C", "R", "M")
COLOUR = {"C": "#2a78d6", "R": "#eb6834", "M": "#1baf7a", "E": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# +1 higher is better, -1 lower is better, 0 closest to one is better.
DIRECTION = {
    "pearson": 1, "spearman": 1, "scc": 1, "multimm": 1, "exponent": 1, "eig_r": 1,
    "saddle": 1, "apa_vs_map": 0, "wb_aa": -1, "wb_sa": -1, "xb": -1,
    "partial": -1, "rho_3d": -1, "e1": -1, "e2": 1, "e3": -1, "jump_rho": -1, "family_sep": 1,
}  # fmt: skip
STRUCTURE = ("scc", "pearson", "multimm", "saddle", "eig_r", "apa_vs_map", "wb_aa", "xb")
EXPRESSION = ("partial", "rho_3d", "e1", "e3", "jump_rho", "family_sep")
NAMES = {"HFFC6": "HFFc6", "H1ESC": "H1"}


def dataset_label(dataset: str, records: dict[str, dict[str, object]]) -> str:
    """The display name, with the stand in line when the RNAPII loops came from one."""
    name = NAMES.get(dataset, dataset)
    src = str(records.get(dataset, {}).get("rnapii_source", ""))
    return f"{name} (RNAPII from BJ)" if src.startswith("RNAPII from BJ") else name


def collect(manifest: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    """The long table from the manifest's readout files."""
    rows: list[dict[str, object]] = []
    for kind in ("battery", "saddle"):
        for e in manifest.get(kind, []):
            arms = dict(e["arms"])  # type: ignore[arg-type]
            for r in json.loads(Path(str(e["path"])).read_text()):
                if r["arm"] not in arms:
                    continue
                for k, v in r.items():
                    if k not in ("arm", "n"):
                        rows.append({"dataset": e["dataset"], "kind": e["kind"], "arm": arms[r["arm"]],
                                     "metric": k, "value": float(v), "n": int(r["n"])})  # fmt: skip
    for e in manifest.get("pileup", []):
        arms = dict(e["arms"])  # type: ignore[arg-type]
        data = json.loads(Path(str(e["path"])).read_text())
        ref = next((float(r["apa"]) for r in data if r["arm"] == "map"), float("nan"))
        for r in data:
            if r["arm"] in arms:
                rows.append({"dataset": e["dataset"], "kind": e["kind"], "arm": arms[r["arm"]],
                             "metric": "apa_vs_map", "value": float(r["apa"]) / ref, "n": int(r["n_models"])})  # fmt: skip
    for e in manifest.get("expression", []):
        for r in pd.read_csv(str(e["path"])).itertuples():
            rows.append({"dataset": e["dataset"], "kind": e["kind"], "arm": str(r.arm),
                         "metric": str(r.metric), "value": float(r.value), "n": int(r.n)})  # fmt: skip
    df = pd.DataFrame(rows, columns=["dataset", "kind", "arm", "metric", "value", "n"])
    dup = df[df.duplicated(["dataset", "arm", "metric"], keep=False)]
    if len(dup):
        raise ValueError(
            "two rows for one dataset, arm and metric; give every arm directory a unique name:\n"
            + dup.to_string(index=False)
        )
    return df


def best_arm(values: dict[str, float], metric: str) -> str | None:
    """The arm whose value is best for the metric, None when the metric has no direction."""
    d = DIRECTION.get(metric)
    values = {a: v for a, v in values.items() if np.isfinite(v)}
    if d is None or not values:
        return None
    if d == 0:
        return min(values, key=lambda a: abs(values[a] - 1.0))
    return max(values, key=lambda a: d * values[a])


def _style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _order(df: pd.DataFrame) -> list[str]:
    kinds = df.drop_duplicates("dataset").set_index("dataset")["kind"]
    return sorted(kinds.index, key=lambda d: (kinds[d] != "line", d))


def figure_grid(df: pd.DataFrame, records: dict[str, dict[str, object]], out: Path) -> Figure:
    """Every dataset by every metric; each cell lists the arms' values and is tinted by the best."""
    datasets = _order(df)
    present = list(dict.fromkeys(df["metric"]))
    metrics = [m for m in (*STRUCTURE, *EXPRESSION) if m in present] + [
        m for m in present if m not in STRUCTURE + EXPRESSION
    ]
    fig, ax = plt.subplots(figsize=(1.6 * len(metrics) + 2.5, 0.55 * len(datasets) + 1.2))
    ax.set_xlim(0, len(metrics))
    ax.set_ylim(len(datasets), 0)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(metrics):
            cell = df[(df["dataset"] == ds) & (df["metric"] == m)]
            vals = {str(r.arm): float(r.value) for r in cell.itertuples()}
            ns = {str(r.arm): int(r.n) for r in cell.itertuples()}
            best = best_arm(vals, m)
            if best:
                ax.add_patch(plt.Rectangle((j + 0.02, i + 0.03), 0.96, 0.94, color=COLOUR[best], alpha=0.18, linewidth=0))
            lines = [
                f"{a} {vals[a]:.2f}" + (f" n={ns[a]}" if ns[a] < 10 else "")
                for a in ARMS
                if a in vals
            ]
            ax.text(j + 0.5, i + 0.5, "\n".join(lines), ha="center", va="center", fontsize=6, color=INK)
    ax.set_xticks(np.arange(len(metrics)) + 0.5, metrics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(datasets)) + 0.5, [dataset_label(d, records) for d in datasets], fontsize=8)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Arms per dataset and metric. Tint marks the better arm: C blue, R orange, M aqua, E yellow",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def _spread(ys: list[float], gap: float) -> list[float]:
    """Label heights pushed apart to at least `gap`, in the order of `ys`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + gap)
    return out


def _paired(ax: plt.Axes, df: pd.DataFrame, metric: str, records: dict[str, dict[str, object]], title: str) -> None:
    """One line per dataset across the arms it has. Trio individuals are thin and drawn first,
    cell lines thick and labelled at their last arm. Dots take the arm colour, the axis the letter."""
    sub = df[df["metric"] == metric]
    arms = [a for a in ARMS if a in set(sub["arm"])]
    x = {a: i for i, a in enumerate(arms)}
    ends: list[tuple[str, float, float]] = []
    datasets = sorted(_order(sub), key=lambda d: sub[sub["dataset"] == d]["kind"].iloc[0] == "line")
    for ds in datasets:
        d = sub[sub["dataset"] == ds].set_index("arm")
        line = d["kind"].iloc[0] == "line"
        xs = [x[a] for a in arms if a in d.index]
        ys = [float(d.loc[a, "value"]) for a in arms if a in d.index]
        ax.plot(xs, ys, color=MUTED if line else GRID, linewidth=2 if line else 1, zorder=2 if line else 1)
        for a in arms:
            if a in d.index:
                v, n = float(d.loc[a, "value"]), int(d.loc[a, "n"])
                ax.scatter(x[a], v, s=40 if line else 12, color=COLOUR[a],
                           edgecolor="white", linewidth=1.5 if line else 0.5, zorder=3 if line else 1)  # fmt: skip
                if n < 10:
                    ax.annotate(f"n={n}", (x[a], v), xytext=(-4, 5), textcoords="offset points",
                                fontsize=6, color=MUTED, ha="right")  # fmt: skip
        if line and ys:
            ends.append((dataset_label(ds, records), xs[-1], ys[-1]))
    lo, hi = ax.get_ylim()
    for (text, x0, y0), y in zip(ends, _spread([e[2] for e in ends], (hi - lo) * 0.045), strict=True):
        ax.annotate(text, (x0, y0), xytext=(x0 + 0.15, y), textcoords="data", fontsize=7, color=INK,
                    va="center", arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8})  # fmt: skip
    ax.set_xticks(range(len(arms)), arms)
    ax.set_xlim(-0.4, len(arms) - 1 + 1.6)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.text(0.0, -0.13, "thick: cell lines, thin: trio individuals", transform=ax.transAxes, fontsize=7, color=MUTED)
    _style(ax)


def figure_main(
    df: pd.DataFrame, piles: dict[str, np.ndarray], records: dict[str, dict[str, object]], out: Path
) -> Figure:
    """Panels c to f. `piles` maps `dataset/arm` to a pile-up window, `map` for the measured one."""
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 1.4])
    ax_c = fig.add_subplot(gs[0, :])
    ax_d, ax_e, ax_f = (fig.add_subplot(gs[1, i]) for i in range(3))

    ax_c.axis("off")
    ax_c.set_title("c  pile-up at RNAPII loops, observed over expected", fontsize=9, color=INK, loc="left")
    if piles:
        keys = list(piles)
        vmax = max(float(np.nanmax(p)) for p in piles.values())
        for k, key in enumerate(keys):
            inset = ax_c.inset_axes((k / len(keys) + 0.01, 0.0, 1 / len(keys) - 0.02, 0.85))
            inset.imshow(piles[key], cmap="Blues", vmin=0, vmax=vmax)
            inset.set_xticks([])
            inset.set_yticks([])
            ds, arm = key.split("/")
            inset.set_title(f"{dataset_label(ds, records)} {arm}", fontsize=7, color=INK)
    else:
        ax_c.text(0.5, 0.5, "no pile-ups given", ha="center", color=MUTED, fontsize=8)

    _paired(ax_d, df, "scc", records, "d  structure, SCC against Hi-C")
    expr = next((m for m in ("partial", "e3", "rho_3d") if m in set(df["metric"])), None)
    if expr is None:
        for ax, name in ((ax_e, "e  expression"), (ax_f, "f  trade-off")):
            ax.axis("off")
            ax.set_title(name, fontsize=9, color=INK, loc="left")
            ax.text(0.5, 0.5, "no expression readout yet", ha="center", color=MUTED, fontsize=8, transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        return fig
    _paired(ax_e, df, expr, records, f"e  expression, {expr}")

    ax_f.axhline(0, color=GRID, linewidth=1)
    ax_f.axvline(0, color=GRID, linewidth=1)
    piv = df[df["metric"].isin(["scc", expr])].pivot_table(index=["dataset", "arm"], columns="metric", values="value")
    for ds in _order(df):
        if (ds, "C") not in piv.index:
            continue
        base = piv.loc[(ds, "C")]
        for arm in ("R", "M"):
            if (ds, arm) in piv.index:
                p = piv.loc[(ds, arm)]
                dx = float(p["scc"] - base["scc"])
                dy = DIRECTION[expr] * float(p[expr] - base[expr])
                line = df[df["dataset"] == ds]["kind"].iloc[0] == "line"
                ax_f.scatter(dx, dy, s=40 if line else 14, color=COLOUR[arm], edgecolor="white",
                             linewidth=1.5 if line else 0.5, zorder=2 if line else 1)  # fmt: skip
                if line:
                    ax_f.annotate(f"{arm} {dataset_label(ds, records)}", (dx, dy), xytext=(4, 3),
                                  textcoords="offset points", fontsize=6, color=INK)  # fmt: skip
    ax_f.set_xlabel("change in SCC from C", fontsize=8, color=MUTED)
    ax_f.set_ylabel(f"gain in expression signal, minus change in {expr}", fontsize=8, color=MUTED)
    ax_f.set_title("f  trade-off, R and M against C", fontsize=9, color=INK, loc="left")
    ax_f.text(0.0, -0.2, "large dots: cell lines, small: trio individuals", transform=ax_f.transAxes, fontsize=7, color=MUTED)
    _style(ax_f)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def candidate_loci(
    obs: np.ndarray,
    c_sim: np.ndarray,
    r_sim: np.ndarray,
    bin_starts: np.ndarray,
    binsize: int,
    loops: list[tuple[str, int, int, str, int, int, int]],
    window_bp: int = 1_000_000,
    top: int = 10,
) -> list[tuple[int, int, float, float]]:
    """Windows for panel b, half overlapping, ranked by the map's observed over expected at the
    RNAPII loops inside times how much R's differs from C's there. Each row is start, end, the
    map's mean and the mean absolute difference between R and C."""
    from pileup import observed_over_expected

    o, c, r = (observed_over_expected(m) for m in (obs, c_sim, r_sim))
    first = int(bin_starts[0])
    ends = [(((s1 + e1) // 2 - first) // binsize, ((s2 + e2) // 2 - first) // binsize) for _, s1, e1, _, s2, e2, _ in loops]
    w = window_bp // binsize
    rows: list[tuple[int, int, float, float]] = []
    for lo in range(0, max(1, len(bin_starts) - w + 1), max(1, w // 2)):
        inside = [(a, b) for a, b in ends if lo <= a and b < lo + w]
        if not inside:
            continue
        ia, ib = np.array(inside).T
        rows.append((first + lo * binsize, first + (lo + w) * binsize, float(o[ia, ib].mean()),
                     float(np.abs(r[ia, ib] - c[ia, ib]).mean())))  # fmt: skip
    return sorted(rows, key=lambda t: t[2] * t[3], reverse=True)[:top]


def _loci_cli(args: list[str]) -> None:
    """Print the candidate windows for panel b from a map and the C and R ensembles."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crosscell_prep import read_bedpe

    from playground.validation_battery import CONTACT_BEADS, load
    from validation.metrics.hic import observed_hic, simulated_contacts

    mcool, region, loops_path, c_dir, r_dir = args
    binsize = 10_000
    chrom = region.split(":")[0]
    obs, starts = observed_hic(mcool, region, binsize, balance=False)
    sims = []
    for d in (c_dir, r_dir):
        m = np.zeros_like(obs)
        for f in sorted(Path(d).glob("*.cif")):
            pos, _, mid, _ = load(f)
            rad = CONTACT_BEADS * float(np.median(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
            m += simulated_contacts(pos, mid, starts, binsize, rad)
        sims.append(m)
    lo, hi = int(starts[0]), int(starts[-1]) + binsize
    loops = [lp for lp in read_bedpe(Path(loops_path)) if lp[0] == chrom and lp[1] >= lo and lp[5] <= hi]
    for s0, e0, m, dlt in candidate_loci(obs, sims[0], sims[1], starts, binsize, loops):
        print(f"{chrom}:{s0}-{e0}  map O/E at loops {m:.2f}  |R - C| {dlt:.2f}")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--loci":
        _loci_cli(args[1:])
        return
    manifest = json.loads(Path(args[0]).read_text())
    out = Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    records = {k: json.loads(Path(v).read_text()) for k, v in manifest.get("records", {}).items()}
    piles = {k: np.load(v["path"])[v["key"]] for k, v in manifest.get("piles", {}).items()}
    df = collect(manifest)
    missing = sorted(set(df.loc[df["kind"] == "line", "dataset"]) - set(records))
    if missing:
        raise SystemExit(f"no prep record in the manifest for {missing}; labels would hide a stand in line")
    df.to_csv(out / "rnapii_long.csv", index=False)
    figure_main(df, piles, records, out / "rnapii_main.png")
    figure_main(df, piles, records, out / "rnapii_main.pdf")
    figure_grid(df, records, out / "rnapii_grid.png")
    figure_grid(df, records, out / "rnapii_grid.pdf")
    print(f"wrote {out}/rnapii_main.(png|pdf), rnapii_grid.(png|pdf) and rnapii_long.csv")


if __name__ == "__main__":
    main()
