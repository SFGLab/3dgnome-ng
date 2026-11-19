"""The whole genome cross line readouts added up into the layout the figure script reads.

    python playground/rnapii/crossline_genome_collect.py <readouts dir of the runs> <out dir>

The runs are `readouts_crossline_genome.sh` and `microc_genome.sh`, one chromosome a call. The
out dir gets `readouts/` and `fig_inputs/` as `figures_two.py` takes them.

- `readouts/e3d_<ARM>/genes_<a>_to_<b>.tsv` is the gene tables of the chromosomes one under the
  other, a gene's cluster prefixed with its chromosome so no two chromosomes share one.
  `summary.tsv` holds the pair, the number of genes and `r_total`, the Spearman correlation of
  `total` with `log2fc` over all of them.
- `readouts/<LINE>_pooled.txt` has one row per arm, every score the mean over chromosomes
  weighted by chromosome length. The arm is named `<LINE>_<ARM>_genome`.
- `readouts/<LINE>_pileup.json` is the pile-up at the line's RNAPII loops added up loop by loop.
- `readouts/<LINE>_microc_loops.json` has the rows of `microc_loop_compare.py` with the loops
  of the chromosomes counted together. A share is the placed loops over the loops, and the
  share above the moved loops is taken against the arm's moved loops of every chromosome.
- `fig_inputs/ep_<LINE>.tsv` is the pair tables one under the other, with the chromosome.

- `readouts/by_chromosome.tsv` has line, arm, chrom, measure and value, one row a chromosome:
  `hic`, the raw stratum adjusted correlation of the summed map, and `microc_loops`, the share
  of Micro-C loops placed above the moved loops. The figures test two arms over these.

A chromosome whose file is missing is named and left out of that sum, so a partial run can be
looked at. The counts of chromosomes read are printed per file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINES = ("GM12878", "H1ESC", "HFFC6")
ARMS = ("E", "C", "R", "M")
CHROMS = tuple(f"chr{i}" for i in range(1, 23))
# hg38 autosome lengths in bp.
LENGTH = dict(
    zip(
        CHROMS,
        (248956422, 242193529, 198295559, 190214555, 181538259, 170805979, 159345973, 145138636, 138394717, 133797422, 135086622,
         133275309, 114364328, 107043718, 101991189, 90338345, 83257441, 80373285, 58617616, 64444167, 46709983, 50818468),
        strict=True,
    )
)
PAIRS = ("GM12878_to_H1ESC", "GM12878_to_HFFC6", "H1ESC_to_HFFC6")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def apa_score(p: np.ndarray) -> float:
    """The centre pixel over the mean of the 3 by 3 lower left corner, as pileup.py scores it."""
    c = p.shape[0] // 2
    corner = float(p[-3:, :3].mean())
    return float(p[c, c]) / corner if corner > 0 else float("nan")


def expression(src: Path, out: Path) -> None:
    for arm in ARMS:
        rows = []
        for pair in PAIRS:
            parts = []
            for chrom in CHROMS:
                path = src / f"e3d_{arm}" / chrom / f"genes_{pair}.tsv"
                if path.is_file():
                    t = pd.read_csv(path, sep="\t")
                    t["cluster"] = chrom + ":" + t["cluster"].astype(str)
                    t.insert(1, "chrom", chrom)
                    parts.append(t)
            if not parts:
                continue
            g = pd.concat(parts, ignore_index=True)
            d = out / "readouts" / f"e3d_{arm}"
            d.mkdir(parents=True, exist_ok=True)
            g.to_csv(d / f"genes_{pair}.tsv", sep="\t", index=False)
            ok = np.isfinite(g.total) & np.isfinite(g.log2fc)
            a, b = pair.split("_to_")
            rows.append({"pair": f"{a}->{b}", "chromosomes": len(parts), "n_with_deseq": int(ok.sum()),
                         "r_total": _spearman(g.total[ok].to_numpy(), g.log2fc[ok].to_numpy())})
        if rows:
            pd.DataFrame(rows).to_csv(out / "readouts" / f"e3d_{arm}" / "summary.tsv", sep="\t", index=False)
            print(f"expression {arm}: " + ", ".join(f"{r['pair']} {r['r_total']:+.3f} on {r['n_with_deseq']} genes of {r['chromosomes']} chromosomes" for r in rows))


def structure(src: Path, out: Path) -> None:
    for line in LINES:
        scores: dict[str, np.ndarray] = {}
        weight: dict[str, float] = {}
        models: dict[str, int] = {}
        piles: dict[str, np.ndarray | float] = {}
        loops = 0
        n_pooled = n_pile = 0
        for chrom in CHROMS:
            pooled = src / "structure" / f"{line}_{chrom}_pooled.txt"
            if pooled.is_file():
                n_pooled += 1
                for ln in pooled.read_text().splitlines():
                    f = ln.split()
                    if f and f[0].startswith(f"{line}_") and f[0].endswith(f"_{chrom}"):
                        arm = f[0][len(line) + 1 : -len(chrom) - 1]
                        v = np.array([float(x) for x in f[2:]])
                        scores[arm] = scores.get(arm, 0.0) + LENGTH[chrom] * v
                        weight[arm] = weight.get(arm, 0.0) + LENGTH[chrom]
                        models[arm] = int(f[1])
            pj, pz = src / "structure" / f"{line}_{chrom}_pileup.json", src / "structure" / f"{line}_{chrom}_pileup.npz"
            if pj.is_file() and pz.is_file():
                n_pile += 1
                rows = {r["arm"]: r for r in json.loads(pj.read_text())}
                z = np.load(pz)
                n = int(rows["map"]["n_loops"])
                loops += n
                piles["map"] = piles.get("map", 0.0) + n * z["map"]
                for arm in ARMS:
                    name = f"{line}_{arm}_{chrom}"
                    if name in z.files:
                        piles[arm] = piles.get(arm, 0.0) + n * z[name]
        d = out / "readouts"
        d.mkdir(parents=True, exist_ok=True)
        if scores:
            head = (f"chr1 to chr22, pooled ensemble against Hi-C at 25000 bp, mean over {n_pooled} chromosomes weighted by length\n"
                    f"  {'arm':>44s} {'n':>3s}   raw P  raw Sp  raw SCC   bal P  bal Sp  bal SCC   mapped bins only: raw SCC  bal SCC  bal stratum Spearman\n")
            body = ""
            for arm in ARMS:
                if arm in scores:
                    v = scores[arm] / weight[arm]
                    body += (f"  {f'{line}_{arm}_genome':>44s} {models[arm]:3d}  " + "  ".join(f"{x:6.3f}" for x in v[:6])
                             + "                    " + "   ".join(f"{x:6.3f}" for x in v[6:]) + "\n")
            (d / f"{line}_pooled.txt").write_text(head + body)
            print(f"structure {line}: {n_pooled} chromosomes, raw SCC " + ", ".join(f"{a} {scores[a][2] / weight[a]:.3f}" for a in ARMS if a in scores))
        if loops:
            ref = apa_score(np.asarray(piles["map"]) / loops)
            rows_out = [{"arm": "map", "n_models": 0, "n_loops": loops, "apa": ref}]
            rows_out += [{"arm": f"{line}_{a}_genome", "n_models": models.get(a, 10), "n_loops": loops, "apa": apa_score(np.asarray(piles[a]) / loops)} for a in ARMS if a in piles]
            (d / f"{line}_pileup.json").write_text(json.dumps(rows_out, indent=2) + "\n")
            print(f"pile-up {line}: {n_pile} chromosomes, {loops} loops, map {ref:.2f}, " + ", ".join(f"{r['arm'].split('_')[1]} {r['apa'] / ref:.2f} of it" for r in rows_out[1:]))


def microc(src: Path, out: Path) -> None:
    for line in LINES:
        count: dict[tuple[str, str], dict[str, float]] = {}
        n = 0
        for chrom in CHROMS:
            path = src / "microc" / f"{line}_{chrom}_5000.json"
            if not path.is_file():
                continue
            n += 1
            for r in json.loads(path.read_text())["rows"]:
                arm = "map" if r["arm"] == "map" else str(r["arm"])[len(line) + 1 : -len(chrom) - 1]
                c = count.setdefault((arm, str(r["class"])), {"loops": 0, "scored": 0, "placed": 0.0})
                c["loops"] += int(r["loops"])
                c["scored"] += int(r.get("scored", 0))
                c["placed"] += float(r.get("placed", 0.0)) * int(r["loops"])
        if not count:
            continue
        rows = []
        for (arm, cls), c in count.items():
            row: dict[str, object] = {"arm": "map" if arm == "map" else f"{line}_{arm}_genome", "class": cls, "loops": int(c["loops"]), "scored": int(c["scored"])}
            if arm != "map":
                moved = next((v for (a, k), v in count.items() if a == arm and k.startswith("moved")), None)
                share = c["placed"] / max(c["loops"], 1)
                row["placed"] = share
                if moved is not None:
                    row["placed_above_moved"] = share - moved["placed"] / max(moved["loops"], 1)
            rows.append(row)
        d = out / "readouts"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{line}_microc_loops.json").write_text(json.dumps({"binsize": 5000, "chromosomes": n, "rows": rows}, indent=1) + "\n")
        every = {a: next((r for r in rows if r["arm"] == f"{line}_{a}_genome" and r["class"] == "all"), None) for a in ARMS}
        print(f"Micro-C {line}: {n} chromosomes, {count.get(('map', 'all'), {}).get('loops', 0)} loops, placed above moved "
              + ", ".join(f"{a} {r['placed_above_moved']:.2f}" for a, r in every.items() if r and "placed_above_moved" in r))


def pairs(src: Path, out: Path) -> None:
    for line in LINES:
        parts = []
        for chrom in CHROMS:
            path = src / "pairs" / f"ep_{line}_{chrom}.tsv"
            if path.is_file():
                t = pd.read_csv(path, sep="\t")
                t.insert(1, "chrom", chrom)
                parts.append(t)
        if parts:
            d = out / "fig_inputs"
            d.mkdir(parents=True, exist_ok=True)
            e = pd.concat(parts, ignore_index=True)
            e.to_csv(d / f"ep_{line}.tsv", sep="\t", index=False)
            only = e[(e.set == "rnapii_only") & (e.arm == "R")]
            print(f"pairs {line}: {len(parts)} chromosomes, {len(only)} pairs an RNAPII loop alone joins")


def by_chromosome(src: Path, out: Path) -> None:
    rows = []
    for line in LINES:
        for chrom in CHROMS:
            pooled = src / "structure" / f"{line}_{chrom}_pooled.txt"
            if pooled.is_file():
                for ln in pooled.read_text().splitlines():
                    f = ln.split()
                    if f and f[0].startswith(f"{line}_") and f[0].endswith(f"_{chrom}"):
                        rows.append({"line": line, "arm": f[0][len(line) + 1 : -len(chrom) - 1], "chrom": chrom, "measure": "hic", "value": float(f[4])})
            path = src / "microc" / f"{line}_{chrom}_5000.json"
            if path.is_file():
                for r in json.loads(path.read_text())["rows"]:
                    if r["arm"] != "map" and r["class"] == "all" and "placed_above_moved" in r:
                        rows.append({"line": line, "arm": str(r["arm"])[len(line) + 1 : -len(chrom) - 1], "chrom": chrom,
                                     "measure": "microc_loops", "value": float(r["placed_above_moved"])})
    if rows:
        d = out / "readouts"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(d / "by_chromosome.tsv", sep="\t", index=False)
        print(f"by chromosome: {len(rows)} rows")


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    expression(src, out)
    structure(src, out)
    microc(src, out)
    pairs(src, out)
    by_chromosome(src, out)


if __name__ == "__main__":
    main()
