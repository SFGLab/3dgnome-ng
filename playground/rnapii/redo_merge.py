"""The inputs of the figures from two sets of runs of the uniform chain, with and without the loop dropout.

    python playground/rnapii/redo_merge.py <readouts with> <fig_inputs with> \
        <readouts without> <fig_inputs without> <out dir> [<resolutions>]

The four folders are those of `crossline_genome_collect.py` and `model_maps.py` for the two readout
roots. The out dir gets `readouts/` and `fig_inputs/` as `figures_two.py` takes them.

- Every comparison with Hi-C is from the runs with the dropout: `<LINE>_pooled.txt`,
  `<LINE>_pileup.json`, the rows of `by_chromosome.tsv` whose measure starts with `hic` and the
  contact maps of `maps_GM12878.npz`.
- The enhancer to promoter distances, the link to expression and the Micro-C loops are from the
  runs without it: `ep_<LINE>.tsv`, `e3d_<ARM>/`, `<LINE>_microc_loops.json`, the rows of
  `by_chromosome.tsv` whose measure starts with `microc_loops` and the structures of the locus in
  `maps_GM12878.npz`.

With `<resolutions>`, the folder of the step `resolutions` of `readouts_crossline_genome.sh` for the
runs with the dropout, the correlations with Hi-C are those of the contacts per bead pair on the
pairs up to 2 Mb apart. The rows `hic` of `by_chromosome.tsv` are the score at 25 kb, the rows
`hic <res> kb, <from> to <to> kb` the same score by resolution and separation, and
`hic_per_bead_pair.tsv` has the mean of a line and arm at 25 kb over the chromosomes by length,
which `fig_data.agreement` reads in place of `<LINE>_pooled.txt`.

`fig_inputs/runs_note.txt` holds the sentence on the two sets that the figures print.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from crossline_genome_collect import LENGTH  # noqa: E402

LINES = ("GM12878", "H1ESC", "HFFC6")
PER_PAIR = " The correlations with Hi-C count the contacts per pair of beads, on the pairs up to 2 Mb apart."
NOTE = (
    "Every comparison with Hi-C is on models built with the loop dropout. The enhancer to promoter distances, "
    "the link to expression and the Micro-C loops are on models built without it."
)


def per_bead_pair(folder: Path) -> pd.DataFrame:
    """Returns the rows by chromosome of the correlation with Hi-C per bead pair, named as the plain ones."""
    rows = []
    for path in sorted(folder.glob("*.json")):
        line = path.name.split("_")[0]
        table = json.loads(path.read_text())
        chrom = table["chrom"]
        for r in table["rows"]:
            if r["bins"] != "all" or not np.isfinite(r["scc_pair"]):
                continue
            arm = str(r["arm"])[len(line) + 1 : -len(chrom) - 1]
            res, lo, hi = int(r["res"]) // 1000, int(r["sep_lo"]) // 1000, int(r["sep_hi"]) // 1000
            row = {"line": line, "arm": arm, "chrom": chrom, "value": float(r["scc_pair"])}
            rows.append({**row, "measure": f"hic {res} kb, {lo} to {hi} kb"})
            if (res, lo, hi) == (25, 0, 2000):
                rows.append({**row, "measure": "hic"})
    return pd.DataFrame(rows, columns=["line", "arm", "chrom", "measure", "value"])


def main() -> None:
    with_r, with_i, without_r, without_i, out = (Path(x) for x in sys.argv[1:6])
    resolutions = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    readouts, inputs = out / "readouts", out / "fig_inputs"
    readouts.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    for line in LINES:
        shutil.copy(with_r / f"{line}_pooled.txt", readouts)
        shutil.copy(with_r / f"{line}_pileup.json", readouts)
        shutil.copy(without_r / f"{line}_microc_loops.json", readouts)
        shutil.copy(without_i / f"ep_{line}.tsv", inputs)
    for d in sorted(without_r.glob("e3d_*")):
        shutil.copytree(d, readouts / d.name, dirs_exist_ok=True)
    a = pd.read_csv(with_r / "by_chromosome.tsv", sep="\t")
    b = pd.read_csv(without_r / "by_chromosome.tsv", sep="\t")
    hic = a[a.measure.str.startswith("hic")]
    if resolutions is not None:
        pair = per_bead_pair(resolutions)
        plain = hic[hic.measure == "hic"]
        assert len(pair[pair.measure == "hic"]) == len(plain), f"{len(plain)} rows by chromosome and {len(pair[pair.measure == 'hic'])} per bead pair"
        hic = pair
        mean = pair[pair.measure == "hic"].assign(w=lambda t: t.chrom.map(LENGTH))
        mean = mean.groupby(["line", "arm"]).apply(lambda t: float(np.average(t.value, weights=t.w)), include_groups=False)
        mean.rename("value").reset_index().to_csv(readouts / "hic_per_bead_pair.tsv", sep="\t", index=False)
    rows = pd.concat([hic, b[b.measure.str.startswith("microc_loops")]])
    assert resolutions is not None or len(rows) == len(a) == len(b), f"by_chromosome.tsv: {len(a)} and {len(b)} rows give {len(rows)}"
    rows.to_csv(readouts / "by_chromosome.tsv", sep="\t", index=False)
    for f in sorted(with_i.iterdir()):
        if f.is_file() and not f.name.startswith(("ep_", "maps_")):
            shutil.copy(f, inputs)
    maps, locus = np.load(with_i / "maps_GM12878.npz"), np.load(without_i / "maps_GM12878.npz")
    assert sorted(maps.files) == sorted(locus.files), "the two maps files hold different keys"
    np.savez_compressed(
        inputs / "maps_GM12878.npz",
        **{k: (locus if k.startswith(("ensemble_", "locus_")) else maps)[k] for k in maps.files},
    )
    (inputs / "runs_note.txt").write_text(NOTE + (PER_PAIR if resolutions is not None else "") + "\n")
    print(f"wrote {out}: {len(rows)} rows by chromosome, {len(maps.files)} arrays of the maps")


if __name__ == "__main__":
    main()
