"""The inputs of the figures from two sets of runs of the uniform chain, with and without the loop dropout.

    python playground/rnapii/redo_merge.py <readouts with> <fig_inputs with> \
        <readouts without> <fig_inputs without> <out dir>

The four folders are those of `crossline_genome_collect.py` and `model_maps.py` for the two readout
roots. The out dir gets `readouts/` and `fig_inputs/` as `figures_two.py` takes them.

- Every comparison with Hi-C is from the runs with the dropout: `<LINE>_pooled.txt`,
  `<LINE>_pileup.json`, the rows of `by_chromosome.tsv` whose measure starts with `hic` and the
  contact maps of `maps_GM12878.npz`.
- The enhancer to promoter distances, the link to expression and the Micro-C loops are from the
  runs without it: `ep_<LINE>.tsv`, `e3d_<ARM>/`, `<LINE>_microc_loops.json`, the rows of
  `by_chromosome.tsv` whose measure starts with `microc_loops` and the structures of the locus in
  `maps_GM12878.npz`.

`fig_inputs/runs_note.txt` holds the sentence on the two sets that the figures print.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINES = ("GM12878", "H1ESC", "HFFC6")
NOTE = (
    "Every comparison with Hi-C is on models built with the loop dropout. The enhancer to promoter distances, "
    "the link to expression and the Micro-C loops are on models built without it."
)


def main() -> None:
    with_r, with_i, without_r, without_i, out = (Path(x) for x in sys.argv[1:6])
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
    rows = pd.concat([a[a.measure.str.startswith("hic")], b[b.measure.str.startswith("microc_loops")]])
    assert len(rows) == len(a) == len(b), f"by_chromosome.tsv: {len(a)} and {len(b)} rows give {len(rows)}"
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
    (inputs / "runs_note.txt").write_text(NOTE + "\n")
    print(f"wrote {out}: {len(rows)} rows by chromosome, {len(maps.files)} arrays of the maps")


if __name__ == "__main__":
    main()
