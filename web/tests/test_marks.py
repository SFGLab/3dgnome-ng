"""Enhancer and promoter marks on model beads and the distance table.

    python web/tests/test_marks.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from _check import check, done

from gnome3d_web.marks import ep_table, mark_beads


def main() -> int:
    starts = np.arange(0, 100_000, 2_000)
    ends = starts + 2_000
    genes = pd.DataFrame({"gene_id": ["G1", "G2"], "name": ["A", "B"], "chrom": "chr1", "tss": [10_500, 250_000], "strand": "+"})
    enh = pd.DataFrame({0: "chr1", 1: [40_000, 98_000], 2: [41_000, 99_000]})
    m = mark_beads("chr1", starts, ends, genes, enh)
    check("a TSS inside the chain marks its bead", m.promoters == {"G1": 5}, str(m.promoters))
    check("enhancers within 5 kb mark their beads", m.enhancers == [20, 49], str(m.enhancers))
    pos = np.zeros((len(starts), 3))
    pos[:, 0] = np.arange(len(starts))
    t = ep_table([pos, pos * 2], m, genes, starts, ends)
    row = t.set_index("gene_id").loc["G1"]
    check("the nearest enhancer of a gene and its mean distance over models", row["enhancer_bead"] == 20 and np.isclose(row["distance"], 22.5), str(row.to_dict()))
    check("a gene off the chain is not in the table", "G2" not in set(t["gene_id"]))
    return done()


if __name__ == "__main__":
    sys.exit(main())
