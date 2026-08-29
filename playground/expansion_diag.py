"""Why are trio structures 5x more extended than the cell lines at the same bond length?

Measured on chr1: trio HG00512 gives Rg 224.7 and within-block contact 0.040, GM12878 gives 44.7
and 0.538, with median bond lengths of 1.856 and 1.835. The local chain is therefore the same and
something is failing to fold it globally. Deeper contact input does not explain it on its own,
since more contacts should compact more, not less.

This compares the inputs that set the folding, per config, without running any MC:

  anchors, arcs        how many constraints exist at all
  arcs per anchor      constraint density; a sparser graph folds less
  dtn                  chain bond targets from genomic_length_to_distance, the local scale
  arc expected dist    pairwise arc targets from freq_to_distance
  ratio                arc target over median dtn. This is the load-bearing number: an arc
                       target far above the bond scale asks the chain to stay extended, while
                       one below it pulls the chain together. Two runs with matched bond
                       lengths can still differ five-fold in Rg through this ratio alone.

    python playground/expansion_diag.py --configs slurm/ensemble/hg00512_trio_fixed.ini \
        slurm/ensemble/gm12878_chiapet_tads.ini --chrom chr1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import (  # noqa: E402
    build_state,
    calc_anchor_expected_distances,
)
from gnome3d.settings import Settings  # noqa: E402


def describe(cfg_path: Path, chrom: str, data_dir: str | None) -> dict[str, float]:
    s = Settings()
    if not s.load_ini(str(cfg_path)):
        raise SystemExit(f"cannot load {cfg_path}")
    if data_dir:
        s.data_dir = data_dir

    chrs, bed = parse_chrs_arg(chrom)
    data = ContactData.from_files(s, chrs, bed)
    state = build_state(s, data, chrs, bed)
    cl = state.clusters

    anchors = [i for i, c in enumerate(cl) if c.level == Level.ANCHOR]
    blocks = [i for i, c in enumerate(cl) if c.level == Level.INTERACTION_BLOCK]
    arcs = state.arcs.get(chrom, [])

    # chain bond targets between consecutive anchors
    order = sorted(anchors, key=lambda i: cl[i].genomic_pos)
    gaps = [
        abs(cl[order[k + 1]].genomic_pos - cl[order[k]].genomic_pos) for k in range(len(order) - 1)
    ]
    dtn = np.array([s.genomic_length_to_distance(g) for g in gaps], dtype=float)

    # arc targets over one representative block, which is what the arc MC actually sees
    biggest = max(blocks, key=lambda b: len(cl[b].children)) if blocks else None
    arc_exp = np.array([], dtype=float)
    if biggest is not None and len(cl[biggest].children) > 1:
        m = calc_anchor_expected_distances(state, list(cl[biggest].children), chrom, None)
        arc_exp = m[m > 0.0]

    med_dtn = float(np.median(dtn)) if dtn.size else float("nan")
    med_arc = float(np.median(arc_exp)) if arc_exp.size else float("nan")
    return {
        "anchors": len(anchors),
        "blocks": len(blocks),
        "arcs": len(arcs),
        "arcs_per_anchor": len(arcs) / max(len(anchors), 1),
        "median_gap_kb": float(np.median(gaps)) / 1000 if gaps else float("nan"),
        "median_dtn": med_dtn,
        "median_arc_exp": med_arc,
        "arc_over_dtn": med_arc / med_dtn if med_dtn else float("nan"),
        "singletons": len(state.singletons),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--chrom", default="chr1")
    ap.add_argument("--data-dirs", nargs="*", default=None,
                    help="optional per-config data_dir override, same order as --configs")
    args = ap.parse_args()

    rows = []
    for i, c in enumerate(args.configs):
        dd = args.data_dirs[i] if args.data_dirs and i < len(args.data_dirs) else None
        r = describe(Path(c), args.chrom, dd)
        r["config"] = Path(c).name
        rows.append(r)
        print(f"[diag] {r['config']}", flush=True)

    keys = ["anchors", "blocks", "arcs", "arcs_per_anchor", "median_gap_kb",
            "median_dtn", "median_arc_exp", "arc_over_dtn", "singletons"]
    w = max(len(k) for k in keys) + 2
    head = "".join(f"{r['config'][:26]:>28s}" for r in rows)
    print(f"\n{'metric':{w}s}{head}")
    for k in keys:
        line = "".join(f"{r[k]:28.3f}" for r in rows)
        print(f"{k:{w}s}{line}")
    print("\n[diag] arc_over_dtn is the one to compare: a higher value keeps the chain extended")


if __name__ == "__main__":
    main()
