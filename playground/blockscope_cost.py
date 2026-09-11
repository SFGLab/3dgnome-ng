"""What does segment scope anchor placement buy, and what does it cost, per block source?

`use_segment_arcs` exists because anchors enter the per block arc MC collapsed on their block
centroid, so an arc whose two anchors sit in different blocks constrains nothing. Refining a
whole segment at once makes those arcs ordinary in chain arcs.

The benefit is therefore bounded by how many arcs cross a block boundary at all. TAD boundaries
come from a contact map call and cut wherever insulation says so, including through arcs.
Arc gap boundaries are placed where arc coverage falls to zero, so by construction almost
nothing crosses one. `cross_block_arc_frac` measures that directly and is the whole benefit.

The cost is the repulsion. `calc_anchor_expected_distances` marks every pair without an arc as
-1, which the arcs kernel scores as an unbounded 1/d push apart. Arcs grow linearly with the
anchors in a chain while those pairs grow quadratically, so widening the chain from block to
segment raises repulsion per arc in proportion to chain length. `rep_per_arc_*` is that ratio
under each scope and `cost_ratio` is how much worse segment scope is.

    python playground/blockscope_cost.py --configs slurm/ensemble/gm12878_chiapet_arcs.ini \
        slurm/ensemble/gm12878_chiapet_tads.ini --chrom chr1
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.hierarchy import Level  # noqa: E402
from gnome3d.io import parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import build_state  # noqa: E402
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
    anchor_set = set(anchors)

    # benefit: how many arcs have their two anchors in different blocks
    cross = same = 0
    per_block_arcs: dict[int, int] = defaultdict(int)
    per_seg_arcs: dict[int, int] = defaultdict(int)
    for a in arcs:
        if a.start not in anchor_set or a.end not in anchor_set:
            continue
        pa, pb = cl[a.start].parent, cl[a.end].parent
        if pa == pb:
            same += 1
            per_block_arcs[pa] += 1
        else:
            cross += 1
        sa, sb = cl[pa].parent, cl[pb].parent
        if sa == sb:
            per_seg_arcs[sa] += 1
    total = cross + same

    # cost: pairs without an arc, which the kernel repels, per arc actually present
    n_per_block = {b: len(cl[b].children) for b in blocks}
    seg_children: dict[int, int] = defaultdict(int)
    for b in blocks:
        seg_children[cl[b].parent] += n_per_block[b]

    def rep_per_arc(sizes: dict[int, int], arc_counts: dict[int, int]) -> float:
        pairs = sum(n * (n - 1) / 2 for n in sizes.values() if n > 1)
        got = sum(arc_counts.get(k, 0) for k in sizes)
        return (pairs - got) / got if got else float("nan")

    rb = rep_per_arc(n_per_block, per_block_arcs)
    rs = rep_per_arc(dict(seg_children), per_seg_arcs)
    sizes_b = np.array([n for n in n_per_block.values() if n > 0], dtype=float)
    sizes_s = np.array([n for n in seg_children.values() if n > 0], dtype=float)
    return {
        "anchors": len(anchors),
        "blocks": len(blocks),
        "segments": float(len(seg_children)),
        "arcs_in_chr": float(total),
        "cross_block_arcs": float(cross),
        "cross_block_arc_frac": cross / total if total else float("nan"),
        "median_anchors_per_block": float(np.median(sizes_b)) if sizes_b.size else float("nan"),
        "median_anchors_per_segment": float(np.median(sizes_s)) if sizes_s.size else float("nan"),
        "rep_per_arc_block": rb,
        "rep_per_arc_segment": rs,
        "cost_ratio": rs / rb if rb else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--chrom", default="chr1")
    ap.add_argument("--data-dirs", nargs="*", default=None)
    args = ap.parse_args()

    rows = []
    for i, c in enumerate(args.configs):
        dd = args.data_dirs[i] if args.data_dirs and i < len(args.data_dirs) else None
        r = describe(Path(c), args.chrom, dd)
        r["config"] = Path(c).name
        rows.append(r)
        print(f"[cost] {r['config']}", flush=True)

    keys = ["anchors", "blocks", "segments", "arcs_in_chr", "cross_block_arcs",
            "cross_block_arc_frac", "median_anchors_per_block", "median_anchors_per_segment",
            "rep_per_arc_block", "rep_per_arc_segment", "cost_ratio"]
    w = max(len(k) for k in keys) + 2
    print(f"\n{'metric':{w}s}" + "".join(f"{r['config'][:26]:>28s}" for r in rows))
    for k in keys:
        print(f"{k:{w}s}" + "".join(f"{r[k]:28.4f}" for r in rows))
    print("\n[cost] cross_block_arc_frac is the entire benefit; cost_ratio is what segment scope "
          "adds in repulsion per arc")


if __name__ == "__main__":
    main()
