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
  arc_over_dtn         arc target over median dtn, the local folding scale

Measured on chr1 the arc geometry is the same in both, 0.213 against 0.201, so the arcs are not
what inflates the trio. The input that does differ is the singleton count, 1.54M against 30.5k,
because the trio feeds Hi-C at 25 kb where the cell line feeds ChIA-PET singletons. Those
singletons build the segment heatmap, and `create_distance_heatmap` leaves a pair unconstrained
when its frequency is below 1e-6 while giving every other pair an explicit distance target. A
sparse heatmap therefore lets the chain fold freely and a dense one holds most segment pairs
apart at once, which is why the second block of metrics measures that heatmap:

  seg_bins             segment bins on this chromosome, the heatmap's side length
  seg_density          fraction of off-diagonal pairs carrying a target at all. This is the
                       load-bearing number. Near 1 means every segment pair is held at a
                       prescribed distance and the chain cannot compact.
  seg_avg_dist         mean of the positive targets, the scale the segment MC works at
  seg_med_target       median positive target
  seg_p90_target       90th percentile target
  seg_clipped_frac     fraction of targets sitting on the heatmap_distance_stretching ceiling

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
from gnome3d.hierarchy import Level, set_level  # noqa: E402
from gnome3d.io import create_singleton_heatmap, parse_chrs_arg  # noqa: E402
from gnome3d.pipeline.coarse.build import (  # noqa: E402
    CoarseState,
    add_long_pet_to_segment_heatmap,
    build_state,
    calc_anchor_expected_distances,
    compute_segment_bins,
)
from gnome3d.pipeline.coarse.heatmap import (  # noqa: E402
    create_distance_heatmap,
    get_diagonal_size,
    normalize_heatmap,
    normalize_heatmap_diagonal_total,
)
from gnome3d.settings import Settings  # noqa: E402


def segment_heatmap_stats(state: CoarseState, s: Settings) -> dict[str, float]:
    """Density and target scale of the segment level heatmap the singletons build.

    Repeats the frequency to distance path of `reconstruct_segment_level` without running MC.
    Returns the metrics described in the module doc, all nan when the chromosome has one
    segment and no heatmap exists.
    """
    current_level = set_level(
        Level.SEGMENT - Level.CHROMOSOME, state.chr_root, state.clusters, state.chrs
    )
    bins, start_ind, total_size, bin_lengths_mb = compute_segment_bins(state, current_level)
    nan = float("nan")
    if total_size < 2:
        return {"seg_bins": total_size, "seg_density": nan, "seg_avg_dist": nan,
                "seg_med_target": nan, "seg_p90_target": nan, "seg_clipped_frac": nan}

    h_raw = create_singleton_heatmap(
        state.singletons, bins, start_ind, total_size, bin_lengths_mb=bin_lengths_mb
    )
    add_long_pet_to_segment_heatmap(state, h_raw, bins, start_ind, total_size)
    h_norm = normalize_heatmap(h_raw, total_size)
    h_norm = normalize_heatmap_diagonal_total(h_norm, total_size, 1.0)
    dist, avg = create_distance_heatmap(s, h_norm, total_size, inter=False)

    # a pair counts as constrained when it carries a positive target. Band cells are -1 and
    # unconstrained cells are 0, so both are excluded, and the denominator drops the band too.
    diag = get_diagonal_size(h_norm, total_size)
    iu = np.triu_indices(total_size, k=max(diag, 1))
    cells = dist[iu]
    active = cells[cells > 0.0]
    ceiling = avg * s.heatmap_distance_stretching
    return {
        "seg_bins": float(total_size),
        "seg_density": float(active.size / cells.size) if cells.size else nan,
        "seg_avg_dist": avg,
        "seg_med_target": float(np.median(active)) if active.size else nan,
        "seg_p90_target": float(np.percentile(active, 90)) if active.size else nan,
        "seg_clipped_frac": (
            float((active >= ceiling - 1e-9).mean()) if active.size else nan
        ),
    }


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
    row = {
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
    row.update(segment_heatmap_stats(state, s))
    return row


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
            "median_dtn", "median_arc_exp", "arc_over_dtn", "singletons",
            "seg_bins", "seg_density", "seg_avg_dist", "seg_med_target",
            "seg_p90_target", "seg_clipped_frac"]
    w = max(len(k) for k in keys) + 2
    head = "".join(f"{r['config'][:26]:>28s}" for r in rows)
    print(f"\n{'metric':{w}s}{head}")
    for k in keys:
        line = "".join(f"{r[k]:28.3f}" for r in rows)
        print(f"{k:{w}s}{line}")
    print("\n[diag] seg_density is the one to compare: near 1 holds every segment pair apart")


if __name__ == "__main__":
    main()
