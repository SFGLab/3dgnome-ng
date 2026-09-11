"""Which anchor pairs the joint arcs matrix holds, by class and by block.

Builds the chromosome scope target matrix the way the joint solve does and counts, over
anchor pairs, the arcs, the background springs the contact background adds for arcless
pairs, and the pairs left to the repulsion alone, split by compartment class pair and by
whether the two anchors share a block. Says whether the data gives the solve anything to
sort blocks with.

    python playground/background_pairs.py CONFIG.ini REGION
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.data import ContactData
from gnome3d.io import parse_chrs_arg
from gnome3d.pipeline import coarse as cb
from gnome3d.pipeline.coarse.build import compartment_for_clusters
from gnome3d.settings import Settings
from gnome3d.skeleton import _collect_ib_work


def main() -> None:
    config, region = sys.argv[1], sys.argv[2]
    s = Settings()
    s.load_ini(config)
    chrom = region.split(":")[0]
    chrs, reg = parse_chrs_arg(region)
    data = ContactData.from_files(s, chrs, reg)
    state = cb.build_state(s, data, chrs, reg)
    seg_level = {c: [i for i, cl in enumerate(state.clusters) if int(cl.level) == 2 and cl.parent >= 0 and state.clusters[cl.parent].level == 1 and c == chrom] for c in chrs}
    segs = [i for i, cl in enumerate(state.clusters) if int(cl.level) == 2]
    work = _collect_ib_work(state, chrom, {chrom: segs})
    active = [a for _, _, ar in work for a in ar]
    block = np.concatenate([np.full(len(ar), k) for k, (_, _, ar) in enumerate(work)])
    heat = None
    if s.use_anchor_heatmap and state.singletons:
        heat, _ = cb.build_contact_heatmaps(state, active, chrom, with_subanchor=False)
    m = cb.calc_anchor_expected_distances(state, active, chrom, heat)
    cls = compartment_for_clusters(state, active, chrom)
    n = len(active)
    iu = np.triu_indices(n, k=1)
    v = m[iu]
    same = block[iu[0]] == block[iu[1]]
    ci, cj = cls[iu[0]], cls[iu[1]]
    kind = np.where((ci > 0) & (cj > 0), "AA", np.where((ci < 0) & (cj < 0), "BB", np.where(ci * cj < 0, "AB", "none")))
    arc = v > 0
    bg = v <= -0.75
    rep = (v < 0) & (v > -0.75)
    print(f"{region}: {n} anchors, {len(work)} blocks, classes A {int((cls > 0).sum())} B {int((cls < 0).sum())} none {int((cls == 0).sum())}")
    print(f"  {'pairs':<8}{'within':>28}{'cross':>28}")
    print(f"  {'':<8}{'arc':>9}{'background':>11}{'repulsion':>10}{'arc':>9}{'background':>11}{'repulsion':>10}")
    for k in ("AA", "BB", "AB"):
        sel = kind == k
        row = [int((sel & same & arc).sum()), int((sel & same & bg).sum()), int((sel & same & rep).sum()), int((sel & ~same & arc).sum()), int((sel & ~same & bg).sum()), int((sel & ~same & rep).sum())]
        print(f"  {k:<8}{row[0]:>9}{row[1]:>11}{row[2]:>10}{row[3]:>9}{row[4]:>11}{row[5]:>10}")
    cross_bg = bg & ~same
    if cross_bg.any():
        sep = np.abs(np.array([state.clusters[a].genomic_pos for a in active])[iu[0]] - np.array([state.clusters[a].genomic_pos for a in active])[iu[1]])
        law = s.polymer_law()
        ratio = -v[cross_bg] / np.array([law.background(int(x)) for x in sep[cross_bg]])
        print(f"  cross block background springs: {int(cross_bg.sum())} of {int((~same).sum())} cross pairs ({100 * cross_bg.mean() / max((~same).mean(), 1e-9):.2f}%), target over law median {np.median(ratio):.2f}, separations median {np.median(sep[cross_bg]) / 1e6:.1f} Mb")


if __name__ == "__main__":
    main()
