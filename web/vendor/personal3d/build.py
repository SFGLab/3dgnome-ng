"""One haplotype's engine inputs for a region, from a panel and the haplotype's variants.

The panel's network for the person is the consensus of the other people. The rules take the
whole chromosome's loops, which set their distance window and strengths, and only the region's
variants, the narrowed deletion rule of personal3d.motifs when motifs are given. The variants
upstream of the region shift it as a whole. With no variants this is the panel arm, the control
the variants are read against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, hicdiffusion, motifs, panel, rewrite, variants
from .lift import Lift


def region_inputs(
    P: panel.Panel,
    others: list[str],
    chrom: str,
    start: int,
    end: int,
    svs: list[variants.SV],
    base: pd.DataFrame,
    segments: pd.DataFrame,
    centromeres: pd.DataFrame,
    template: Path,
    out: Path,
    name: str,
    motifs_of: dict | None = None,
    hic: tuple[np.ndarray, int] | None = None,
    knockouts: set | None = None,
) -> tuple[Path, str, Lift]:
    """Write the inputs and config, and return the config, the region in the haplotype's
    coordinates and the haplotype's lift. hic is a predicted window and its reference start,
    knockouts the anchors the haplotype's SNVs knock out, snv.knockouts."""
    L = Lift(svs)
    inside = variants.in_region(svs, chrom, start, end)
    before = L.edge(start) - start
    length = int(P.anchors.end.max()) + 10_000_000
    loops = {}
    for f in panel.FACTORS:
        clusters = P.clusters(P.consensus(f, others))
        segs = rewrite.domains(clusters)
        if motifs_of is not None:
            clipped, ref_of = motifs.narrow(clusters, inside, motifs_of)
            got = rewrite.apply(clipped, segs, {chrom: length}, inside, ref_of, knockouts, motifs.THRESHOLD)
        else:
            got = rewrite.apply(clusters, segs, {chrom: length}, inside, None, knockouts)
        loops[f] = [
            rewrite.Loop(lp.chrom, (lp.a[0] + before, lp.a[1] + before), (lp.b[0] + before, lp.b[1] + before), lp.orient, lp.weight, lp.ref_a, lp.ref_b)
            for lp in got
            if start <= (lp.ref_a or lp.a)[0] and (lp.ref_b or lp.b)[1] <= end
        ]
    sing = engine.lift_bedpe(base, {chrom: L}) if svs else base
    if hic is not None:
        cond, ref_start = hic
        sing = engine.splice_window(sing, chrom, L.edge(ref_start), cond, hicdiffusion.BIN)
    seg = segments.copy()
    seg[1] = L.many(seg[1].to_numpy())
    seg[2] = seg[1]
    seg = seg[seg[1] >= 0]
    cen = centromeres.copy()
    cen[1], cen[2] = L.many(cen[1].to_numpy()), L.many(cen[2].to_numpy())
    cen = cen[(cen[1] >= 0) & (cen[2] >= 0)]
    keys = engine.write_inputs(out, name, loops, sing, seg, cen)
    ini = engine.write_config(template, keys, out / f"{name}.ini")
    region = f"{chrom}:{L.edge(start)}-{L.edge(end - 1) + 1}"
    return ini, region, L
