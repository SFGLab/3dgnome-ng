"""The engine's inputs for one request, written to the job's directory.

The cell type's loops of the region's chromosome are rewired by the sample's variants with the
2019 rules of personal3d, CTCF and RNAPII separately, and the loops wholly inside the region
are kept. The anchors are the cell type's own, lifted to the haplotype's coordinates, plus any
loop end the rules made. The singletons of the whole chromosome are kept, since the engine fits
the contact decay on the file, lifted end by end, with a HiCDiffusion window spliced in when
one is given. Segments and centromeres are lifted the same way. The config is the cell type's
with only its data section replaced. Only variants inside the region are applied, so
coordinates before the region do not move.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from personal3d import engine, hicdiffusion, motifs, rewrite
from personal3d.lift import Lift
from personal3d.variants import SV

from .catalog import CellType
from .regions import Region
from .settings_schema import apply_overrides

ENGINE_FACTOR = {"CTCF": "CTCF", "RNAPII": "RNAPOL2"}


@dataclass
class Inputs:
    config: Path
    region: str
    n_svs: int
    loops: dict[str, int]


def _read(path: Path, chrom: str, ncol: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    df = df[df[0] == chrom]
    return df.iloc[:, :ncol].reset_index(drop=True) if df.shape[1] >= ncol else df.reset_index(drop=True)


def _orientation(anchors: pd.DataFrame) -> dict[tuple[int, int], str]:
    return {(int(s), int(e)): (str(o) if len(anchors.columns) > 3 else "N") for s, e, o in anchors.iloc[:, 1:4].itertuples(index=False)}


def _clusters(loops: pd.DataFrame, anchors: pd.DataFrame) -> list[list]:
    """Loops as the rules' nine columns, orientation from the anchor holding each end's midpoint."""
    starts = anchors[1].to_numpy()
    ends = anchors[2].to_numpy()
    orient = anchors[3].astype(str).to_numpy() if anchors.shape[1] > 3 else np.full(len(anchors), "N")

    def o(s: int, e: int) -> str:
        m = (s + e) // 2
        k = int(np.searchsorted(starts, m, side="right")) - 1
        return str(orient[k]) if k >= 0 and ends[k] >= m else "N"

    return [[r[0], int(r[1]), int(r[2]), r[3], int(r[4]), int(r[5]), o(int(r[1]), int(r[2])), o(int(r[4]), int(r[5])), int(r[6])] for r in loops.itertuples(index=False)]


def build_inputs(
    cell: CellType,
    region: Region,
    factors: str,
    svs: list[SV],
    out: Path,
    hic: tuple[np.ndarray, int] | None = None,
    background: str = "hic",
) -> Inputs:
    """Write the inputs for one haplotype under `out` and return the config and the region in
    the haplotype's coordinates. `hic` is a predicted HiCDiffusion window and its reference start.
    `background` none turns off the contact terms and keeps the singletons for the exponent fit.
    RNAPII alone has its loop ends as anchors, orientation N, and no CTCF orientation term."""
    out.mkdir(parents=True, exist_ok=True)
    chrom = region.chrom
    L = Lift(svs)
    anchors = _read(cell.anchors_for(factors), chrom, 4).sort_values(1).reset_index(drop=True)
    if factors == "RNAPII":
        ends = _read(cell.clusters_for("RNAPII")[0][1], chrom, 7)
        spans = list(zip(ends[1], ends[2], strict=True)) + list(zip(ends[4], ends[5], strict=True))
        hit = np.zeros(len(anchors), dtype=bool)
        for s, e in spans:
            hit |= (anchors[1].to_numpy() < e) & (anchors[2].to_numpy() > s)
        anchors = anchors[hit].reset_index(drop=True)
        anchors[3] = "N"
    length = int(anchors[2].max()) + 10_000_000 if len(anchors) else region.end + 10_000_000

    files: list[str] = []
    names: list[str] = []
    counts: dict[str, int] = {}
    extra: dict[tuple[int, int], str] = {}
    for fac, path in cell.clusters_for(factors):
        clusters = _clusters(_read(path, chrom, 7), anchors)
        if svs:
            # An inserted motif becomes an anchor only above triosformer's score cutoff. At the
            # rules' own 0, repeats inside insertions made dozens of 19 bp anchors per region.
            got = rewrite.apply(clusters, rewrite.domains(clusters), {chrom: length}, svs, None, None, motifs.THRESHOLD)
        else:
            got = [rewrite.Loop(c[0], (c[1], c[2]), (c[4], c[5]), (c[6], c[7]), float(c[8]), (c[1], c[2]), (c[4], c[5])) for c in clusters]
        kept = [lp for lp in got if region.start <= (lp.ref_a or lp.a)[0] and (lp.ref_b or lp.b)[1] <= region.end]
        rows = sorted((lp.chrom, *lp.a, lp.chrom, *lp.b, max(1, int(round(lp.weight)))) for lp in kept)
        name = f"{fac.lower()}_clusters.bedpe"
        pd.DataFrame(rows, columns=range(7)).to_csv(out / name, sep="\t", header=False, index=False)
        files.append(name)
        names.append(ENGINE_FACTOR[fac])
        counts[fac] = len(rows)
        for lp in kept:
            extra[lp.a] = lp.orient[0]
            extra[lp.b] = lp.orient[1]

    inside = anchors[(anchors[1] >= region.start) & (anchors[2] <= region.end)]
    s = L.many(inside[1].to_numpy())
    e = L.many(inside[2].to_numpy() - 1) + 1
    keep = (s >= 0) & (e > 0)
    lifted = {(int(a), int(b)): str(o) for a, b, o in zip(s[keep], e[keep], inside[3].astype(str).to_numpy()[keep], strict=True)}
    for iv, o in extra.items():
        lifted.setdefault((int(iv[0]), int(iv[1])), o)
    pd.DataFrame([(chrom, a, b, o) for (a, b), o in sorted(lifted.items())]).to_csv(out / "anchors.bed", sep="\t", header=False, index=False)

    sing = _read(cell.path("singletons"), chrom, 7)
    sing = sing[sing[3] == chrom]
    if svs:
        sing = engine.lift_bedpe(sing, {chrom: L})
    if hic is not None:
        cond, ref_start = hic
        sing = engine.splice_window(sing, chrom, L.edge(ref_start), cond, hicdiffusion.BIN)
    sing.to_csv(out / "singletons.bedpe", sep="\t", header=False, index=False)
    seg = _read(cell.path("segments"), chrom, 3)
    seg[1] = L.many(seg[1].to_numpy())
    seg[2] = seg[1]
    seg[seg[1] >= 0].to_csv(out / "segments.bed", sep="\t", header=False, index=False)
    cen = _read(cell.path("centromeres"), chrom, 3)
    cen[1], cen[2] = L.many(cen[1].to_numpy()), L.many(cen[2].to_numpy())
    cen.to_csv(out / "centromeres.bed", sep="\t", header=False, index=False)

    data = {
        "data_dir": str(out.resolve()),
        "anchors": "anchors.bed",
        "clusters": ",".join(files),
        "singletons": "singletons.bedpe",
        "singletons_inter": "",
        "segment_split": "segments.bed",
        "centromeres": "centromeres.bed",
    }
    if len(names) > 1 or factors == "RNAPII":
        data["factors"] = ",".join(names)
    config = engine.write_config(cell.path("config"), data, out / "config.ini", header=f"{cell.name} {region} {factors}")
    fixed = {}
    if factors == "RNAPII":
        fixed["motif_orientation.use_motif_orientation"] = "no"
    if background == "none":
        fixed |= {"springs.use_contact_background": "no", "anchor_heatmap.use_anchor_heatmap": "no"}
    if fixed:
        apply_overrides(config, fixed)
    hap_region = f"{chrom}:{L.edge(region.start)}-{L.edge(region.end - 1) + 1}"
    return Inputs(config=config, region=hap_region, n_svs=len(svs), loops=counts)
