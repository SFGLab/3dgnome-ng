"""The inputs of the engine for one request, written to the directory of the job.

The loops of the cell type on the chromosome are rewired by the variants of the sample with
the 2019 rules of personal3d. The loops wholly inside the region are kept. Anchors, singletons,
segments, centromeres and compartments are lifted to the coordinates of the haplotype. A
HiCDiffusion window is spliced into the singletons when given. The config is the cell type's
with its data section replaced.
"""

from __future__ import annotations

import hashlib

import re

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from personal3d import engine, hicdiffusion, motifs, rewrite
from personal3d.lift import Lift
from personal3d.variants import SV

from ..catalog.model import CellType
from ..density import hosted_density, thin
from ..regions import Region
from .settings import apply_overrides

ENGINE_FACTOR = {"CTCF": "CTCF", "RNAPII": "RNAPOL2"}


def _no_loops() -> list[tuple[str, Path]]:
    return []


@dataclass
class Own:
    """A job's own files: loops per factor and anchors, contacts and enhancers in place of the
    cell type's. Without anchors the loop ends are the anchors."""

    loops: list[tuple[str, Path]] = field(default_factory=_no_loops)
    anchors: Path | None = None
    contacts: Path | None = None
    enhancers: Path | None = None


def _loop_end_anchors(paths: list[Path], chrom: str) -> pd.DataFrame:
    """The ends of the loops on a chromosome, overlapping ends merged, orientation N."""
    spans = []
    for p in paths:
        t = _read(p, chrom, 7)
        spans += list(zip(t[1], t[2], strict=True)) + list(zip(t[4], t[5], strict=True))
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], int(b))
        else:
            merged.append([int(a), int(b)])
    return pd.DataFrame([(chrom, a, b, "N") for a, b in merged], columns=range(4))


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


def _thin_to_hosted(sing: pd.DataFrame, cell: CellType, chrom: str) -> pd.DataFrame:
    """Own contacts thinned to the contacts per Mb of the cell type's hosted map on the
    chromosome, over the span the own contacts cover. Every route of own contacts meets here,
    so a deep map and a shallow one give the engine the density its maps are read at."""
    span = (int(sing[5].max()) - int(sing[1].min())) / 1e6
    # At least one contact, or a short span would be thinned to nothing.
    target = max(1, round(hosted_density(cell.path("singletons"), chrom) * span))
    seed = int.from_bytes(hashlib.blake2b(f"{chrom} {len(sing)} {int(sing[6].sum())}".encode(), digest_size=8).digest(), "little")
    out = sing.copy()
    out[6] = thin(out[6].to_numpy(dtype=np.int64), target, seed)
    return out[out[6] > 0].reset_index(drop=True)


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


def _snap(loops: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Own loops with each end replaced by the anchor holding its midpoint, when one does."""
    starts = anchors[1].to_numpy()
    ends = anchors[2].to_numpy()
    out = loops.copy()
    for s_col, e_col in ((1, 2), (4, 5)):
        mids = ((out[s_col] + out[e_col]) // 2).to_numpy()
        k = np.searchsorted(starts, mids, side="right") - 1
        hit = (k >= 0) & (ends[np.clip(k, 0, None)] >= mids)
        out.loc[hit, s_col] = starts[k[hit]]
        out.loc[hit, e_col] = ends[k[hit]]
    return out


def loop_network(
    cell: CellType, region: Region, factors: str, own_loops: list[tuple[str, Path]], own_anchors: Path | None
) -> tuple[pd.DataFrame, int, list[tuple[str, list[list]]]]:
    """The anchors of the region's chromosome, the length the rules give that chromosome and the
    loops of each factor on those anchors. Own loops replace the loops of the cell type and each
    of their ends is moved to the anchor that holds its midpoint. RNAPII alone keeps the anchors
    that hold an end of an RNAPII loop, without their orientation. The job and the preview of the
    variants both rewire these loops."""
    chrom = region.chrom
    if own_loops:
        anchors = _read(own_anchors, chrom, 4) if own_anchors else _loop_end_anchors([p for _, p in own_loops], chrom)
        if anchors.shape[1] < 4:
            anchors[3] = "N"
        anchors = anchors.sort_values(1).reset_index(drop=True)
    else:
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
    network = []
    for factor, path in own_loops or cell.clusters_for(factors):
        raw = _read(path, chrom, 7)
        network.append((factor, _clusters(_snap(raw, anchors) if own_loops else raw, anchors)))
    return anchors, length, network


def build_inputs(
    cell: CellType,
    region: Region,
    factors: str,
    svs: list[SV],
    out: Path,
    hic: tuple[np.ndarray, int] | None = None,
    background: str = "hic",
    own: Own | None = None,
) -> Inputs:
    """Write the inputs for one haplotype under `out` and return the config and the region in
    the haplotype's coordinates. `hic` is a predicted HiCDiffusion window and its reference start.
    `background` none turns off the contact terms and keeps the singletons for the exponent fit.
    RNAPII alone has its loop ends as anchors, orientation N and no CTCF orientation term."""
    out.mkdir(parents=True, exist_ok=True)
    chrom = region.chrom
    L = Lift(svs)
    anchors, length, network = loop_network(cell, region, factors, own.loops if own else [], own.anchors if own else None)

    files: list[str] = []
    names: list[str] = []
    counts: dict[str, int] = {}
    extra: dict[tuple[int, int], str] = {}
    for fac, clusters in network:
        if svs:
            # An inserted motif becomes an anchor only above triosformer's score cutoff. At the
            # rules' own 0, repeats inside insertions made dozens of 19 bp anchors per region.
            got = rewrite.apply(clusters, rewrite.domains(clusters), {chrom: length}, svs, None, None, motifs.THRESHOLD)
        else:
            got = [rewrite.Loop(c[0], (c[1], c[2]), (c[4], c[5]), (c[6], c[7]), float(c[8]), (c[1], c[2]), (c[4], c[5])) for c in clusters]
        kept = [lp for lp in got if region.start <= (lp.ref_a or lp.a)[0] and (lp.ref_b or lp.b)[1] <= region.end]
        rows = sorted((lp.chrom, *lp.a, lp.chrom, *lp.b, max(1, int(round(lp.weight)))) for lp in kept)
        name = f"{re.sub(r'[^a-z0-9]', '', fac.lower()) or 'loops'}_clusters.bedpe"
        pd.DataFrame(rows, columns=range(7)).to_csv(out / name, sep="\t", header=False, index=False)
        files.append(name)
        names.append(ENGINE_FACTOR.get(fac, fac.upper()))
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

    sing = _read(own.contacts if own and own.contacts else cell.path("singletons"), chrom, 7)
    sing = sing[sing[3] == chrom]
    if own and own.contacts and len(sing):
        sing = _thin_to_hosted(sing, cell, chrom)
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
    # The compartments are the cell type's own and are not read with own contacts. An edge inside a
    # deletion goes where the deletion joins its two sides. Neighbours stay joined.
    compartments = "compartments" in cell.files and not (own and own.contacts)
    if compartments:
        comp = _read(cell.path("compartments"), chrom, 4)
        ends = [(L.edge(int(a)), L.edge(int(b))) for a, b in zip(comp[1], comp[2], strict=True)]
        comp[1], comp[2] = [min(e) for e in ends], [max(e) for e in ends]
        comp[comp[2] > comp[1]].to_csv(out / "compartments.bed", sep="\t", header=False, index=False)

    data = {
        "data_dir": str(out.resolve()),
        "anchors": "anchors.bed",
        "clusters": ",".join(files),
        "singletons": "singletons.bedpe",
        "singletons_inter": "",
        "segment_split": "segments.bed",
        "centromeres": "centromeres.bed",
    }
    if compartments:
        data["compartments"] = "compartments.bed"
    if len(names) > 1 or factors == "RNAPII" or (own and own.loops):
        data["factors"] = ",".join(names)
    # The beads of a stretch without loops are drawn against the cell type's map, which is in
    # reference coordinates and is this cell's. It is not used with variants or own contacts.
    if cell.contact_map and not svs and not (own and own.contacts):
        data["contact_map"] = cell.contact_map
    config = engine.write_config(cell.path("config"), data, out / "config.ini", header=f"{cell.name} {region} {factors}")
    fixed = {}
    if factors == "RNAPII" or (own and own.loops and own.loops[0][0].upper() != "CTCF"):
        fixed["motif_orientation.use_motif_orientation"] = "no"
    if background == "none":
        fixed |= {"springs.use_contact_background": "no", "anchor_heatmap.use_anchor_heatmap": "no"}
    if fixed:
        apply_overrides(config, fixed)
    hap_region = f"{chrom}:{L.edge(region.start)}-{L.edge(region.end - 1) + 1}"
    return Inputs(config=config, region=hap_region, n_svs=len(svs), loops=counts)
