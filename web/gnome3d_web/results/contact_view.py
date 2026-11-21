"""The contact maps of a job page.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import contacts as cm
from .. import remote
from ..engine import hicdiffusion
from .views import ExploreError, _json, _loop_files, _positions, models


def _read_deep(url: str, chrom: str, start: int, end: int, res: int, allow_private: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bin pairs of a window of a remote deep map, bin starts and raw counts."""
    kind = remote.check_link(url, allow_private)
    size, final = remote.probe(url, allow_private)
    w = remote.read_window(final, kind, chrom, start, end, size, allow_private, res=res)
    return w.bin1, w.bin2, w.counts


def _deep_map(job: Path, url: str, chrom: str, start: int, end: int, res: int, allow_private: bool) -> np.ndarray:
    """A deep map's window on the job's grid. The pixels of the job's whole region are read once
    per resolution and kept in the job directory. Zooming and panning cut windows from them
    instead of reading the remote file again."""
    status = _json(job / "status.json").get("region") or f"{chrom}:{start}-{end}"
    jr = re.fullmatch(r"(chr\w+):(\d+)-(\d+)", str(status))
    rs, re_ = (int(jr[2]), int(jr[3])) if jr else (start, end)
    cache = job / f"explore_deep_{chrom}_{rs}_{re_}_{res}.npz"
    if cache.is_file():
        z = np.load(cache)
        b1, b2, counts = z["b1"], z["b2"], z["counts"]
    else:
        b1, b2, counts = (np.asarray(x) for x in _read_deep(url, chrom, rs, re_, res, allow_private))
        np.savez(cache, b1=b1, b2=b2, counts=counts)
    edges = cm.bins_for(start, end, res)
    n = len(edges)
    i, j = (b1 - edges[0]) // res, (b2 - edges[0]) // res
    ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
    m = np.zeros((n, n))
    np.add.at(m, (i[ok], j[ok]), counts[ok])
    off = i[ok] != j[ok]
    np.add.at(m, (j[ok][off], i[ok][off]), counts[ok][off])
    return m


def _span(bp: int) -> str:
    """A separation in words, megabases from one up."""
    return f"{bp / 1e6:g} Mb" if bp >= 1_000_000 else f"{bp // 1000} kb"


def contact_maps(job: Path, region: str, res: int, upper: str, lower: str, norm: str, hic_url: str = "", hic_name: str = "", allow_private: bool = False) -> dict[str, object]:
    """Two contact maps of a window on one grid with two agreements of the halves, the rank
    agreement per distance and the stratum adjusted correlation. Each map is symmetric and is
    returned with each pair of bins once, as `_once` writes it.

    @param hic_url a deep map of the cell type in place of the job's own input contacts
    @param hic_name the deep map's name for the labels
    """
    m = re.fullmatch(r"(chr\w+):(\d+)-(\d+)", region.replace(",", ""))
    if not m:
        raise ExploreError(f"{region!r} is not chr:start-end")
    chrom, start, end = m[1], int(m[2]), int(m[3])
    # The window is cut to the job's region: outside it there are no beads to count.
    job_region = _json(job / "status.json").get("region") or ""
    jr = re.fullmatch(r"(chr\w+):(\d+)-(\d+)", str(job_region))
    if jr:
        jc, js, je = jr[1], int(jr[2]), int(jr[3])
        if chrom != jc or end <= js or start >= je:
            raise ExploreError(f"{chrom}:{start:,}-{end:,} lies outside the job's region {jc}:{js:,}-{je:,}.")
        start, end = max(start, js), min(end, je)
    if res not in cm.RESOLUTIONS:
        raise ExploreError(f"resolution {res} is not one of {', '.join(f'{r // 1000} kb' for r in cm.RESOLUTIONS)}")
    if len(cm.bins_for(start, end, res)) > cm.MAX_BINS:
        fit = cm.coarsest_fit(start, end)
        raise ExploreError(f"{region} at {res // 1000} kb is more than {cm.MAX_BINS} bins on a side. Use {fit // 1000} kb or a shorter window.")
    n_models = len(models(job))
    cell = _json(job / "request.json").get("cell_type", "")
    pos, mid = _positions(job)
    bond = cm.median_bond(pos)
    hic = None
    if hic_url:
        try:
            hic, hic_label = _deep_map(job, hic_url, chrom, start, end, res, allow_private), f"Hi-C of {cell}, {hic_name or 'deep map'}"
        except Exception:  # noqa: BLE001
            # A deep map that cannot be read leaves the contacts the job read and the label says so.
            hic = None
    if hic is None:
        sing = pd.read_csv(job / "inputs" / "singletons.bedpe", sep="\t", header=None, usecols=range(7))
        hic = cm.hic_map(sing, chrom, start, end, res)
        hic_label = f"Hi-C of {cell}, the contacts this job read" + (", the deep map could not be read" if hic_url else "")
    counted = cm.model_map(pos, mid, start, end, res, bond)

    loop_sets = {f"loops_{factor.lower()}": (factor, path) for factor, path in _loop_files(job)}
    predicted = job / hicdiffusion.MAP_FILE
    kinds = ["hic", "model", "model_count", *(["hicdiffusion"] if predicted.is_file() else []), *loop_sets]

    def one(kind: str) -> tuple[np.ndarray, str]:
        if kind == "hic":
            return hic, hic_label
        if kind == "hicdiffusion" and predicted.is_file():
            on, first, step, counts = hicdiffusion.read_map(predicted)
            grid = cm.window_map(counts, first, step, start, end, res)
            return (grid if on == chrom else np.zeros_like(grid)), "HiCDiffusion, the contacts predicted from the sequence"
        if kind in loop_sets:
            factor, path = loop_sets[kind]
            rows = []
            for ln in path.read_text().splitlines():
                f = ln.split()
                if len(f) >= 7 and f[0] == chrom and f[3] == chrom:
                    rows.append(((int(f[1]) + int(f[2])) // 2, (int(f[4]) + int(f[5])) // 2, int(float(f[6]))))
            return cm.loop_map(rows, start, end, res), f"{factor} loops, PETs at the anchors"
        if kind == "model":
            return cm.multimm_map(pos, mid, start, end, res), f"This job, 1/(d+1)³ between bins, mean of {n_models} structures"
        if kind == "model_count":
            return counted, f"This job, contacts within the median bond, sum of {n_models} structures"
        raise ExploreError(f"{kind!r} is not one of {', '.join(kinds)}")

    a, la = one(upper)
    b, lb = one(lower)
    raw_a, raw_b = a, b
    if norm == "ice":
        # A loop half is a list of loops, not a contact map and is never balanced.
        a = a if upper in loop_sets else cm.ice(a)
        b = b if lower in loop_sets else cm.ice(b)
    elif norm != "raw":
        raise ExploreError(f"{norm!r} is not ice or raw")
    # The rank agreement reads a Hi-C half balanced and over the bins Hi-C maps. An
    # unmapped bin reads as zero and a bead there would count against the model.
    keep = np.ones(len(raw_a), dtype=bool)
    rk_a, rk_b = raw_a, raw_b
    if upper == "hic":
        keep &= raw_a.sum(axis=1) > 0
        rk_a = cm.ice(raw_a)
    if lower == "hic":
        keep &= raw_b.sum(axis=1) > 0
        rk_b = cm.ice(raw_b)
    return {
        "bins": cm.bins_for(start, end, res).tolist(),
        "upper": _once(a),
        "lower": _once(b),
        "upper_label": la,
        "lower_label": lb,
        # The agreement of the two halves shown, on raw values; Hi-C against the counted model is
        # the validation battery's number.
        "scc": cm.scc(raw_a, raw_b, res),
        "scc_about": (
            f"{hic_label} against this job's contacts within the median bond, raw counts, diagonals up to 2 Mb, as the validation battery computes it"
            if {upper, lower} == {"hic", "model_count"}
            else f"{la} against {lb}, raw values, diagonals up to 2 Mb"
        ),
        "rank": cm.rank_agreement(rk_a, rk_b, res, keep),
        "rank_reach": cm.rank_reach(rk_a, rk_b, res, keep),
        "rank_about": f"{la} against {lb}, each distance up to {_span(cm.rank_reach(rk_a, rk_b, res, keep))} ranked on its own, the distances at which half of the pairs hold a contact in both maps"
        + (", Hi-C balanced, over the bins Hi-C maps" if "hic" in (upper, lower) else ""),
        "loop_enrichment": (
            cm.loop_enrichment(raw_a, raw_b)
            if upper in loop_sets and lower not in loop_sets
            else cm.loop_enrichment(raw_b, raw_a)
            if lower in loop_sets and upper not in loop_sets
            else None
        ),
    }


def _once(m: np.ndarray) -> list[list[float]]:
    """A symmetric map with each pair of bins once: row `i` holds the cells of the columns from
    `i` to the last. Every map of contact_maps is symmetric and its answer is half as long this way."""
    return [np.round(m[i, i:], 6).tolist() for i in range(len(m))]
