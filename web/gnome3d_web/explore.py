"""What the job explorer reads from a job directory: its summary, structures, annotations,
ensemble picks, distances, gene statistics and loops.

Positions are in the job's own coordinates, those of its haplotype when it has variants.
Results that read every structure are cached as explore_<name>.json in the job directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from gnome3d.geometry import smooth_chain

from . import contacts as cm
from . import expression, hicdiffusion
from .marks import read_cif

MAX_GAP = 5_000
PER_BOND = 4
_FACTOR = {"ctcf": "CTCF", "rnapii": "RNAPII"}


class ExploreError(ValueError):
    """A request the job's data cannot answer, with the reason."""


def _json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _cached(job: Path, name: str, make):  # noqa: ANN001, ANN202
    path = job / f"explore_{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    value = make()
    if _json(job / "status.json").get("state") == "done":
        path.write_text(json.dumps(value))
    return value


def models(job: Path) -> list[Path]:
    """The job's structure files in member order."""
    names = _json(job / "marks.json").get("models", [])
    return [job / "models" / n for n in names]


def _public_request(req: dict) -> dict:
    """The request without the server's paths of uploaded files."""
    out = {k: v for k, v in req.items() if k not in ("upload_vcf", "own", "expression")}
    # The source of the expression by its id. The table of an upload has a path on the server.
    if req.get("expression"):
        out["expression"] = str(req["expression"].get("source", ""))
    own = req.get("own") or {}
    if own:
        out["own"] = {
            "loops": [f for f, _ in own.get("loops", [])],
            **{k: bool(own.get(k)) for k in ("anchors", "contacts", "enhancers")},
            **{k: own[k] for k in ("loops_from", "enhancers_from") if own.get(k)},
        }
    return out


def _public_expression(job: Path) -> dict[str, object] | None:
    """The expression of a job as its page states it: the label, the kind, the unit of the gene
    table and the number of samples. None for a job without expression."""
    meta = expression.job_meta(job)
    if meta is None:
        return None
    levels = expression.job_levels(job)
    return {"label": meta.get("label", ""), "kind": meta["kind"], "unit": levels[1] if levels else "", "samples": len(meta["columns"]), "source": meta.get("source", "")}  # type: ignore[arg-type]


def summary(job: Path, ahead: int | None) -> dict[str, object]:
    """The state, the request, the counts and the files of a job."""
    status = _json(job / "status.json") or {"state": "queued"}
    files = [{"name": p.name, "size": p.stat().st_size} for p in sorted(job.iterdir()) if p.is_file() and not p.name.startswith(("explore_", "upload."))]
    ep = job / "ep.tsv"
    return {
        "state": status.get("state"),
        "stage": status.get("stage"),
        "error": status.get("error"),
        "log_tail": _public_log(status["log_tail"], job) if status.get("log_tail") else None,
        "ahead": ahead,
        "started": status.get("started"),
        "finished": status.get("finished"),
        "region": status.get("region"),
        "request": _public_request(_json(job / "request.json")),
        "loops": status.get("loops", {}),
        "n_svs": status.get("n_svs"),
        "structures": len(models(job)),
        "genes": max(0, len(ep.read_text().splitlines()) - 1) if ep.is_file() else 0,
        "loop_count": sum(status.get("loops", {}).values()),
        "files": files,
        "models": [{"name": m.name, "size": m.stat().st_size} for m in models(job) if m.is_file()],
        "expression": _public_expression(job),
        "stages": status.get("stages", []),
        # The clock the stages are timed on. A page counts the time of the running stage from it.
        "now": time.time(),
        "structures_done": len(list((job / "models").glob("*.cif"))) if (job / "models").is_dir() else 0,
        "structures_asked": _json(job / "request.json").get("n_models"),
        "log_recent": _public_log(_log_tail(job / "engine.log", 200), job) if status.get("state") in ("running", "failed") else "",
    }


def _log_tail(path: Path, n: int) -> str:
    """The last n lines of a log, read from its end so a long log costs little."""
    if not path.is_file():
        return ""
    with path.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 32_768))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])


def _loop_files(job: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted((job / "inputs").glob("*_clusters.bedpe")):
        out.append((_FACTOR.get(p.name.split("_")[0], p.name.split("_")[0].upper()), p))
    return out


def loops(job: Path) -> list[dict[str, object]]:
    """The loops the engine read, with factor, ends, span and PET count."""
    rows = []
    for factor, path in _loop_files(job):
        for ln in path.read_text().splitlines():
            f = ln.split()
            if len(f) < 7:
                continue
            a, b = [int(f[1]), int(f[2])], [int(f[4]), int(f[5])]
            rows.append({"factor": factor, "chrom": f[0], "a": a, "b": b, "span": b[0] - a[1], "pets": int(f[6])})
    return rows


def annotations(job: Path) -> dict[str, object]:
    """The genes, enhancers and loops of the job's region."""
    elements = _json(job / "marks.json").get("elements", [])
    genes = [{"name": e["name"], "tss": (e["start"] + e["end"]) // 2} for e in elements if e["kind"] == "promoter"]
    enh = [[e["start"], e["end"]] for e in elements if e["kind"] == "enhancer"]
    return {"genes": genes, "enhancers": enh, "loops": loops(job)}


def _chain(pos: np.ndarray, mid: np.ndarray) -> np.ndarray:
    curve = smooth_chain(pos, PER_BOND)
    if len(curve) == len(pos):
        bp = mid.astype(np.float64)
    else:
        t = np.arange(PER_BOND) / PER_BOND
        bp = np.concatenate([(mid[:-1, None] + t[None, :] * np.diff(mid)[:, None]).ravel(), [float(mid[-1])]])
    return np.column_stack([curve, bp])


def structure(job: Path, i: int) -> dict[str, object]:
    """One structure, i from 1: the smoothed chain with a genomic position per point, the beads,
    the marked beads and the anchors with their loops."""
    files = models(job)
    if not 1 <= i <= len(files):
        raise ExploreError(f"structure {i} of {len(files)} does not exist")
    pos, start, end, anchor = read_cif(files[i - 1])
    mid = (start + end) // 2
    chain = _chain(pos, mid)
    marks = _json(job / "marks.json")
    names = {e["name"]: (e["start"] + e["end"]) // 2 for e in marks.get("elements", []) if e["kind"] == "promoter"}
    by_bead: dict[int, list[str]] = {}
    for gid, b in marks.get("promoters", {}).items():
        by_bead.setdefault(int(b), []).append(gid)
    promoters = []
    for b in sorted(by_bead):
        near = [n for n, t in names.items() if abs(t - int(mid[b])) <= MAX_GAP]
        promoters.append({"bead": b, "genes": near})
    loop_rows = loops(job)
    anchors = []
    for k in np.flatnonzero(anchor):
        s, e = int(start[k]), int(end[k])
        hits = []
        for lp in loop_rows:
            for end_here, other in ((lp["a"], lp["b"]), (lp["b"], lp["a"])):
                if end_here[0] < e and end_here[1] > s:  # type: ignore[index]
                    hits.append({"factor": lp["factor"], "other": other, "pets": lp["pets"]})
        anchors.append({"bead": int(k), "start": s, "end": e, "loops": hits})
    return {
        "chain": np.round(chain, 3).tolist(),
        "beads": [[int(s), int(e), round(float(x), 3), round(float(y), 3), round(float(z), 3), "anchor" if a else "subanchor"] for s, e, (x, y, z), a in zip(start, end, pos, anchor, strict=True)],
        "marks": {"promoters": promoters, "enhancers": sorted(int(b) for b in marks.get("enhancers", []))},
        "anchors": anchors,
    }


def _positions(job: Path) -> tuple[list[np.ndarray], np.ndarray]:
    loaded = [read_cif(m) for m in models(job)]
    mid = (loaded[0][1] + loaded[0][2]) // 2
    return [p for p, *_ in loaded], mid


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    s[-1] *= d
    e = (a**2).sum() + (b**2).sum() - 2 * s.sum()
    return float(np.sqrt(max(e, 0.0) / len(a)))


def picks(job: Path) -> dict[str, int]:
    """The medoid by RMSD after superposition, the most compact by radius of gyration and a
    fixed random pick, as structure numbers from 1."""

    def make() -> dict[str, int]:
        pos, _ = _positions(job)
        n = len(pos)
        rg = [float(np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean())) for p in pos]
        if n > 1:
            d = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d[i, j] = d[j, i] = _kabsch_rmsd(pos[i], pos[j])
            medoid = int(np.argmin(d.sum(axis=1)))
        else:
            medoid = 0
        rnd = int(hashlib.sha256(job.name.encode()).hexdigest(), 16) % n
        return {"medoid": medoid + 1, "compact": int(np.argmin(rg)) + 1, "random": rnd + 1}

    return _cached(job, "picks", make)


def _nearest(mid: np.ndarray, bp: int) -> tuple[int, int]:
    k = int(np.argmin(np.abs(mid - bp)))
    return k, int(mid[k]) - bp


def distance(job: Path, a: int, b: int) -> dict[str, object]:
    """The 3D distance between the beads nearest two positions in every structure, with the
    mean and the standard deviation, 0 for one structure."""
    pos, mid = _positions(job)
    out = {}
    for name, bp in (("a", a), ("b", b)):
        k, off = _nearest(mid, bp)
        if abs(off) > MAX_GAP:
            raise ExploreError(f"position {bp:,} is {abs(off) / 1000:.1f} kb from the nearest bead, more than {MAX_GAP // 1000} kb")
        out[name] = {"bead": k, "bp": int(mid[k]), "offset": off}
    ka, kb = out["a"]["bead"], out["b"]["bead"]  # type: ignore[index]
    values = [float(np.linalg.norm(p[ka] - p[kb])) for p in pos]
    v = np.array(values)
    return out | {"values": values, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def genes(job: Path) -> list[dict[str, object]]:
    """The enhancer to promoter table with statistics over the structures and the loop factors
    whose ends hold the promoter and the enhancer bead."""

    def make() -> list[dict[str, object]]:
        ep = job / "ep.tsv"
        if not ep.is_file():
            return []
        table = pd.read_csv(ep, sep="\t")
        if table.empty:
            return []
        pos, mid = _positions(job)
        _, start, end, _ = read_cif(models(job)[0])
        loop_rows = loops(job)
        levels = expression.job_levels(job)

        def level(r) -> float | None:  # noqa: ANN001
            """The expression of the gene of a row. The second copy of a duplicated gene has the id of the gene."""
            if levels is None:
                return None
            name = str(r.gene_id).removesuffix(".copy2").split(".")[0] if levels[2] == "gene_id" else str(r.name)
            v = levels[0].get(name)
            return None if v is None else round(v, 3)

        def holds(end_: list[int], bead: int) -> bool:
            return end_[0] < int(end[bead]) and end_[1] > int(start[bead])

        rows = []
        for r in table.itertuples(index=False):
            pb, eb = int(r.promoter_bead), int(r.enhancer_bead)
            d = np.array([float(np.linalg.norm(p[pb] - p[eb])) for p in pos])
            q = np.percentile(d, [0, 25, 50, 75, 100])
            factors = sorted(
                {lp["factor"] for lp in loop_rows if (holds(lp["a"], pb) and holds(lp["b"], eb)) or (holds(lp["a"], eb) and holds(lp["b"], pb))}  # type: ignore[arg-type]
            )
            rows.append(
                {
                    "gene": str(r.name),
                    "gene_id": str(r.gene_id),
                    "expression": level(r),
                    "tss": int(r.tss),
                    "enhancer": int(r.enhancer_centre),
                    "linear_bp": int(r.linear_bp),
                    "distance": float(r.distance),
                    "promoter_bead": pb,
                    "enhancer_bead": eb,
                    "min": float(q[0]),
                    "q1": float(q[1]),
                    "median": float(q[2]),
                    "q3": float(q[3]),
                    "max": float(q[4]),
                    "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                    "n": len(d),
                    "joined_by": " and ".join(factors),
                    # A promoter can anchor loops that reach elsewhere; the table says so.
                    "promoter_loops": sum(1 for lp in loop_rows if holds(lp["a"], pb) or holds(lp["b"], pb)),  # type: ignore[arg-type]
                }
            )
        return rows

    return _cached(job, "genes_v3", make)


def _public_log(text: str, job: Path) -> str:
    """A log with the job's directory on the server cut from every path. A private job's
    location on disk is not shown."""
    for p in {str(job.resolve()), str(job)}:
        text = text.replace(p + "/", "").replace(p, ".")
    return text


def log_text(job: Path) -> str:
    """The engine log."""
    path = job / "engine.log"
    return _public_log(path.read_text(errors="replace"), job) if path.is_file() else ""


def zip_bytes(job: Path) -> bytes:
    """Every file of the job but the upload, its structures under models/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(job.rglob("*")):
            rel = p.relative_to(job)
            if p.is_file() and not rel.name.startswith(("explore_", "upload.")) and not re.match(r"inputs/", str(rel)):
                z.write(p, str(rel))
    return buf.getvalue()


def _read_deep(url: str, chrom: str, start: int, end: int, res: int, allow_private: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bin pairs of a window of a remote deep map, bin starts and raw counts."""
    from . import remote

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


def compare(a: Path, b: Path) -> dict[str, object]:
    """The genes both jobs have, with each job's mean and SD of the enhancer to promoter distance
    and the ratio b over a, the largest changes first, at most 20."""
    ra, rb = _json(a / "status.json").get("region"), _json(b / "status.json").get("region")
    ga = {r["gene"]: r for r in genes(a)}
    gb = {r["gene"]: r for r in genes(b)}
    rows = []
    for name in ga.keys() & gb.keys():
        x, y = ga[name], gb[name]
        if x["distance"] <= 0:
            continue
        rows.append(
            {
                "gene": name,
                "linear_bp": x["linear_bp"],
                "a": {"mean": x["distance"], "sd": x["sd"]},
                "b": {"mean": y["distance"], "sd": y["sd"]},
                "ratio": y["distance"] / x["distance"],
            }
        )
    rows.sort(key=lambda r: max(r["ratio"], 1 / r["ratio"]), reverse=True)
    return {"same_region": ra is not None and ra == rb, "rows": rows[:20]}


def compare_expression(a: Path, b: Path, cache: Path) -> dict[str, object]:
    """The expression of job b against job a for the genes both jobs have, beside the change of
    their enhancer to promoter distance.

    Each row has the gene, log2 of the distance of b over a, the log2 fold change of the
    expression, its adjusted p or None and the two levels. `rho` is the Spearman correlation of
    the two changes over the rows, None under five rows. The mode and its reason are those of
    `expression.contrast`.
    """
    c = expression.contrast(a, b, cache)
    table = c.pop("genes", None)
    if not table:
        return {**c, "rows": [], "rho": None}
    by_id = c.get("key") == "gene_id"
    ga = {r["gene"]: r for r in genes(a)}
    gb = {r["gene"]: r for r in genes(b)}
    rows = []
    for name in ga.keys() & gb.keys():
        x, y = ga[name], gb[name]
        e = table.get(str(x["gene_id"]).removesuffix(".copy2").split(".")[0] if by_id else name)
        if e is None or x["distance"] <= 0 or y["distance"] <= 0:
            continue
        rows.append({"gene": name, "distance": round(float(np.log2(y["distance"] / x["distance"])), 4), "log2fc": e[0], "padj": e[1], "a": e[2], "b": e[3]})
    rows.sort(key=lambda r: abs(r["log2fc"]), reverse=True)
    rho = None
    if len(rows) >= 5:
        d = pd.DataFrame(rows)
        rho = float(d["distance"].corr(d["log2fc"], method="spearman"))
        rho = None if not np.isfinite(rho) else round(rho, 3)
    return {**c, "rows": rows, "rho": rho}
