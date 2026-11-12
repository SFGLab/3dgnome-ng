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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from gnome3d.geometry import smooth_chain

from . import contacts as cm
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


def summary(job: Path, ahead: int | None) -> dict[str, object]:
    """The state, the request, the counts and the files of a job."""
    status = _json(job / "status.json") or {"state": "queued"}
    files = [{"name": p.name, "size": p.stat().st_size} for p in sorted(job.iterdir()) if p.is_file() and not p.name.startswith(("explore_", "upload."))]
    ep = job / "ep.tsv"
    return {
        "state": status.get("state"),
        "stage": status.get("stage"),
        "error": status.get("error"),
        "log_tail": status.get("log_tail"),
        "ahead": ahead,
        "started": status.get("started"),
        "finished": status.get("finished"),
        "region": status.get("region"),
        "request": {k: v for k, v in _json(job / "request.json").items() if k != "upload_vcf"},
        "loops": status.get("loops", {}),
        "n_svs": status.get("n_svs"),
        "structures": len(models(job)),
        "genes": max(0, len(ep.read_text().splitlines()) - 1) if ep.is_file() else 0,
        "loop_count": sum(status.get("loops", {}).values()),
        "files": files,
        "models": [{"name": m.name, "size": m.stat().st_size} for m in models(job) if m.is_file()],
    }


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
                }
            )
        return rows

    return _cached(job, "genes", make)


def log_text(job: Path) -> str:
    """The engine log."""
    path = job / "engine.log"
    return path.read_text(errors="replace") if path.is_file() else ""


def zip_bytes(job: Path) -> bytes:
    """Every file of the job but the upload, its structures under models/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(job.rglob("*")):
            rel = p.relative_to(job)
            if p.is_file() and not rel.name.startswith(("explore_", "upload.")) and not re.match(r"inputs/", str(rel)):
                z.write(p, str(rel))
    return buf.getvalue()


def contact_maps(job: Path, region: str, res: int, upper: str, lower: str, norm: str) -> dict[str, object]:
    """Two contact maps of a window on one grid, each the job's Hi-C or its model, raw or ICE
    balanced, and their stratum adjusted correlation."""
    m = re.fullmatch(r"(chr\w+):(\d+)-(\d+)", region.replace(",", ""))
    if not m:
        raise ExploreError(f"{region!r} is not chr:start-end")
    chrom, start, end = m[1], int(m[2]), int(m[3])
    if res not in cm.RESOLUTIONS:
        raise ExploreError(f"resolution {res} is not one of {', '.join(f'{r // 1000} kb' for r in cm.RESOLUTIONS)}")
    if len(cm.bins_for(start, end, res)) > cm.MAX_BINS:
        fit = cm.coarsest_fit(start, end)
        raise ExploreError(f"{region} at {res // 1000} kb is more than {cm.MAX_BINS} bins on a side. Use {fit // 1000} kb or a shorter window.")
    n_models = len(models(job))
    cell = _json(job / "request.json").get("cell_type", "")

    def one(kind: str) -> tuple[np.ndarray, str]:
        if kind == "hic":
            sing = pd.read_csv(job / "inputs" / "singletons.bedpe", sep="\t", header=None, usecols=range(7))
            return cm.hic_map(sing, chrom, start, end, res), f"Hi-C of {cell}"
        if kind == "model":
            pos, mid = _positions(job)
            return cm.model_map(pos, mid, start, end, res), f"This job, sum of {n_models} structures"
        raise ExploreError(f"{kind!r} is not hic or model")

    a, la = one(upper)
    b, lb = one(lower)
    if norm == "ice":
        a, b = cm.ice(a), cm.ice(b)
    elif norm != "raw":
        raise ExploreError(f"{norm!r} is not ice or raw")
    return {
        "bins": cm.bins_for(start, end, res).tolist(),
        "upper": np.round(a, 4).tolist(),
        "lower": np.round(b, 4).tolist(),
        "upper_label": la,
        "lower_label": lb,
        "scc": cm.scc(a, b, res),
    }


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
