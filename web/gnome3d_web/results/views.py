"""The answers of a job page from a job directory: its summary, structures, annotations, ensemble
picks, distances, gene statistics, loops and files.

Positions are in the job's own coordinates, those of its haplotype when it has variants.
Results that read every structure are cached as explore_<name>.json in the job directory.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gnome3d.geometry import smooth_chain

from .. import expression
from ..jobs import store


MAX_GAP = 5_000
PER_BOND = 4
_FACTOR = {"ctcf": "CTCF", "rnapii": "RNAPII"}


class ExploreError(ValueError):
    """A request the job's data cannot answer, with the reason."""


def cached(job: Path, name: str, make):  # noqa: ANN001, ANN202
    path = job / f"explore_{name}.json"
    if path.is_file():
        return json.loads(path.read_text())
    value = make()
    if store.status(job).get("state") == "done":
        path.write_text(json.dumps(value))
    return value


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
    status = store.status(job)
    files = store.sizes(p for p in sorted(job.iterdir()) if not p.name.startswith(("explore_", "upload.")))
    ep = job / "ep.tsv"
    return {
        "state": status.get("state"),
        "stage": status.get("stage"),
        "error": store.public_log(str(status["error"]), job) if status.get("error") else None,
        "log_tail": store.public_log(status["log_tail"], job) if status.get("log_tail") else None,
        "ahead": ahead,
        "started": status.get("started"),
        "finished": status.get("finished"),
        "region": status.get("region"),
        "request": store.public_request(store.request(job)),
        "loops": status.get("loops", {}),
        "n_svs": status.get("n_svs"),
        "structures": len(store.structures(job)),
        "genes": max(0, len(ep.read_text().splitlines()) - 1) if ep.is_file() else 0,
        "loop_count": sum(status.get("loops", {}).values()),
        "files": files,
        "models": store.sizes(store.structures(job)),
        "expression": _public_expression(job),
        "stages": status.get("stages", []),
        # The clock the stages are timed on. A page counts the time of the running stage from it.
        "now": time.time(),
        "structures_done": len(list((job / "models").glob("*.cif"))) if (job / "models").is_dir() else 0,
        "structures_asked": store.request(job).get("n_models"),
        "log_recent": store.public_log(store.log_end(job / "engine.log", 200), job) if status.get("state") in ("running", "failed") else "",
    }


def loop_files(job: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted((job / "inputs").glob("*_clusters.bedpe")):
        out.append((_FACTOR.get(p.name.split("_")[0], p.name.split("_")[0].upper()), p))
    return out


def loops(job: Path) -> list[dict[str, object]]:
    """The loops the engine read, with factor, ends, span and PET count."""
    rows = []
    for factor, path in loop_files(job):
        for ln in path.read_text().splitlines():
            f = ln.split()
            if len(f) < 7:
                continue
            a, b = [int(f[1]), int(f[2])], [int(f[4]), int(f[5])]
            rows.append({"factor": factor, "chrom": f[0], "a": a, "b": b, "span": b[0] - a[1], "pets": int(f[6])})
    return rows


def annotations(job: Path) -> dict[str, object]:
    """The genes, enhancers and loops of the job's region."""
    elements = store.read_json(job / "marks.json").get("elements", [])
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
    files = store.structures(job)
    if not 1 <= i <= len(files):
        raise ExploreError(f"structure {i} of {len(files)} does not exist")
    pos, start, end, anchor = store.read_cif(files[i - 1])
    mid = (start + end) // 2
    chain = _chain(pos, mid)
    marks = store.read_json(job / "marks.json")
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


def positions(job: Path) -> tuple[list[np.ndarray], np.ndarray]:
    loaded = [store.read_cif(m) for m in store.structures(job)]
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
        pos, _ = positions(job)
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

    return cached(job, "picks", make)


def nearest(mid: np.ndarray, bp: int) -> tuple[int, int]:
    k = int(np.argmin(np.abs(mid - bp)))
    return k, int(mid[k]) - bp


def distance(job: Path, a: int, b: int) -> dict[str, object]:
    """The 3D distance between the beads nearest two positions in every structure, with the
    mean and the standard deviation, 0 for one structure."""
    pos, mid = positions(job)
    out = {}
    for name, bp in (("a", a), ("b", b)):
        k, off = nearest(mid, bp)
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
        pos, mid = positions(job)
        _, start, end, _ = store.read_cif(store.structures(job)[0])
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

    return cached(job, "genes_v3", make)
