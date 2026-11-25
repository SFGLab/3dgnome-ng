"""Run FLAMINGO on the Hi-C map of a region and collect an ensemble for the validation battery.

    python playground/flamingo_arm.py MCOOL REGION OUT_DIR [--n 10] [--rscript Rscript] [--threads 20]

FLAMINGO builds a structure from a Hi-C map and reads no loops. The map here is the half that our
runs read, and the score reads the other half. The fragments are the 5 kb bins of the region and
the domains are 1 Mb, the settings of its example. The matrix of the fragments holds the balanced
counts of the map at 5 kb, scaled to the sum of the raw counts of the region. The half maps have
no resolution of 1 Mb, and the matrix of the domains is the sum of those counts over each pair of
domains.

The run is FLAMINGOrLite, the version its authors maintain, through `playground/flamingo_run.R`.
FLAMINGO gives one structure a run. The ensemble is `--n` runs, each with its own draw of the 75
percent of the entries that it fits. A fragment without contacts has no bead. Every bead is a
subanchor with the span of its bin. The matrices and the tables of the runs stay in `<OUT_DIR>_run`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnome3d.io import write_cif  # noqa: E402
from gnome3d.types import BeadOut  # noqa: E402

FRAG_RES = 5_000
DOMAIN_RES = 1_000_000


def _flag(name: str, default: str) -> str:
    if name not in sys.argv:
        return default
    k = sys.argv.index(name)
    v = sys.argv[k + 1]
    del sys.argv[k : k + 2]
    return v


def write_matrices(mcool: str, chrom: str, lo: int, hi: int, work: Path) -> tuple[int, int, int]:
    """Writes the upper triangles of the two matrices of a region and returns the number of
    fragments, the number of domains and the start of the first fragment."""
    import cooler

    c = cooler.Cooler(f"{mcool}::resolutions/{FRAG_RES}")
    first = lo // FRAG_RES
    last = min(-(-hi // FRAG_RES), -(-int(c.chromsizes[chrom]) // FRAG_RES))
    n = last - first
    per = DOMAIN_RES // FRAG_RES
    n_domain = -(-n // per)
    offset = c.offset(chrom) + first
    step = 2_000

    def slabs():  # noqa: ANN202
        for s in range(0, n, step):
            e = min(n, s + step)
            span = (f"{chrom}:{(first + s) * FRAG_RES}-{(first + e) * FRAG_RES}", f"{chrom}:{(first + s) * FRAG_RES}-{last * FRAG_RES}")
            px = c.matrix(balance=True, as_pixels=True, join=False).fetch(*span)
            yield px[np.isfinite(px["balanced"]) & (px["balanced"] > 0) & (px["bin2_id"] >= px["bin1_id"])]

    # FLAMINGO takes the distance of a pair as its count to the power -0.25 and reads counts
    # normalised on the scale of counts, as KR in a .hic file. The weights of a cooler put a
    # balanced count near 1e-3. The balanced counts are scaled to the sum of the raw counts.
    raw = balanced = 0.0
    for px in slabs():
        raw += float(px["count"].sum())
        balanced += float(px["balanced"].sum())
    scale = raw / balanced
    coarse = np.zeros((n_domain, n_domain))
    kept = 0
    with open(work / "frag.tsv", "w") as fh:
        for px in slabs():
            i = px["bin1_id"].to_numpy() - offset
            j = px["bin2_id"].to_numpy() - offset
            v = px["balanced"].to_numpy() * scale
            np.add.at(coarse, (i // per, j // per), v)
            np.savetxt(fh, np.c_[i + 1, j + 1, v], fmt="%d\t%d\t%.8g")
            kept += len(v)
    ii, jj = np.nonzero(np.triu(coarse))
    np.savetxt(work / "domain.tsv", np.c_[ii + 1, jj + 1, coarse[ii, jj]], fmt="%d\t%d\t%.8g")
    print(
        f"{chrom}:{first * FRAG_RES}-{last * FRAG_RES}: {n} fragments, {n_domain} domains, {kept:,} cells with a balanced count,"
        f" scaled by {scale:.0f} to a mean of {raw / kept:.2f}",
        flush=True,
    )
    return n, n_domain, first * FRAG_RES


def main() -> None:
    n = int(_flag("--n", "10"))
    rscript = _flag("--rscript", "Rscript")
    threads = _flag("--threads", "20")
    mcool, region, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    chrom, span = region.split(":")
    lo, hi = (int(v) for v in span.split("-"))
    work = out.parent / f"{out.name}_run"
    work.mkdir(parents=True, exist_ok=True)
    n_frag, n_domain, start = write_matrices(mcool, chrom, lo, hi, work)
    script = Path(__file__).resolve().parent / "flamingo_run.R"
    cmd = [rscript, str(script), str(work / "frag.tsv"), str(work / "domain.tsv"), str(n_frag), str(n_domain),
           str(FRAG_RES), str(DOMAIN_RES), str(n), threads, str(work / "structure")]  # fmt: skip
    with open(out.parent / f"{out.name}_run.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=work).returncode
    out.mkdir(parents=True, exist_ok=True)
    found = 0
    for k in range(1, n + 1):
        table = work / f"structure_{k}.tsv"
        if not table.is_file():
            continue
        t = np.loadtxt(table, skiprows=1, ndmin=2)
        order = np.argsort(t[:, 0])
        beads = [
            BeadOut(start + (int(f) - 1) * FRAG_RES, start + int(f) * FRAG_RES, float(x), float(y), float(z), "subanchor")
            for f, x, y, z in t[order]
        ]
        write_cif(str(out / f"member_{k}.cif"), beads)
        found += 1
    print(f"FLAMINGO exit {rc}, {found} of {n} structures collected into {out.name}")
    if rc != 0 or found != n:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
