"""Derive the epigenomic tracks the compartment and accessibility terms read.

Two transforms, both writing the plain-text formats `gnome3d.io` parses, so the
engine never grows a cooler or bigWig dependency.

  * mcool -> phased compartment eigenvector bedGraph, via `cooltools.eigs_cis`
  * bigWig -> binned signal bedGraph, via pyBigWig

Both are idempotent. An output is rebuilt only when it is missing, when its input
is newer, or when --force is given.

    python -m validation tracks --cell GM12878
    python -m validation tracks --cell GM12878 --resolution 100000
    python -m validation tracks --cell GM12878 --chroms chr1,chr2

Outputs land next to that cell line's other engine inputs, using the same naming
convention `cell_data_section` follows:

    data/<CELL>/<CELL>_compartments.bedGraph
    data/<CELL>/<CELL>_atac.bedGraph

A lockfile records resolution, phasing decision and input hash per output, so a
rebuild is reproducible and an accidental re-derivation at a different resolution
is visible.

"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np

from validation.studies import Context, register

# Compartment calling is a megabase-scale question, so the default bin is coarse.
# Finer bins make the eigenvector noisy without adding compartment detail.
DEFAULT_RESOLUTION = 100_000

# ATAC varies bead to bead at subanchor scale, which is the scale HiP-HoP works at.
DEFAULT_SIGNAL_RESOLUTION = 5_000


def _hash_head(path: Path, n_bytes: int = 1 << 20) -> str:
    """Cheap identity for a large input. First megabyte plus size."""
    h = hashlib.sha256()
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def _load_lock(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path) as f:
            obj: dict[str, Any] = json.load(f)
            return obj
    return {}


def _save_lock(path: Path, lock: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
        f.write("\n")


def _up_to_date(out: Path, key: str, entry: dict[str, Any] | None, force: bool) -> bool:
    return not force and out.exists() and entry is not None and entry.get("input") == key


def _cache_key(source: Path, resolution: int, chroms: list[str] | None) -> str:
    """What a cached track is keyed on.

    The chromosome selection belongs in the key.  Without it, widening `--chroms`
    reads as up to date against a track built for one chromosome, and every
    downstream region outside it silently scores as unassigned rather than failing.
    """
    return f"{_hash_head(source)}@{resolution}@{','.join(chroms) if chroms else 'all'}"


def _write_bedgraph(path: Path, rows: list[tuple[str, int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for chr_, s, e, v in rows:
            f.write(f"{chr_}\t{s}\t{e}\t{v:.6g}\n")


# --- compartments from an mcool ---------------------------------------------


def _ensure_balanced(uri: str) -> Any:
    """Open a cooler, computing and storing ICE weights when it has none.

    `eigs_cis` needs a balanced matrix and refuses to run without a weight column.
    4DN mcools are inconsistent about shipping one: the GM12878 file has `weight`,
    the H1ESC and HFFc6 files carry only chrom/start/end. Balancing is what
    `cooler balance` does, it is deterministic, and storing it in the file makes
    the result available to every later cooltools call rather than recomputing.
    Idempotent: a file that already has weights is opened and returned unchanged.
    """
    import cooler  # noqa: PLC0415

    clr = cooler.Cooler(uri)
    if "weight" in clr.bins().columns:
        return clr

    print(f"[tracks]   no balancing weights in {uri.rsplit('/', 1)[-1]}, computing (one time)")
    bias, stats = cooler.balance_cooler(clr, store=True, store_name="weight")
    print(
        f"[tracks]   balanced: {int(np.isfinite(bias).sum())} usable bins, "
        f"converged={bool(stats.get('converged', False))}"
    )
    return cooler.Cooler(uri)


def derive_compartments(
    mcool: Path, out: Path, resolution: int, chroms: list[str] | None
) -> dict[str, Any]:
    """mcool -> compartment eigenvector bedGraph, one chromosome at a time.

    Uses `cis_eig` per chromosome rather than `eigs_cis` over the whole cooler, for
    three reasons. It is the same function `validation.metrics.hic` scores
    structures with, so derivation and scoring cannot drift apart. It lets us pass
    `clip_percentile=99.9` explicitly, which is not that function's default and
    without which outlier pixels dominate the decomposition. And it lets us
    symmetrize each matrix first: `eigs_cis` rejects a freshly balanced 4DN cooler
    outright because float round-off leaves the matrix a hair asymmetric.

    The sign is arbitrary and is left as produced. `gnome3d.tracks.phase_compartments`
    orients it at load time against accessibility or loop-anchor counts, which is
    the only evidence available without a genome fasta.
    """
    from cooltools.api import eigdecomp  # noqa: PLC0415

    uri = f"{mcool}::resolutions/{resolution}"
    clr = _ensure_balanced(uri)
    avail = list(clr.chromnames)
    want = [c for c in (chroms or avail) if c in avail]
    if not want:
        raise SystemExit(f"none of the requested chromosomes are in {mcool}: have {avail[:5]}...")

    rows: list[tuple[str, int, int, float]] = []
    quality: dict[str, Any] = {}
    per_chr: dict[str, int] = {}

    for c in want:
        a = np.asarray(clr.matrix(balance=True).fetch(c), dtype=np.float64)
        starts = clr.bins().fetch(c)["start"].to_numpy()
        ends = clr.bins().fetch(c)["end"].to_numpy()
        per_chr[c] = len(starts)
        if a.shape[0] < 8:
            continue
        # Balancing leaves the matrix asymmetric at the last bit; cis_eig checks.
        a = 0.5 * (a + a.T)
        try:
            _ev, evec = eigdecomp.cis_eig(a, n_eigs=1, clip_percentile=99.9)
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"[tracks]   WARNING {c}: eigendecomposition failed ({exc})")
            continue
        e = np.asarray(evec[0], dtype=float)

        finite = e == e
        for s, en, v in zip(starts[finite], ends[finite], e[finite], strict=True):
            rows.append((c, int(s), int(en), float(v)))

        # A compartment eigenvector is smooth on the megabase scale, so neighbouring
        # bins correlate strongly.  A near-zero lag-1 autocorrelation means the
        # leading eigenvector is not the compartment one, which is what a too-sparse
        # contact map produces.  Warn rather than write a noise track silently.
        ev = e[finite]
        if len(ev) >= 10:
            lag1 = float(np.corrcoef(ev[:-1], ev[1:])[0, 1])
            quality[c] = round(lag1, 4)
            if lag1 < 0.3:
                print(
                    f"[tracks]   WARNING {c}: lag-1 autocorrelation {lag1:.3f} is too low for a "
                    f"compartment profile. The contact map is probably too sparse at this "
                    f"resolution. Try a coarser --resolution or a deeper mcool."
                )

    _write_bedgraph(out, rows)
    return {
        "resolution": resolution,
        "bins": len(rows),
        "chroms": per_chr,
        "lag1_autocorr": quality,
    }


# --- signal from a bigWig ----------------------------------------------------


def derive_signal(
    bigwig: Path, out: Path, resolution: int, chroms: list[str] | None
) -> dict[str, Any]:
    """bigWig -> mean signal per fixed-width bin, as a bedGraph.

    Values are written raw.  `gnome3d.tracks.normalize_signal_map` applies the
    HiP-HoP log-and-rescale once over everything the engine loads, which keeps the
    bridging strength comparable across regions.
    """
    import pyBigWig  # noqa: PLC0415

    bw = pyBigWig.open(str(bigwig))
    try:
        avail = bw.chroms()
        want = [c for c in (chroms or list(avail)) if c in avail]
        if not want:
            raise SystemExit(f"none of the requested chromosomes are in {bigwig}")

        rows: list[tuple[str, int, int, float]] = []
        for chr_ in want:
            length = int(avail[chr_])
            n_bins = length // resolution
            if n_bins < 1:
                continue
            vals = bw.stats(chr_, 0, n_bins * resolution, type="mean", nBins=n_bins)
            for i, v in enumerate(vals):
                if v is None:
                    continue
                fv = float(v)
                if fv != fv:
                    continue
                rows.append((chr_, i * resolution, (i + 1) * resolution, fv))
    finally:
        bw.close()

    _write_bedgraph(out, rows)
    return {"resolution": resolution, "bins": len(rows), "chroms": sorted({r[0] for r in rows})}


# --- study -------------------------------------------------------------------


def _find_one(root: Path, patterns: list[str]) -> Path | None:
    """Largest match wins.

    A cell line's directory can hold more than one file for an assay, and some are
    shallow stubs.  Picking alphabetically once selected a 293k-contact mcool over
    the 3M-contact one beside it, which produced a compartment track that was pure
    noise.  Size is a crude but reliable proxy for depth.
    """
    for pat in patterns:
        hits = sorted(root.glob(pat), key=lambda p: p.stat().st_size, reverse=True)
        if hits:
            return hits[0]
    return None


class TracksStudy:
    name = "tracks"
    help = "derive compartment and accessibility bedGraphs from an mcool and a bigWig"

    def add_args(self, p: ArgumentParser) -> None:
        p.add_argument(
            "--mcool", default=None, help="Hi-C mcool (default: search data/_hic/<cell>)"
        )
        p.add_argument(
            "--bigwig", default=None, help="ATAC bigWig (default: search data/_epigenome)"
        )
        p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
        p.add_argument("--signal-resolution", type=int, default=DEFAULT_SIGNAL_RESOLUTION)
        p.add_argument("--chroms", default=None, help="comma list; default every chromosome")
        p.add_argument("--force", action="store_true", help="rebuild even when up to date")
        p.add_argument("--skip-compartments", action="store_true")
        p.add_argument("--skip-signal", action="store_true")

    def run(self, ctx: Context, args: Namespace) -> None:
        cell = ctx.cell
        root = Path(ctx.data_root)
        out_dir = root / cell
        chroms = [c.strip() for c in args.chroms.split(",")] if args.chroms else None

        lock_path = out_dir / f"{cell}_tracks.lock.json"
        lock = _load_lock(lock_path)

        if not args.skip_compartments:
            mcool = Path(args.mcool) if args.mcool else _find_one(root / "_hic" / cell, ["*.mcool"])
            if mcool is None or not mcool.exists():
                print(
                    f"[tracks] no mcool for {cell} under {root / '_hic' / cell}. "
                    f"Fetch it first: python -m validation fetch "
                    f"--manifest validation/manifests/{cell}_hic.json --out {root}/_hic"
                )
            else:
                out = out_dir / f"{cell}_compartments.bedGraph"
                key = _cache_key(mcool, args.resolution, chroms)
                if _up_to_date(out, key, lock.get("compartments"), args.force):
                    print(f"[tracks] compartments up to date: {out}")
                else:
                    print(f"[tracks] deriving compartments from {mcool.name} @ {args.resolution}")
                    meta = derive_compartments(mcool, out, args.resolution, chroms)
                    lock["compartments"] = {
                        "input": key,
                        "source": str(mcool),
                        "out": str(out),
                        **meta,
                    }
                    print(f"[tracks]   wrote {meta['bins']} bins -> {out}")

        if not args.skip_signal:
            bw = (
                Path(args.bigwig)
                if args.bigwig
                else _find_one(
                    root / "_epigenome",
                    [f"{cell}/*ATAC*.bigWig", f"{cell}/*ATAC*.bw", "*ATAC*.bigWig"],
                )
            )
            if bw is None or not bw.exists():
                print(
                    f"[tracks] no ATAC bigWig for {cell} under {root / '_epigenome'}. "
                    f"Fetch it first: python -m validation fetch "
                    f"--manifest validation/manifests/{cell}.json --out {root}/_epigenome"
                )
            else:
                out = out_dir / f"{cell}_atac.bedGraph"
                key = _cache_key(bw, args.signal_resolution, chroms)
                if _up_to_date(out, key, lock.get("atac"), args.force):
                    print(f"[tracks] atac up to date: {out}")
                else:
                    print(f"[tracks] binning {bw.name} @ {args.signal_resolution}")
                    meta = derive_signal(bw, out, args.signal_resolution, chroms)
                    lock["atac"] = {"input": key, "source": str(bw), "out": str(out), **meta}
                    print(f"[tracks]   wrote {meta['bins']} bins -> {out}")

        if lock:
            _save_lock(lock_path, lock)
            print(f"[tracks] lockfile {lock_path}")


register(TracksStudy())
