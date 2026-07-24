"""Synthetic ground-truth reconstruction and noise robustness. The 2016 paper's core model
validation from Suppl. §III to IV, Fig. S19 to S22, for all three variants.

Generate a known 3D structure, synthesize its singleton contact heatmap by inverting the model's
freq to distance power law, reconstruct with the reference, python parity, and python tuned,
and measure how faithfully each recovers the truth via RMSD, which is mirror-insensitive, and the
paper's contact measure d_AB. Sweeping Gaussian heatmap noise gives the robustness curve, Fig. S19
to S22.

Beads are bins, the paper's nodes. Each bin is a tiny 2-anchor loop so the anchors survive
remove_empty_anchors. The inter-bin gaps auto-split into one-loop segments that the singleton
heatmap positions. Reconstructed leaf beads are aggregated back to per-bin centroids. The model's
distance units are normalized, so absolute scale is not recoverable. Structures are therefore
scale-matched to the truth by median pairwise distance before scoring, measuring shape fidelity
per the paper's alignment.

    python -m validation synthetic --nodes 50 -n 5 --noise 0.0,0.25,0.5,1.0
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from gnome3d.data import ContactData
from gnome3d.settings import Settings
from validation import metrics
from validation.core import variants
from validation.core.config import apply_flags, settings_for_cell
from validation.core.ensemble import run_ensemble
from validation.core.regions import parse_region_arg
from validation.studies import Context, Study, register

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)

# --------------------------------------------------------------------------- synthetic structure


def synthetic_structure(n: int, seed: int, step: float = 1.0, ev: float = 0.75) -> np.ndarray:
    """A known (N,3) ground-truth structure. A roughly self-avoiding random walk, the paper's
    synthetic chromatin fibre. ev rejects steps that come within that radius of the last 15 beads."""
    rng = np.random.default_rng(seed)
    pts = [np.zeros(3)]
    while len(pts) < n:
        d = rng.normal(size=3)
        for _ in range(300):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            cand = pts[-1] + step * d
            if all(np.linalg.norm(cand - p) > ev for p in pts[-15:]):
                pts.append(cand)
                break
        else:
            pts.append(pts[-1] + step * d)
    return np.asarray(pts[:n])


def write_synthetic_inputs(
    coords: np.ndarray,
    tmp: Path,
    chrom: str = "chr1",
    binsize: int = 1_000_000,
    start: int = 5_000_000,
    scale: float = 25.0,
    power: float = -0.6,
    K: float = 3000.0,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[str, np.ndarray]:
    """Write anchors, clusters, singletons, and breaks for a synthetic structure. Each bin is a tiny
    2-anchor loop. Singleton counts invert dist = scale·freq^power, so count is proportional to
    (d/scale)^(1/power). Gaussian noise, in units of mean count, perturbs the heatmap for the
    robustness sweep. Returns (region, edges)."""
    n = len(coords)
    g = start + np.arange(n) * binsize + binsize // 2
    with open(tmp / "anchors.bed", "w") as fa, open(tmp / "clusters.bedpe", "w") as fc:
        for m in g:
            fa.write(f"{chrom}\t{m-3000}\t{m-1000}\tN\n")
            fa.write(f"{chrom}\t{m+1000}\t{m+3000}\tN\n")
            fc.write(f"{chrom}\t{m-3000}\t{m-1000}\t{chrom}\t{m+1000}\t{m+3000}\t50\n")
    (tmp / "breaks.bed").write_text("")
    D = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    D[iu] = pdist(coords)
    counts = np.zeros((n, n))
    counts[iu] = K * (D[iu] / scale) ** (1.0 / power)
    if noise > 0:
        rng = np.random.default_rng(seed)
        counts[iu] = counts[iu] + rng.normal(scale=noise * counts[iu].mean(), size=counts[iu].shape)
    rows = [
        f"{chrom}\t{g[i]}\t{g[i]+1}\t{chrom}\t{g[j]}\t{g[j]+1}\t{int(round(counts[i, j]))}"
        for i, j in zip(*iu, strict=True)
        if round(counts[i, j]) >= 1
    ]
    (tmp / "singletons.bedpe").write_text("\n".join(rows) + "\n")
    edges = start + np.arange(n + 1) * binsize
    return f"{chrom}:{start}-{start + n * binsize}", edges


# --------------------------------------------------------------------------- reconstruction (3 variants)


def _py_settings(tmp: Path, variant: str) -> Settings:
    s = settings_for_cell("GM12878", "data", None)
    s = apply_flags(
        s,
        {
            "data_anchors": str((tmp / "anchors.bed").resolve()),
            "data_pet_clusters": str((tmp / "clusters.bedpe").resolve()),
            "data_singletons": str((tmp / "singletons.bedpe").resolve()),
            "data_singletons_inter": "",
            "data_segment_split": str((tmp / "breaks.bed").resolve()),
        },
    )
    if variant == "parity":
        s = apply_flags(s, variants.FEATURES_OFF)
    return s


def _cpp_config(tmp: Path, fast: bool) -> Path:
    """3dnome .ini pointing at the synthetic anchors, clusters, and singletons. Empty data_dir
    means the file paths are absolute."""
    base = tmp / "synth_base.ini"
    ig.write_config(base, fast=fast)
    s = settings_for_cell("GM12878", "data")
    block = (
        "[data]\n"
        "data_dir = \n"
        f"anchors = {(tmp / 'anchors.bed').resolve()}\n"
        f"clusters = {(tmp / 'clusters.bedpe').resolve()}\n"
        "factors = CTCF\n"
        f"singletons = {(tmp / 'singletons.bedpe').resolve()}\n"
        "split_singleton_files_by_chr = no\n"
        "singletons_inter = \n"
        f"segment_split = {(tmp / 'breaks.bed').resolve()}\n"
        f"centromeres = {Path(s.data_path(s.data_centromeres)).resolve()}"
    )
    text = re.sub(r"\[data\].*?(?=\n\[)", block, base.read_text(), count=1, flags=re.DOTALL)
    cfg = tmp / "synth_cpp.ini"
    cfg.write_text(text)
    return cfg


def _to_centroids(beads: list, edges: np.ndarray) -> np.ndarray:
    """Aggregate reconstructed leaf beads to per-bin centroids by genomic midpoint."""
    coords, mids = metrics.to_arrays(beads)
    idx = np.searchsorted(edges, mids, side="right") - 1
    n = len(edges) - 1
    cent = np.full((n, 3), np.nan)
    for b in range(n):
        sel = idx == b
        if sel.any():
            cent[b] = coords[sel].mean(0)
    return cent


def reconstruct(variant: str, tmp: Path, region: str, edges: np.ndarray, n: int,
                 ref_workers: int, fast: bool) -> list[np.ndarray]:
    """Reconstruct n structures for a variant. Return per-structure per-bin centroid arrays.

    parity and tuned build their own custom Settings pointing at the synthetic anchors, clusters,
    and singletons written by write_synthetic_inputs, then run through run_ensemble. reference
    dispatches to the shared variants.run_reference helper with the equivalent .ini."""
    if variant in ("tuned", "parity"):
        chrs_list, bed = parse_region_arg(region)
        s = _py_settings(tmp, variant)
        data = ContactData.from_files(s, chrs_list, bed)
        ens = run_ensemble(s, data, chrs_list, bed, n)
    else:  # reference
        cfg = _cpp_config(tmp, fast)
        ens = variants.run_reference(cfg, region, n, ref_workers=ref_workers, label="synth")
    return [_to_centroids(b, edges) for b in ens]


# --------------------------------------------------------------------------- scoring


def _norm_scale(truth: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x * (np.median(pdist(truth)) / np.median(pdist(x)))


def score(truth: np.ndarray, centroid_list: list[np.ndarray]) -> dict[str, float]:
    """Mean RMSD, mirror-insensitive, and contact measure vs the ground truth over the ensemble,
    each structure scale-matched to the truth. Uses the bins recovered in that structure."""
    rmsds, cms = [], []
    for cent in centroid_list:
        ok = ~np.isnan(cent[:, 0])
        if ok.sum() < 4:
            continue
        t, r = truth[ok], _norm_scale(truth[ok], cent[ok])
        rmsds.append(metrics.rmsd_superpose(t, r))
        cms.append(metrics.contact_measure(t, r, expected=pdist(t)))
    return {
        "rmsd": float(np.mean(rmsds)) if rmsds else float("nan"),
        "contact": float(np.mean(cms)) if cms else float("nan"),
        "n_ok": float(len(rmsds)),
    }


class Synthetic(Study):
    name = "synthetic"
    help = "synthetic ground-truth reconstruction and noise robustness, 3 variants"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--nodes", type=int, default=50, help="synthetic structure size (paper: 50, 100)")
        p.add_argument("--seed", type=int, default=1, help="ground-truth structure seed")
        p.add_argument(
            "--noise",
            default="0.0",
            help="comma-sep Gaussian noise levels (× mean count). '0.0' = Test 1 only; "
            "e.g. '0.0,0.1,0.25,0.5,1.0' = Test 2 robustness sweep",
        )
        p.add_argument("--variants", default="reference,parity,tuned")

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        if not ig.CPP_BIN.exists() and "reference" in args.variants:
            sys.exit(f"[error] reference binary not found: {ig.CPP_BIN}\n  run: make -C 3dnome")

        noise_levels = [float(x) for x in args.noise.split(",")]
        variant_names = args.variants.split(",")
        truth = synthetic_structure(args.nodes, args.seed)
        print(f"synthetic ground truth: {args.nodes} nodes (seed {args.seed}), "
              f"ensemble n={ctx.n} per variant\n")
        print(f"  {'noise':>6} | " + " | ".join(f"{v:^21}" for v in variant_names))
        print(f"  {'':>6} | " + " | ".join(f"{'RMSD':>9} {'contact':>10}" for _ in variant_names))
        print("  " + "-" * (9 + len(variant_names) * 24))

        baseline_random = None
        for noise in noise_levels:
            tmp = Path(tempfile.mkdtemp(prefix=f"synth_{noise}_"))
            region, edges = write_synthetic_inputs(truth, tmp, noise=noise, seed=args.seed)
            cells = []
            for v in variant_names:
                try:
                    cl = reconstruct(v, tmp, region, edges, ctx.n, ctx.ref_workers, ctx.fast)
                    sc = score(truth, cl)
                except Exception as e:  # noqa: BLE001
                    sc = {"rmsd": float("nan"), "contact": float("nan"), "n_ok": 0.0}
                    print(f"  [warn] {v} @ noise={noise}: {e}")
                cells.append(sc)
            row = " | ".join(f"{c['rmsd']:>9.3f} {c['contact']:>10.4f}" for c in cells)
            print(f"  {noise:>6.2f} | {row}")
            if baseline_random is None:
                rng = np.random.default_rng(999)
                rnd = _norm_scale(truth, rng.normal(size=truth.shape))
                baseline_random = (
                    metrics.rmsd_superpose(truth, rnd),
                    metrics.contact_measure(truth, rnd, expected=pdist(truth)),
                )
        print(f"\n  random-structure baseline: RMSD {baseline_random[0]:.3f}  "
              f"contact {baseline_random[1]:.4f}  (any real reconstruction must be far below)")
        print("  lower = better; RMSD & contact both scale-matched to the truth (arbitrary units).")


register(Synthetic())
