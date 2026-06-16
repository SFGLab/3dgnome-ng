#!/usr/bin/env python3
"""Hi-C self-correlation study: feed the experimental Hi-C into 3dgnome as additional singleton
contacts, then correlate the reconstructed structure against HELD-OUT Hi-C contacts.

Why held-out: 3dgnome turns singleton contact frequencies into target distances the MC minimises
toward (``io.load_singletons`` -> ``create_singleton_heatmap`` -> ``util.freq_to_dist_heatmap``).
So correlating the output against the SAME contacts you fed in measures self-consistency by
construction, not prediction. We therefore split the Hi-C bin-pairs into TRAIN (fed in as
singletons) and TEST (held out) and correlate only against TEST — a genuine generalisation check.
This mirrors the original 3D-GNOME paper, which both showed ChIA-PET≈Hi-C (ρ≈0.67–0.88, Fig. 2) and
ran the same engine on Hi-C input (Suppl. S7).

This module provides the converter + split + held-out scorer; the run driver (override
``data_singletons`` -> run ensemble -> score) is wired in ``run_self_correlation``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import load_singletons  # noqa: E402
from gnome3d.types import F64Array, I64Array  # noqa: E402
from validation import metrics  # noqa: E402
from validation.cell_config import settings_for_cell  # noqa: E402
from validation.compare_reference import TUNED_FEATURES  # noqa: E402
from validation.validate import _apply_flags, _chrs_and_region, run_ensemble  # noqa: E402


def _stable_holdout(i: int, j: int, seed: int, frac: float = 0.5) -> bool:
    """Deterministic per-pair train/test assignment (True = TEST/held-out), a value-independent
    hash of the bin-pair so the split is stable across runs. ``frac`` is the held-out fraction.

    NOTE: random per-pair hold-out is a LENIENT test — each held-out pair is surrounded by TRAIN
    pairs, so the smooth Hi-C signal leaks through neighbours. A low held-out score under this
    split is therefore meaningful (the easy case failed); a strict test would hold out contiguous
    blocks / distance bands instead."""
    h = (i * 2654435761 + j * 40503 + seed * 2246822519) & 0xFFFFFFFF
    return (h % 1000) < int(frac * 1000)


def hic_to_singleton_bedpe(
    mcool_path: str,
    region: str,
    binsize: int,
    out_path: Path,
    holdout: bool = True,
    seed: int = 0,
    min_count: int = 1,
    count_scale: float = 1.0,
    holdout_frac: float = 0.5,
) -> tuple[F64Array, I64Array, np.ndarray, np.ndarray]:
    """Read the RAW Hi-C counts for ``region`` and write the TRAIN bin-pairs as a 7-column
    singleton BEDPE (``chr start end chr start end count``) that 3dgnome can ingest. Raw integer
    counts are used (PET-singleton-like; the freq->dist power law handles scaling).

    Returns ``(c_obs_balanced, bin_starts, test_mask, train_mask)`` — boolean (B,B) masks of the
    held-out pairs and the fed-in (train) pairs, and the ICE-balanced observed matrix (for the
    faithful metric). With ``holdout=False`` all pairs are written (pure self-consistency) and
    ``test_mask`` is all upper-tri off-diagonal.
    """
    import cooler

    avail = sorted(
        int(p.rsplit("/", 1)[-1])
        for p in cooler.fileops.list_coolers(mcool_path)
        if p.rsplit("/", 1)[-1].isdigit()
    )
    res = (binsize if binsize in avail else min(avail, key=lambda r: abs(r - binsize))) if avail else binsize
    uri = f"{mcool_path}::/resolutions/{res}" if avail else mcool_path
    c = cooler.Cooler(uri)
    raw = np.nan_to_num(np.asarray(c.matrix(balance=False).fetch(region), dtype=np.float64))
    bal = np.nan_to_num(np.asarray(c.matrix(balance=True).fetch(region), dtype=np.float64))
    bins = c.bins().fetch(region)
    starts = bins["start"].to_numpy().astype(np.int64)
    ends = bins["end"].to_numpy().astype(np.int64)
    chrom = str(bins["chrom"].to_numpy()[0])
    nb = len(starts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_mask = np.zeros((nb, nb), dtype=bool)
    train_mask = np.zeros((nb, nb), dtype=bool)
    n_train = n_test = 0
    with open(out_path, "w") as f:
        for i in range(nb):
            for j in range(i + 1, nb):
                cnt = int(round(raw[i, j]))
                if cnt < min_count:
                    continue
                is_test = holdout and _stable_holdout(i, j, seed, holdout_frac)
                if is_test:
                    test_mask[i, j] = test_mask[j, i] = True
                    n_test += 1
                else:
                    # TRAIN pair -> singleton BEDPE row (bin start as the contact position)
                    train_mask[i, j] = train_mask[j, i] = True
                    sc = max(1, int(round(cnt * count_scale)))
                    f.write(
                        f"{chrom}\t{starts[i]}\t{ends[i]}\t{chrom}\t{starts[j]}\t{ends[j]}\t{sc}\n"
                    )
                    n_train += 1
    if not holdout:
        iu = np.triu_indices(nb, 1)
        test_mask[iu] = True
        test_mask = test_mask | test_mask.T
    print(
        f"[self_corr] {region} @ {res}bp: {n_train} train singletons -> {out_path.name}, "
        f"{n_test if holdout else int(test_mask.sum() // 2)} held-out pairs"
    )
    return bal, starts, test_mask, train_mask


def faithful_on_masks(
    coords_list: list[F64Array],
    mids: I64Array,
    c_obs: F64Array,
    bin_starts: I64Array,
    binsize: int,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Faithful MultiMM-style correlation (1/(d+1)^3 simulated centroids vs ICE-balanced observed),
    evaluated on each named bin-pair mask. Build the simulated map ONCE; correlate on each mask —
    so we get TRAIN (fed-in) and TEST (held-out) Pearson from one pass. TRAIN≈TEST≫0 means the
    structures reproduce contacts and generalise; TRAIN high & TEST low = no generalisation;
    both low = the model isn't even fitting what it was given."""
    from validation import contacts

    cent0 = contacts._bin_centroids(coords_list[0], mids, bin_starts, binsize)
    valid = ~np.isnan(cent0[:, 0])
    if valid.sum() < 4:
        return dict.fromkeys(masks, float("nan"))
    nv = int(valid.sum())
    sim = np.zeros((nv, nv))
    for coords in coords_list:
        cent = contacts._bin_centroids(coords, mids, bin_starts, binsize)[valid]
        step = np.linalg.norm(np.diff(cent, axis=0), axis=1)
        scale = float(np.median(step[step > 0])) if np.any(step > 0) else 1.0
        diff = cent[:, None, :] - cent[None, :, :]
        d = np.sqrt((diff * diff).sum(-1)) / scale
        sim += 1.0 / (d + 1.0) ** 3
    sim /= len(coords_list)
    obs = np.asarray(c_obs)[np.ix_(valid, valid)]
    out: dict[str, float] = {}
    for name, mk in masks.items():
        m = np.asarray(mk)[np.ix_(valid, valid)]
        a, b = sim[m], obs[m]
        out[name] = (
            float(np.corrcoef(a, b)[0, 1])
            if a.size >= 4 and a.std() > 1e-12 and b.std() > 1e-12
            else float("nan")
        )
    return out


def _chiapet_median_and_rows(
    orig_path: str, chr_set: set[str], region: object
) -> tuple[float, list[tuple]]:
    """ChIA-PET singletons for the region: their median score (for count-scaling Hi-C to a
    comparable magnitude) and the rows themselves (to re-emit into the combined BEDPE)."""
    rows = load_singletons(orig_path, chr_set, region)  # (c1,p1,c2,p2,sc)
    med = float(np.median([r[4] for r in rows])) if rows else 1.0
    return med, rows


def run_self_correlation(region: str, args: argparse.Namespace, tmp: Path) -> dict:
    """One region: feed Hi-C TRAIN singletons (replace or augment ChIA-PET) into the tuned model,
    run the ensemble, and correlate the structure against the HELD-OUT Hi-C contacts."""
    tuned = _apply_flags(settings_for_cell(args.cell, args.data_root, args.quality), TUNED_FEATURES)
    if args.coarsen_bp > 0:  # ~20kb/bead keeps a 20 Mb region tractable (MultiMM-comparable geom)
        tuned = _apply_flags(
            tuned, {"use_dynamic_loop_density": True, "target_bp_per_subanchor": args.coarsen_bp}
        )
    chrs_list, bed_region = _chrs_and_region(region)
    chr_set = set(chrs_list)

    safe = region.replace(":", "_").replace("-", "_")
    hic_bedpe = tmp / f"hic_{safe}.bedpe"
    conv = {"holdout": True, "seed": args.seed, "holdout_frac": args.holdout_frac}
    if args.hic_singletons == "augment":
        # scale Hi-C raw counts so their median matches the ChIA-PET singleton median, so neither
        # source swamps the other in the freq->distance heatmap (depths/magnitudes differ wildly).
        med_c, chiapet_rows = _chiapet_median_and_rows(
            tuned.data_path(tuned.data_singletons), chr_set, bed_region
        )
        hic_to_singleton_bedpe(args.hic, region, args.binsize, hic_bedpe, **conv)
        med_h = float(np.median([int(ln.split()[6]) for ln in hic_bedpe.read_text().splitlines()]))
        count_scale = (med_c / med_h) if med_h > 0 else 1.0
        # rewrite Hi-C with the scale, then combine with the ChIA-PET region rows
        bal, bin_starts, test_mask, train_mask = hic_to_singleton_bedpe(
            args.hic, region, args.binsize, hic_bedpe, count_scale=count_scale, **conv
        )
        combined = tmp / f"combined_{safe}.bedpe"
        with open(combined, "w") as f:
            for c1, p1, c2, p2, sc in chiapet_rows:
                f.write(f"{c1}\t{p1}\t{p1 + 1}\t{c2}\t{p2}\t{p2 + 1}\t{sc}\n")
            f.write(hic_bedpe.read_text())
        tuned.data_singletons = str(combined)
        print(f"[self_corr] augment: ChIA-PET median {med_c:.0f}, Hi-C scaled ×{count_scale:.3g}")
    else:  # replace
        bal, bin_starts, test_mask, train_mask = hic_to_singleton_bedpe(
            args.hic, region, args.binsize, hic_bedpe, **conv
        )
        tuned.data_singletons = str(hic_bedpe)
    tuned.data_singletons_inter = ""  # intra-chromosomal region

    data = ContactData.from_files(tuned, chrs_list, bed_region)
    ens = run_ensemble(tuned, data, chrs_list, bed_region, args.n)
    coords_list, mids_l = [], []
    for beads in ens:
        c, mm = metrics.to_arrays(beads)
        coords_list.append(c)
        mids_l.append(mm)
    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else args.binsize
    rho = faithful_on_masks(
        coords_list, mids_l[0], bal, bin_starts, eff, {"test": test_mask, "train": train_mask}
    )
    return {
        "region": region,
        "heldout_pearson": rho["test"],
        "train_pearson": rho["train"],
        "n_test_pairs": int(np.asarray(test_mask).sum() // 2),
        "n_train_pairs": int(np.asarray(train_mask).sum() // 2),
        "mode": args.hic_singletons,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cell", required=True)
    p.add_argument("--hic", required=True, help="observed Hi-C .mcool")
    p.add_argument("--data-root", default="data")
    p.add_argument("--chroms", default=None)
    p.add_argument(
        "--hic-singletons",
        choices=["replace", "augment"],
        default="replace",
        help="replace = pure Hi-C-driven (clean, MultiMM-comparable); augment = ChIA-PET + Hi-C "
        "(count-scaled to match), tests whether Hi-C improves the model",
    )
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--n-regions", type=int, default=4)
    p.add_argument("--quality", default="full")
    p.add_argument("--binsize", type=int, default=25000)
    p.add_argument("--min-ibs", type=int, default=2)
    p.add_argument("--max-ibs", type=int, default=6)
    p.add_argument("--max-mb", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0, help="held-out split seed")
    p.add_argument(
        "--holdout-frac", type=float, default=0.5, help="fraction of bin-pairs held out for testing"
    )
    p.add_argument(
        "--coarsen-bp",
        type=int,
        default=0,
        help="bp/bead coarsening (0=off); set ~20000 with --min-ibs 12 --max-mb 24 for a 20 Mb "
        "MultiMM-comparable geometry that stays tractable",
    )
    p.add_argument("--region", default=None, help="single region override")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from validation.sweep import enumerate_regions

    if args.region:
        regions = [args.region]
    else:
        s_meta = settings_for_cell(args.cell, args.data_root)
        bp = s_meta.data_path(s_meta.data_segment_split)
        chroms = args.chroms.split(",") if args.chroms else None
        regions = enumerate_regions(
            bp, args.n_regions, chroms=chroms, min_ibs=args.min_ibs, max_ibs=args.max_ibs,
            max_mb=args.max_mb,
        )
    if not regions:
        sys.exit("[error] no regions found")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="gnome3d_selfcorr_"))
    results = []
    for region in regions:
        print(f"\n[self_corr] {region} ({args.hic_singletons}, n={args.n}, {args.quality}):")
        r = run_self_correlation(region, args, tmp)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        print(
            f"  [done] train Pearson = {r['train_pearson']:.4f} | held-out = "
            f"{r['heldout_pearson']:.4f}  ({r['n_train_pairs']}/{r['n_test_pairs']} pairs)"
        )

    fin = lambda k: [r[k] for r in results if r[k] == r[k]]  # drop NaN
    med_tr = float(np.median(fin("train_pearson"))) if fin("train_pearson") else float("nan")
    med_te = float(np.median(fin("heldout_pearson"))) if fin("heldout_pearson") else float("nan")
    print(f"\n{'=' * 70}\nHi-C SELF-CORRELATION ({args.hic_singletons}) — median over {len(results)} regions")
    print(f"  TRAIN  Pearson (fed-in contacts):    {med_tr:.4f}")
    print(f"  TEST   Pearson (HELD-OUT contacts):  {med_te:.4f}   <- generalisation")
    print("  Read: train≈test≫0 = fits & generalises; train≫test = no generalisation;")
    print("        both≈0 = model isn't reproducing even the contacts it was given.")
    print("=" * 70)


if __name__ == "__main__":
    main()
