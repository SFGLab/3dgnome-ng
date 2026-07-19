#!/usr/bin/env python3
"""Hi-C self-correlation study: feed the experimental Hi-C into 3dgnome as singleton contacts
(``replace`` = pure Hi-C-driven, or ``augment`` = added to ChIA-PET), then correlate the
reconstructed structure against HELD-OUT Hi-C contacts — for THREE variants (C++ reference,
python parity with features off, python tuned) so the result is diagnosable.

Why held-out: 3dgnome turns singleton contact frequencies into target distances the MC minimises
toward (``io.load_singletons`` -> ``create_singleton_heatmap`` -> ``util.freq_to_dist_heatmap``).
So correlating the output against the SAME contacts you fed in measures self-consistency by
construction, not prediction. We therefore split the Hi-C bin-pairs into TRAIN (fed in as
singletons) and TEST (held out) and correlate only against TEST — a genuine generalisation check.
This mirrors the original 3D-GNOME paper, which both showed ChIA-PET≈Hi-C (ρ≈0.67–0.88, Fig. 2) and
ran the same engine on Hi-C input (Suppl. S7).

Segmentation uses **Hi-C TAD boundaries** (``hic_boundaries``, cooltools insulation), not CTCF/CCD,
so the model's IBs align with the TADs we score against. The run driver (override
``data_singletons`` + ``data_segment_split`` -> run each variant -> score) is ``run_self_correlation``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(_ROOT))

from gnome3d.data import ContactData  # noqa: E402
from gnome3d.io import load_singletons  # noqa: E402
from gnome3d.types import F64Array, I64Array  # noqa: E402
from validation import metrics  # noqa: E402
from validation.cell_config import apply_flags, settings_for_cell, with_singletons  # noqa: E402
from validation.hic_boundaries import write_breakpoints  # noqa: E402
from validation.validate import _chrs_and_region, run_ensemble  # noqa: E402

sys.path.insert(0, str(_ROOT / "harness"))
import integration as ig  # noqa: E402  (the C++ reference runner; harness is not a package)

MAX_LEVEL = 2  # heatmap + arc + smooth MC (same as compare_reference / the integration test)
# Feature flags that distinguish our TUNED config from the reference baseline (turned OFF for parity).
_FEATURES_OFF = {
    "use_excluded_volume": False,
    "use_confinement": False,
    "use_dynamic_loop_density": False,
    "use_ib_mc": False,
}


def _cpp_selfhic_config(
    cell: str, data_root: str, singletons_abs: Path, segments_abs: Path, tmp: Path, fast: bool
) -> Path:
    """Write a 3dnome .ini for the C++ reference's self-HiC run: CTCF anchors/clusters (the loops
    are still CTCF) + the **Hi-C** singletons + **Hi-C TAD** segment-split. Uses an EMPTY data_dir
    so every path is absolute (the C++ does ``data_dir + filename``, so empty + abs = abs)."""
    base = tmp / "selfhic_base.ini"
    ig.write_config(base, fast=fast)  # all the simulation params, parity-faithful
    s = settings_for_cell(cell, data_root)
    abspath = lambda fn: str(Path(s.data_path(fn)).resolve())
    block = (
        "[data]\n"
        "data_dir = \n"  # empty -> the absolute paths below are used as-is
        f"anchors = {abspath(s.data_anchors)}\n"
        f"clusters = {abspath(s.data_pet_clusters)}\n"
        "factors = CTCF\n"
        f"singletons = {Path(singletons_abs).resolve()}\n"
        "split_singleton_files_by_chr = no\n"
        "singletons_inter = \n"
        f"segment_split = {Path(segments_abs).resolve()}\n"
        f"centromeres = {abspath(s.data_centromeres)}"
    )
    text = re.sub(r"\[data\].*?(?=\n\[)", block, base.read_text(), count=1, flags=re.DOTALL)
    cfg = tmp / "selfhic_cpp.ini"
    cfg.write_text(text)
    return cfg


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
    tuned = settings_for_cell(args.cell, args.data_root, args.quality)  # unified canonical config
    # Segment by Hi-C-derived TAD boundaries (not CTCF/CCD) so the model's IBs align with the TADs
    # we score against — the right segmentation for a Hi-C-driven self-correlation study.
    tuned = apply_flags(tuned, {"data_segment_split": str(args.hic_breakpoints)})
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
        singletons_path = combined
        print(f"[self_corr] augment: ChIA-PET median {med_c:.0f}, Hi-C scaled ×{count_scale:.3g}")
    else:  # replace
        bal, bin_starts, test_mask, train_mask = hic_to_singleton_bedpe(
            args.hic, region, args.binsize, hic_bedpe, **conv
        )
        singletons_path = hic_bedpe
    # centralized config modification (no inline Settings mutation)
    tuned = with_singletons(tuned, str(singletons_path), singletons_inter="")

    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else args.binsize
    masks = {"test": test_mask, "train": train_mask}
    data = ContactData.from_files(tuned, chrs_list, bed_region)
    # The fed-in Hi-C train contacts (pos_a, pos_b, score) are the V1 "input heat-map IFs": V1 asks
    # how well the structure reflects their imposed distance ordering (3dgnome 2016 Suppl. Spearman).
    hic_train_contacts = [
        (int(p[1]), int(p[4]), float(p[6]))
        for p in (ln.split() for ln in hic_bedpe.read_text().splitlines())
        if len(p) >= 7
    ]

    def score(cl: list, mids0: object) -> dict:
        d = faithful_on_masks(cl, mids0, bal, bin_starts, eff, masks)
        # V1 (faithful, per 3dgnome papers): mean Spearman(input IF, model 3D distance) over the
        # ensemble; ρ < 0 = high-frequency contacts placed close = good self-consistency.
        d["v1"] = float(
            np.nanmean([metrics.self_consistency(c, mids0, hic_train_contacts)[0] for c in cl])
        )
        return d

    def score_python(settings: object) -> dict:
        ens = run_ensemble(settings, data, chrs_list, bed_region, args.n)  # type: ignore[arg-type]
        cl = [metrics.to_arrays(b)[0] for b in ens]
        return score(cl, metrics.to_arrays(ens[0])[1])

    out: dict = {
        "region": region,
        "mode": args.hic_singletons,
        "n_test_pairs": int(np.asarray(test_mask).sum() // 2),
        "n_train_pairs": int(np.asarray(train_mask).sum() // 2),
    }
    # All three variants get the SAME Hi-C singletons + Hi-C TAD segmentation; only the model
    # differs — reference (C++ baseline), parity (our port, features OFF), tuned (features ON).
    # The reference baseline tells us whether a modest train/test number is OUR doing or the
    # metric/data/region ceiling; parity-vs-tuned isolates the feature contribution.
    print(f"  [{region}] python +tuned ...")
    out["tuned"] = score_python(tuned)
    print(f"  [{region}] python parity (features off) ...")
    out["parity"] = score_python(apply_flags(tuned, _FEATURES_OFF))
    if not args.no_reference and ig.CPP_BIN.exists():
        print(f"  [{region}] C++ reference ...")
        cfg = _cpp_selfhic_config(
            args.cell, args.data_root, Path(singletons_path), Path(args.hic_breakpoints),
            tmp, args.quality == "fast",
        )
        rdir = tmp / f"cpp_{safe}"
        rdir.mkdir(parents=True, exist_ok=True)
        ref_structs, _ = ig.run_cpp_ensemble_parallel(
            rdir, cfg, args.n, MAX_LEVEL, region, "selfhic", workers=getattr(args, "ref_workers", 0)
        )
        rcl = [metrics.to_arrays(b)[0] for b in ref_structs]
        out["reference"] = score(rcl, metrics.to_arrays(ref_structs[0])[1])
    else:
        out["reference"] = {"test": float("nan"), "train": float("nan"), "v1": float("nan")}
    return out


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
        "--window", type=int, default=250_000, help="cooltools insulation window for TAD boundaries"
    )
    p.add_argument(
        "--hic-breakpoints",
        default=None,
        help="precomputed Hi-C TAD breakpoints BED; default = derive from --hic (cooltools)",
    )
    p.add_argument("--region", default=None, help="single region override")
    p.add_argument(
        "--no-reference", action="store_true", help="skip the C++ reference + parity baselines"
    )
    p.add_argument(
        "--ref-workers", type=int, default=0, help="cores for the C++ reference (0 = auto)"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from validation.sweep import enumerate_regions

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chroms = (
        args.chroms.split(",")
        if args.chroms
        else ([args.region.split(":")[0]] if args.region else None)
    )
    # Hi-C TAD boundaries drive BOTH region selection and the model segmentation (self-HiC: the
    # model's IBs align with the Hi-C TADs we score against, instead of CTCF/CCD boundaries).
    if args.hic_breakpoints:
        args.hic_breakpoints = Path(args.hic_breakpoints)
    else:
        args.hic_breakpoints = out_path.with_name(f"{out_path.stem}__hic_tad_breaks.bed")
        if not args.hic_breakpoints.exists():
            write_breakpoints(args.hic, args.hic_breakpoints, chroms, args.window, args.binsize)

    if args.region:
        regions = [args.region]
    else:
        regions = enumerate_regions(
            str(args.hic_breakpoints), args.n_regions, chroms=chroms,
            min_ibs=args.min_ibs, max_ibs=args.max_ibs, max_mb=args.max_mb,
        )
    if not regions:
        sys.exit("[error] no regions found")

    tmp = Path(tempfile.mkdtemp(prefix="gnome3d_selfcorr_"))
    variants = ["reference", "parity", "tuned"]
    results = []
    for region in regions:
        print(f"\n[self_corr] {region} ({args.hic_singletons}, n={args.n}, {args.quality}):")
        r = run_self_correlation(region, args, tmp)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))
        cells = "  ".join(f"{v}:{r[v]['train']:+.3f}/{r[v]['test']:+.3f}" for v in variants)
        print(f"  [done] (train/test)  {cells}")

    def _fin(r: dict, v: str, key: str) -> bool:
        return r[v][key] == r[v][key]  # not NaN

    def med(rows: list, variant: str, key: str) -> float:
        xs = [r[variant][key] for r in rows if _fin(r, variant, key)]
        return float(np.median(xs)) if xs else float("nan")

    # NaN = a variant's structure collapsed on a region (features-off models with no EV can, on
    # underdetermined regions) → its per-variant median would be over a DIFFERENT region set than
    # tuned's. So also report the PAIRED median over regions where ALL variants are finite.
    matched = [r for r in results if all(_fin(r, v, "test") for v in variants)]
    print(f"\n{'=' * 74}\nHi-C SELF-CORRELATION ({args.hic_singletons}) — {len(results)} regions")
    print(f"  {'variant':<14}{'TRAIN':>9}{'TEST':>9}{'V1(ρ)':>9}{'finite':>8}    {'TEST(paired)':>13}")
    for v in variants:
        nfin = sum(_fin(r, v, "test") for r in results)
        flag = "  <- collapsed on some regions" if nfin < len(results) else ""
        print(
            f"  {v:<14}{med(results, v, 'train'):>9.3f}{med(results, v, 'test'):>9.3f}"
            f"{med(results, v, 'v1'):>9.3f}{nfin:>6}/{len(results)}    {med(matched, v, 'test'):>13.3f}{flag}"
        )
    print(f"  (paired = median over the {len(matched)} regions where ALL variants are finite)")
    print("  TEST = held-out Hi-C generalisation; V1(ρ) = 3dgnome-paper self-consistency Spearman")
    print("         (input Hi-C IF vs model 3D distance; want < 0 — high-freq contacts placed close).")
    print("  reference≈tuned ⇒ ceiling not our model; tuned≫parity ⇒ features help (EV stops collapse).")
    print("=" * 74)


if __name__ == "__main__":
    main()
