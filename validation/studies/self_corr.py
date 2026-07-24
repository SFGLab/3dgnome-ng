"""Hi-C self-correlation study. Feed the experimental Hi-C into 3dgnome as singleton contacts,
either replace for a pure Hi-C-driven run or augment to add them to ChIA-PET, then correlate the
reconstructed structure against held-out Hi-C contacts. Three variants are run so the result is
diagnosable. These are the reference, the python parity model with features off, and the python
tuned model.

3dgnome turns singleton contact frequencies into target distances the MC minimises toward, via
load_singletons then create_singleton_heatmap then freq_to_dist_heatmap. Correlating the output
against the same contacts fed in would measure self-consistency by construction rather than
prediction. The Hi-C bin-pairs are therefore split into train, fed in as singletons, and test, held
out, and only the test pairs are correlated against for a genuine generalisation check. This
mirrors the original 3D-GNOME paper, which showed ChIA-PET approximates Hi-C at rho about 0.67 to
0.88 in Fig. 2 and ran the same engine on Hi-C input in Suppl. S7.

Segmentation uses Hi-C TAD boundaries from validation.core.boundaries via cooltools insulation
rather than CTCF or CCD, so the model's IBs align with the TADs we score against. The SelfCorr
study overrides data_singletons and data_segment_split, runs each variant, then scores.

    python -m validation self-corr --hic path/to.mcool --hic-singletons replace --out out/selfcorr.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from gnome3d.data import ContactData
from gnome3d.io import load_singletons
from gnome3d.types import F64Array, I64Array
from validation import metrics
from validation.core import variants
from validation.core.boundaries import write_breakpoints
from validation.core.config import apply_flags, settings_for_cell, with_singletons
from validation.core.ensemble import run_ensemble
from validation.core.regions import enumerate_regions, parse_region_arg
from validation.studies import Context, Study, register

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harness"))
import integration as ig  # noqa: E402  (dev tooling; harness is not a package)


def _cpp_selfhic_config(
    cell: str, data_root: str, singletons_abs: Path, segments_abs: Path, tmp: Path, fast: bool
) -> Path:
    """Write a 3dnome .ini for the reference's self-HiC run. It combines CTCF anchors and
    clusters, since the loops are still CTCF, with the Hi-C singletons and the Hi-C TAD
    segment-split. The data_dir is left empty so every path is absolute. The reference concatenates
    data_dir and filename, so an empty prefix with an absolute path stays absolute."""
    base = tmp / "selfhic_base.ini"
    ig.write_config(base, fast=fast)  # all the simulation params, parity-faithful
    s = settings_for_cell(cell, data_root)
    abspath = lambda fn: str(Path(s.data_path(fn)).resolve())
    block = (
        "[data]\n"
        "data_dir = \n"  # empty so the absolute paths below are used as-is
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
    """Deterministic per-pair train/test assignment where True means the pair is held out for test.
    It is a value-independent hash of the bin-pair so the split is stable across runs. frac is the
    held-out fraction.

    Random per-pair hold-out is a lenient test. Each held-out pair is surrounded by train pairs, so
    the smooth Hi-C signal leaks through neighbours. A low held-out score under this split is
    therefore meaningful, since even the easy case failed. A strict test would hold out contiguous
    blocks or distance bands instead."""
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
    """Read the raw Hi-C counts for region and write the train bin-pairs as a 7-column singleton
    BEDPE of chr start end chr start end count that 3dgnome can ingest. Raw integer counts are used,
    much like PET singletons, since the freq-to-dist power law handles scaling.

    Returns (c_obs_balanced, bin_starts, test_mask, train_mask). test_mask and train_mask are
    boolean (B, B) masks of the held-out pairs and the fed-in train pairs, and c_obs_balanced is the
    ICE-balanced observed matrix used by the faithful metric. With holdout=False all pairs are
    written for a pure self-consistency check and test_mask covers the whole upper-triangular
    off-diagonal.
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
                    # train pair becomes a singleton BEDPE row with the bin start as the contact position
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
    """Faithful MultiMM-style correlation of 1/(d+1)^3 simulated centroids against the ICE-balanced
    observed matrix, evaluated on each named bin-pair mask. The simulated map is built once and
    correlated on each mask, so one pass yields both the train Pearson on fed-in pairs and the test
    Pearson on held-out pairs. Train and test both well above zero means the structures reproduce
    contacts and generalise. High train with low test means no generalisation. Both low means the
    model is not even fitting what it was given."""
    from validation.metrics import hic as contacts

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
    """ChIA-PET singletons for the region. Returns their median score, used to count-scale Hi-C to a
    comparable magnitude, and the rows themselves, re-emitted into the combined BEDPE."""
    rows = load_singletons(orig_path, chr_set, region)  # (c1,p1,c2,p2,sc)
    med = float(np.median([r[4] for r in rows])) if rows else 1.0
    return med, rows


def _run_region(
    region: str, ctx: Context, args: argparse.Namespace, hic_breakpoints: Path, tmp: Path
) -> dict:
    """Run one region. Feed the Hi-C train singletons, either replacing or augmenting ChIA-PET, into
    the tuned model, run the ensemble, and correlate the structure against the held-out Hi-C
    contacts."""
    tuned = settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)  # unified canonical config
    # Segment by Hi-C-derived TAD boundaries rather than CTCF or CCD so the model's IBs align with
    # the TADs we score against, the right segmentation for a Hi-C-driven self-correlation study.
    tuned = apply_flags(tuned, {"data_segment_split": str(hic_breakpoints)})
    chrs_list, bed_region = parse_region_arg(region)
    chr_set = set(chrs_list)

    safe = region.replace(":", "_").replace("-", "_")
    hic_bedpe = tmp / f"hic_{safe}.bedpe"
    conv = {"holdout": True, "seed": args.seed, "holdout_frac": args.holdout_frac}
    if args.hic_singletons == "augment":
        # scale Hi-C raw counts so their median matches the ChIA-PET singleton median, so neither
        # source swamps the other in the freq-to-distance heatmap, since their depths and magnitudes
        # differ widely.
        med_c, chiapet_rows = _chiapet_median_and_rows(
            tuned.data_path(tuned.data_singletons), chr_set, bed_region
        )
        hic_to_singleton_bedpe(ctx.hic, region, args.binsize, hic_bedpe, **conv)
        med_h = float(np.median([int(ln.split()[6]) for ln in hic_bedpe.read_text().splitlines()]))
        count_scale = (med_c / med_h) if med_h > 0 else 1.0
        # rewrite Hi-C with the scale, then combine with the ChIA-PET region rows
        bal, bin_starts, test_mask, train_mask = hic_to_singleton_bedpe(
            ctx.hic, region, args.binsize, hic_bedpe, count_scale=count_scale, **conv
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
            ctx.hic, region, args.binsize, hic_bedpe, **conv
        )
        singletons_path = hic_bedpe
    # centralized config modification instead of inline Settings mutation
    tuned = with_singletons(tuned, str(singletons_path), singletons_inter="")

    eff = int(bin_starts[1] - bin_starts[0]) if len(bin_starts) > 1 else args.binsize
    masks = {"test": test_mask, "train": train_mask}
    data = ContactData.from_files(tuned, chrs_list, bed_region)
    # The fed-in Hi-C train contacts of pos_a, pos_b and score are the self-consistency input
    # heat-map IFs. Self-consistency asks how well the structure reflects their imposed distance
    # ordering, the 3dgnome 2016 Suppl. Spearman.
    hic_train_contacts = [
        (int(p[1]), int(p[4]), float(p[6]))
        for p in (ln.split() for ln in hic_bedpe.read_text().splitlines())
        if len(p) >= 7
    ]

    def score(cl: list, mids0: object) -> dict:
        d = faithful_on_masks(cl, mids0, bal, bin_starts, eff, masks)
        # This is faithful to the 3dgnome papers. It is the mean Spearman of input IF against model
        # 3D distance over the ensemble. rho < 0 means high-frequency contacts placed close, good
        # self-consistency.
        d["self_consistency"] = float(
            np.nanmean([metrics.self_consistency(c, mids0, hic_train_contacts)[0] for c in cl])
        )
        return d

    def score_python(settings: object) -> dict:
        ens = run_ensemble(settings, data, chrs_list, bed_region, ctx.n)  # type: ignore[arg-type]
        cl = [metrics.to_arrays(b)[0] for b in ens]
        return score(cl, metrics.to_arrays(ens[0])[1])

    out: dict = {
        "region": region,
        "mode": args.hic_singletons,
        "n_test_pairs": int(np.asarray(test_mask).sum() // 2),
        "n_train_pairs": int(np.asarray(train_mask).sum() // 2),
    }
    # All three variants get the same Hi-C singletons and Hi-C TAD segmentation. Only the model
    # differs. reference is the reference baseline, parity is our port with features off, tuned has
    # features on. The reference baseline tells us whether a modest train/test number is our doing or
    # the metric, data or region ceiling. parity versus tuned isolates the feature contribution.
    print(f"  [{region}] python +tuned ...")
    out["tuned"] = score_python(tuned)
    print(f"  [{region}] python parity (features off) ...")
    out["parity"] = score_python(apply_flags(tuned, variants.FEATURES_OFF))
    if not args.no_reference and ig.CPP_BIN.exists():
        print(f"  [{region}] reference ...")
        cfg = _cpp_selfhic_config(
            ctx.cell, ctx.data_root, Path(singletons_path), hic_breakpoints,
            tmp, ctx.quality == "fast",
        )
        ref_structs = variants.run_reference(
            cfg, region, ctx.n, ref_workers=ctx.ref_workers, label="selfhic"
        )
        rcl = [metrics.to_arrays(b)[0] for b in ref_structs]
        out["reference"] = score(rcl, metrics.to_arrays(ref_structs[0])[1])
    else:
        out["reference"] = {"test": float("nan"), "train": float("nan"), "self_consistency": float("nan")}
    return out


class SelfCorr(Study):
    name = "self-corr"
    help = "Hi-C self-correlation, feed Hi-C as singletons, correlate vs held-out contacts"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--hic-singletons",
            choices=["replace", "augment"],
            default="replace",
            help="replace = pure Hi-C-driven (clean, MultiMM-comparable); augment = ChIA-PET + Hi-C "
            "(count-scaled to match), tests whether Hi-C improves the model",
        )
        p.add_argument("--chroms", default=None, help="comma-separated; default all in the breakpoints")
        p.add_argument(
            "--hic-breakpoints",
            default=None,
            help="precomputed Hi-C TAD breakpoints BED; default = derive from --hic (cooltools)",
        )
        p.add_argument("--n-regions", type=int, default=4)
        p.add_argument("--binsize", type=int, default=25000)
        p.add_argument("--min-ibs", type=int, default=2, help="minimum interaction-block count per region")
        p.add_argument("--max-ibs", type=int, default=6, help="maximum interaction-block count per region")
        p.add_argument("--max-mb", type=float, default=6.0, help="maximum region span in megabases")
        p.add_argument(
            "--holdout-frac", type=float, default=0.5, help="fraction of bin-pairs held out for testing"
        )
        p.add_argument("--seed", type=int, default=0, help="held-out split seed")
        p.add_argument(
            "--window", type=int, default=250_000, help="cooltools insulation window for TAD boundaries"
        )
        p.add_argument("--region", default=None, help="single region override")
        p.add_argument(
            "--no-reference", action="store_true", help="skip the reference + parity baselines"
        )
        p.add_argument(
            "--out",
            default=None,
            help="results JSON output path. Default out/self_corr_<cell>_<mode>.json",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        if ctx.hic is None:
            sys.exit("[error] self-corr requires --hic, an observed Hi-C .mcool")

        out_path = (
            Path(args.out)
            if args.out
            else Path(f"out/self_corr_{ctx.cell}_{args.hic_singletons}.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        chroms = (
            args.chroms.split(",")
            if args.chroms
            else ([args.region.split(":")[0]] if args.region else None)
        )

        # Hi-C TAD boundaries drive both region selection and the model segmentation. In self-HiC the
        # model's IBs align with the Hi-C TADs we score against instead of CTCF or CCD boundaries.
        if args.hic_breakpoints:
            hic_breakpoints = Path(args.hic_breakpoints)
        else:
            hic_breakpoints = out_path.with_name(f"{out_path.stem}__hic_tad_breaks.bed")
        if not hic_breakpoints.exists():
            write_breakpoints(ctx.hic, hic_breakpoints, chroms, args.window, args.binsize)

        if args.region:
            regions = [args.region]
        else:
            regions = enumerate_regions(
                str(hic_breakpoints), args.n_regions, chroms=chroms,
                min_ibs=args.min_ibs, max_ibs=args.max_ibs, max_mb=args.max_mb,
            )
        if not regions:
            sys.exit("[error] no regions found")

        tmp = Path(tempfile.mkdtemp(prefix="gnome3d_selfcorr_"))
        variant_names = ["reference", "parity", "tuned"]
        results = []
        for region in regions:
            print(f"\n[self_corr] {region} ({args.hic_singletons}, n={ctx.n}, {ctx.quality}):")
            r = _run_region(region, ctx, args, hic_breakpoints, tmp)
            results.append(r)
            out_path.write_text(json.dumps(results, indent=2))
            cells = "  ".join(f"{v}:{r[v]['train']:+.3f}/{r[v]['test']:+.3f}" for v in variant_names)
            print(f"  [done] (train/test)  {cells}")

        def _fin(r: dict, v: str, key: str) -> bool:
            return r[v][key] == r[v][key]  # not NaN

        def med(rows: list, variant: str, key: str) -> float:
            xs = [r[variant][key] for r in rows if _fin(r, variant, key)]
            return float(np.median(xs)) if xs else float("nan")

        # A NaN means a variant's structure collapsed on a region. Features-off models with no EV can
        # collapse on underdetermined regions, and then its per-variant median would be over a different
        # region set than tuned's. So also report the paired median over regions where all variants are
        # finite.
        matched = [r for r in results if all(_fin(r, v, "test") for v in variant_names)]
        print(f"\n{'=' * 74}\nHi-C self-correlation {args.hic_singletons}, {len(results)} regions")
        print(f"  {'variant':<14}{'train':>9}{'test':>9}{'consist(ρ)':>10}{'finite':>8}    {'test(paired)':>13}")
        for v in variant_names:
            nfin = sum(_fin(r, v, "test") for r in results)
            flag = "  collapsed on some regions" if nfin < len(results) else ""
            print(
                f"  {v:<14}{med(results, v, 'train'):>9.3f}{med(results, v, 'test'):>9.3f}"
                f"{med(results, v, 'self_consistency'):>10.3f}{nfin:>6}/{len(results)}    {med(matched, v, 'test'):>13.3f}{flag}"
            )
        print(f"  paired is the median over the {len(matched)} regions where all variants are finite")
        print("  test is held-out Hi-C generalisation. consist(ρ) is the self-consistency Spearman")
        print("         input Hi-C IF vs model 3D distance, want < 0, high-freq contacts placed close")
        print("  reference ≈ tuned means the ceiling is not our model. tuned ≫ parity means features help, EV stops collapse.")
        print("=" * 74)


register(SelfCorr())
