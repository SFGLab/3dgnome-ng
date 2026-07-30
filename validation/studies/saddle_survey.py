"""Survey the baseline model's compartmentalization across many regions.

The epigenome ablation found the baseline model's saddle strength wrong in both
directions on two regions of chr1, too weak on one and too strong on the other.
Two regions cannot tell a systematic calibration error from region-specific
noise, and until that is settled the compartment term cannot be tuned: a weight
raised to fix the first region makes the second worse.

So this runs one arm, the baseline, over many regions and asks two questions.

  * Is the gap between model and experimental saddle systematic, or does it
    average out?
  * What predicts it? The report correlates the gap against the region's
    experimental compartmentalization, the strength of its input compartment
    track, its bead count, its compaction and the density of its simulated
    contact map.

No energy term is enabled here and no weight is scanned. This measures the thing
those terms modify.

    python -m validation saddle-survey --cell GM12878 \\
        --hic data/_hic/GM12878/<file>.mcool --chroms chr1,chr2,chr5 -n 20

See docs/epigenome-energy-terms.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnome3d.types import F64Array
from validation.core import config as cfgmod
from validation.metrics import hic as contacts
from validation.studies import Context, Study, register
from validation.studies.epigenome import (
    _arm_flags,
    _ib_ids,
    _run_arm,
    _track_on_bins,
    _track_paths,
)

# Region-level columns correlated against the gap, and what each one would mean.
# Each is a candidate explanation for why a region's model compartmentalization
# misses. They are reported together because they are not independent: a compact
# region has both a high density and a low Rg.
_PREDICTORS: dict[str, str] = {
    "exp_sad": "experimental saddle. a correlation here means the model regresses to a fixed level",
    "trk_sd": "spread of the input compartment track. weak input, weak segregation",
    "a_frac": "fraction of A bins. an unbalanced region has a small minority block",
    "n_beads": "bead count. a size effect points at the MC schedule, not the physics",
    "rg": "radius of gyration. compaction confounds the contact map",
    "density": "simulated map density. a saturated map cannot express enrichment",
    "ib_over_obs": "how much more block-organized the model is than the experiment",
}


def _chromsizes(mcool_path: str) -> dict[str, int]:
    """Chromosome lengths from the cooler, used to place windows."""
    import cooler

    paths = [p for p in cooler.fileops.list_coolers(mcool_path) if p.rsplit("/", 1)[-1].isdigit()]
    uri = f"{mcool_path}::{max(paths, key=lambda p: int(p.rsplit('/', 1)[-1]))}" if paths else None
    c = cooler.Cooler(uri) if uri else cooler.Cooler(mcool_path)
    return {str(k): int(v) for k, v in c.chromsizes.items()}


def _windows(sizes: dict[str, int], chroms: list[str], width: int, per_chrom: int) -> list[str]:
    """Evenly spaced windows per chromosome, as region strings.

    Placed inside the middle 90 percent so a window does not run off a telomere.
    Windows that land on a centromere are not avoided here; they produce an
    unusable contact map and are dropped by the run loop, which reports the skip.
    """
    out: list[str] = []
    for chrom in chroms:
        length = sizes.get(chrom, 0)
        if length < width:
            continue
        lo, hi = int(0.05 * length), int(0.95 * length) - width
        if hi <= lo:
            lo, hi = 0, length - width
        starts = [lo] if per_chrom == 1 else np.linspace(lo, hi, per_chrom).astype(int)
        for st in np.atleast_1d(starts):
            out.append(f"{chrom}:{int(st)}-{int(st) + width}")
    return out


def _report_correlations(rows: list[dict[str, float]]) -> None:
    """Correlate the gap against each predictor and print the table.

    Both Pearson and Spearman: with 20 or so regions a single outlier can carry a
    Pearson, and a predictor that only holds monotonically still counts.
    """
    from scipy.stats import pearsonr, spearmanr

    gap = np.array([r["gap"] for r in rows])
    if len(gap) < 4:
        print("\n  too few usable regions to correlate")
        return

    print(f"\n  what predicts the gap  (n={len(gap)} regions)")
    print(f"  {'predictor':<10}{'pearson r':>11}{'p':>9}{'spearman':>10}{'p':>9}   meaning")
    print("  " + "-" * 100)
    for key, meaning in _PREDICTORS.items():
        x = np.array([r[key] for r in rows])
        if np.allclose(x, x[0]):
            continue
        pr = pearsonr(x, gap)
        sr = spearmanr(x, gap)
        star = " *" if min(float(pr[1]), float(sr[1])) < 0.05 else "  "
        print(
            f"  {key:<10}{float(pr[0]):>11.3f}{float(pr[1]):>9.3f}"
            f"{float(sr[0]):>10.3f}{float(sr[1]):>9.3f}{star} {meaning}"
        )


def _summarize(rows: list[dict[str, float]]) -> None:
    """The systematic-versus-noise verdict.

    A sign test rather than a t-test on the gap: the question is whether the model
    misses in a consistent direction, and the magnitude distribution is not assumed
    to be anything in particular.
    """
    from scipy.stats import binomtest

    gap = np.array([r["gap"] for r in rows])
    n_over = int((gap > 0).sum())
    n = len(gap)
    p = float(binomtest(n_over, n, 0.5).pvalue) if n else float("nan")

    print(f"\n  gap = model saddle - experimental saddle,  n={n} regions")
    print(f"    mean {gap.mean():+.3f}   median {np.median(gap):+.3f}   sd {gap.std(ddof=1):.3f}")
    print(f"    model over-compartmentalized in {n_over}/{n} regions   sign test p={p:.3f}")
    print(f"    mean absolute gap {np.abs(gap).mean():.3f}")
    if p < 0.05:
        d = "over" if n_over * 2 > n else "under"
        print(f"    -> a consistent {d}-compartmentalization, not region noise")
    else:
        print("    -> no consistent direction. the error is region-specific, and a single")
        print("       global weight cannot correct it")


class SaddleSurvey(Study):
    name = "saddle-survey"
    help = "baseline saddle versus experimental across many regions, and what predicts the gap"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--regions", default=None, help="comma list of explicit regions; overrides --chroms"
        )
        p.add_argument("--chroms", default="chr1,chr2,chr5,chr10,chr14,chr17")
        p.add_argument("--width", type=int, default=12_000_000)
        p.add_argument("--per-chrom", type=int, default=3)
        p.add_argument("--binsize", type=int, default=100_000)
        p.add_argument(
            "--min-bins",
            type=int,
            default=40,
            help="skip a region whose experimental saddle uses fewer bins than this",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        from gnome3d.io import parse_chrs_arg

        comp_path, acc_path = _track_paths(ctx.cell, ctx.data_root)
        if not Path(comp_path).exists():
            print(f"[saddle-survey] missing compartment track: {comp_path}")
            print(f"[saddle-survey] build it: python -m validation tracks --cell {ctx.cell}")
            return
        if not ctx.hic:
            print("[saddle-survey] --hic is required; it supplies the experimental saddle")
            return

        regions = (
            [r.strip() for r in args.regions.split(",") if r.strip()]
            if args.regions
            else _windows(_chromsizes(ctx.hic), args.chroms.split(","), args.width, args.per_chrom)
        )

        print(f"saddle survey  {ctx.cell}  {len(regions)} regions  n={ctx.n}  baseline arm only")
        print(f"  compartments: {comp_path}")
        print(f"  scored against {Path(ctx.hic).name} @ {args.binsize // 1000}kb\n")
        header = (
            f"  {'region':<26}{'exp':>7}{'model':>8}{'gap':>8}"
            f"{'ibE mdl':>9}{'ibE obs':>9}{'ratio':>7}"
            f"{'beads':>8}{'Rg':>8}{'dens':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        flags = _arm_flags("off", args, comp_path, acc_path)
        rows: list[dict[str, float]] = []
        for region in regions:
            try:
                chrs, bed = parse_chrs_arg(region)
                c_obs, bin_starts = contacts.observed_hic(
                    ctx.hic, region, args.binsize, balance=True
                )
                track: F64Array = _track_on_bins(comp_path, chrs[0], bin_starts)
                obs = contacts.compartment_saddle(c_obs, track)
                if not np.isfinite(obs["strength"]) or obs["n_bins"] < args.min_bins:
                    print(f"  {region:<26}  skipped: {int(obs['n_bins'])} usable bins")
                    continue

                blocks = _ib_ids(
                    cfgmod.settings_for_cell(ctx.cell, ctx.data_root, ctx.quality),
                    chrs,
                    bed,
                    bin_starts,
                )
                ib_obs = contacts.block_enrichment(c_obs, blocks)["ratio"]
                row = _run_arm(
                    ctx, args, flags, chrs, bed, c_obs, bin_starts, track, block_id=blocks
                )
                if not np.isfinite(row["saddle"]):
                    print(f"  {region:<26}  skipped: model saddle is nan (map saturated?)")
                    continue

                used = track[track != 0.0]
                rec = {
                    "exp_sad": float(obs["strength"]),
                    "model": float(row["saddle"]),
                    "gap": float(row["saddle"] - obs["strength"]),
                    "trk_sd": float(used.std()) if used.size else 0.0,
                    "a_frac": float((used > 0).mean()) if used.size else 0.0,
                    "n_beads": row["n_beads"],
                    "rg": row["rg"],
                    "density": row["density"],
                    "ib_model": float(row["ib_ratio"]),
                    "ib_obs": float(ib_obs),
                    "ib_over_obs": float(row["ib_ratio"] / ib_obs) if ib_obs > 0 else float("nan"),
                    "ib_between_zero": float(row["ib_ratio"] == float("inf")),
                }
                rows.append(rec)
                print(
                    f"  {region:<26}{rec['exp_sad']:>7.3f}{rec['model']:>8.3f}{rec['gap']:>+8.3f}"
                    f"{rec['ib_model']:>9.3f}{rec['ib_obs']:>9.3f}{rec['ib_over_obs']:>7.2f}"
                    f"{rec['n_beads']:>8.0f}{rec['rg']:>8.2f}{rec['density']:>7.2f}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {region:<26}  ERROR: {type(e).__name__}: {e}")

        if not rows:
            print("\n  no usable regions")
            return
        _summarize(rows)
        _report_correlations(rows)
        print(
            "\n  ibE is within-block over between-block contact enrichment on the O/E map,\n"
            "  for the model and for the experiment, and ratio is the first over the second.\n"
            "  A ratio well above 1 means the structure is organized by the interaction blocks\n"
            "  it was built from rather than by the data, which reads as compartmentalization\n"
            "  because compartment identity runs in long blocks along the genome.\n"
            "\n  A gap near zero everywhere would mean the baseline already reproduces\n"
            "  compartmentalization and the terms have nothing to add. A consistent sign\n"
            "  means one global weight can correct it. Mixed signs mean it cannot, and\n"
            "  whichever predictor carries the gap is where to look next."
        )


register(SaddleSurvey())
