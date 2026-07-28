"""Ablate the epigenome energy terms against real Hi-C.

Reconstructs one region several times, each with a different subset of the
compartment and accessibility terms enabled, and reports for each:

  * compartment eigenvector agreement with that cell line's own Hi-C, the
    question the terms exist to improve.  This is MultiMM's second validation.
  * radius of gyration, bond-length spread and overlap fraction, the polymer
    sanity numbers that must not regress while the compartment score improves.

The terms are purely attractive, so a run that improves compartment agreement by
collapsing the structure has not improved anything.  Reporting both together is
the point of this study.

Needs tracks built first:

    python -m validation fetch  --manifest validation/manifests/<CELL>_hic.json --out data/_hic
    python -m validation fetch  --manifest validation/manifests/<CELL>_accessibility.json \\
                                --out data/_epigenome
    python -m validation tracks --cell <CELL>
    python -m validation epigenome --cell <CELL> --region chr1:20000000-40000000

See docs/epigenome-energy-terms.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnome3d.types import BedRegion, F64Array, I64Array
from validation.core import config as cfgmod
from validation.core import ensemble as ens_mod
from validation.metrics import hic as contacts
from validation.metrics import structure as smetrics
from validation.studies import Context, Study, register

# Each arm names the flags it turns on, on top of the canonical config.
#
# CANONICAL already enables excluded volume and confinement, so the baseline is
# not a bare polymer and there is no point in an EV-only arm. That matters here:
# the affinity terms are attractive and need that repulsion to push back against.
ARMS: dict[str, dict[str, object]] = {
    "off": {},
    "compartments": {"use_compartments": True},
    "bridging": {"use_bridging": True},
    "fibre": {"use_fibre_compaction": True},
    "all": {
        "use_compartments": True,
        "use_bridging": True,
        "use_fibre_compaction": True,
    },
}


def _track_on_bins(comp_path: str, chrom: str, bin_starts: I64Array) -> F64Array:
    """The input compartment track sampled onto the Hi-C bin grid.

    Used to sort bins for the saddle statistic. The same track is used for every
    arm so the quantile definition is fixed, which makes a change in enrichment a
    change in the structure rather than in the binning.
    """
    vals: dict[int, float] = {}
    with open(comp_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0] != chrom:
                continue
            try:
                vals[int(parts[1])] = float(parts[3])
            except ValueError:
                continue
    if not vals:
        return np.zeros(len(bin_starts), dtype=np.float64)
    # A coarser track (H1ESC is 250kb against a 100kb grid) needs the covering
    # interval, not an exact start match, or every bin reads as unassigned.
    keys = np.array(sorted(vals))
    out = np.zeros(len(bin_starts), dtype=np.float64)
    for i, s in enumerate(bin_starts):
        j = int(np.searchsorted(keys, s, side="right")) - 1
        if j >= 0:
            out[i] = vals[int(keys[j])]
    return out


def _track_paths(cell: str, data_root: str) -> tuple[str, str]:
    """Absolute paths.  `Settings.data_path` joins a relative name onto `data_dir`,
    which for these tracks is already `<data_root>/<cell>`, so a repo-relative path
    would resolve to `data/<cell>/data/<cell>/...` and silently load nothing."""
    d = (Path(data_root) / cell).resolve()
    return (
        str(d / f"{cell}_compartments.bedGraph"),
        str(d / f"{cell}_atac.bedGraph"),
    )


def _run_arm(
    ctx: Context,
    args: argparse.Namespace,
    flags: dict[str, object],
    chrs: list[str],
    region: BedRegion | None,
    c_obs: F64Array,
    bin_starts: I64Array,
    sort_track: F64Array,
) -> dict[str, float]:
    """Reconstruct one arm and score it. Returns the metric row.

    Raises rather than returning a partial row: an arm that silently reports the
    baseline because its track failed to load is worse than a visible error.
    """
    from gnome3d.data import ContactData

    s = cfgmod.settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
    s = cfgmod.apply_flags(s, flags)
    data = ContactData.from_files(s, chrs, region)
    if flags.get("use_compartments") and not data.compartments:
        raise RuntimeError(f"no compartment intervals loaded from {flags.get('data_compartments')}")
    if flags.get("use_bridging") and not data.accessibility:
        raise RuntimeError(f"no accessibility bins loaded from {flags.get('data_accessibility')}")

    ens = ens_mod.run_ensemble(s, data, chrs, region, ctx.n)
    cl, ml = ens_mod.to_arrays_list(ens)

    radius = float(np.median(smetrics.bond_lengths(cl[0])))
    c_sim = np.zeros_like(c_obs)
    for coords, mids in zip(cl, ml, strict=True):
        c_sim += contacts.simulated_contacts(coords, mids, bin_starts, args.binsize, radius)
    cc = contacts.compartment_correlation(c_sim, c_obs)
    sad = contacts.compartment_saddle(c_sim, sort_track)

    bl = smetrics.bond_lengths(cl[0])
    return {
        "saddle": sad["strength"],
        "eig": cc["eig_pearson_abs"],
        "agree": cc["agreement"],
        "rg": float(np.mean([smetrics.radius_of_gyration(c) for c in cl])),
        "cv": float(bl.std() / bl.mean()) if bl.mean() > 0 else float("nan"),
        "overlap": float(smetrics.overlap_fraction(cl[0], radius)[0]),
    }


def _arm_flags(
    name: str, args: argparse.Namespace, comp_path: str, acc_path: str
) -> dict[str, object]:
    flags = dict(ARMS[name])
    if flags.get("use_compartments"):
        flags["compartment_weight"] = args.compartment_weight
    if flags.get("use_bridging"):
        flags["bridging_weight"] = args.bridging_weight
    if flags.get("use_fibre_compaction"):
        flags["fibre_compaction"] = args.fibre
    # The tracks are always pointed at; only the flags decide whether a term reads them.
    flags["data_compartments"] = comp_path
    flags["data_accessibility"] = acc_path
    return flags


class Epigenome(Study):
    name = "epigenome"
    help = "ablate the compartment and accessibility terms against real Hi-C"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", default="chr1:20000000-40000000")
        p.add_argument(
            "--binsize", type=int, default=100_000, help="Hi-C resolution for the eigenvector"
        )
        p.add_argument("--arms", default=",".join(ARMS), help="comma list of arm names")
        p.add_argument("--compartment-weight", type=float, default=2.0)
        p.add_argument("--bridging-weight", type=float, default=1.0)
        p.add_argument("--fibre", type=float, default=0.2)
        p.add_argument(
            "--baseline-repeats",
            type=int,
            default=1,
            help="run the off arm this many times to measure the noise floor; "
            "an effect smaller than that spread is not a result",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        from gnome3d.io import parse_chrs_arg

        comp_path, acc_path = _track_paths(ctx.cell, ctx.data_root)
        for path, what in ((comp_path, "compartments"), (acc_path, "accessibility")):
            if not Path(path).exists():
                print(f"[epigenome] missing {what} track: {path}")
                print(f"[epigenome] build it: python -m validation tracks --cell {ctx.cell}")
                return
        if not ctx.hic:
            print("[epigenome] --hic is required; it is the target the arms are scored against")
            return

        chrs, region = parse_chrs_arg(args.region)
        c_obs, bin_starts = contacts.observed_hic(ctx.hic, args.region, args.binsize, balance=True)
        sort_track = _track_on_bins(comp_path, chrs[0], bin_starts)
        obs_saddle = contacts.compartment_saddle(c_obs, sort_track)

        print(f"epigenome ablation  {ctx.cell}  {args.region}  n={ctx.n}")
        print(f"  compartments: {comp_path}")
        print(f"  accessibility: {acc_path}")
        print(f"  scored against {Path(ctx.hic).name} @ {args.binsize // 1000}kb")
        print(
            f"  experimental saddle = {obs_saddle['strength']:.3f} "
            f"over {int(obs_saddle['n_bins'])} bins  (1.0 = no compartmentalization)\n"
        )
        header = (
            f"  {'arm':<14}{'saddle':>9}{'eig |r|':>9}{'agree':>8}"
            f"{'Rg':>9}{'bondCV':>9}{'overlap':>9}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Measuring the detection floor is not optional.  An earlier run reported
        # treatment effects of 0.003 to 0.010 while three identical baselines spanned
        # 0.039, so every one of those "effects" was noise.  Repeating the baseline
        # here makes that visible in the same table instead of needing three separate
        # invocations to notice.
        floor: float | None = None
        if args.baseline_repeats > 1:
            off_flags = _arm_flags("off", args, comp_path, acc_path)
            eigs = [
                _run_arm(ctx, args, off_flags, chrs, region, c_obs, bin_starts, sort_track)["eig"]
                for _ in range(args.baseline_repeats)
            ]
            if len(eigs) > 1:
                floor = float(np.std(eigs, ddof=1))
                print(
                    f"  {'off x' + str(len(eigs)):<14}"
                    f"{np.mean(eigs):>9.3f}{'':>8}{'':>9}{'':>9}{'':>9}"
                    f"   noise floor sd={floor:.3f}  range={max(eigs) - min(eigs):.3f}"
                )

        base: dict[str, float] = {}
        for name in args.arms.split(","):
            name = name.strip()
            if name not in ARMS:
                print(f"  {name:<14}  unknown arm")
                continue
            flags = _arm_flags(name, args, comp_path, acc_path)
            try:
                row = _run_arm(ctx, args, flags, chrs, region, c_obs, bin_starts, sort_track)
                if name == "off":
                    base = row
                mark = ""
                if base and name != "off":
                    d_sad = row["saddle"] - base["saddle"]
                    d_eig = row["eig"] - base["eig"]
                    d_rg = (row["rg"] - base["rg"]) / base["rg"] if base["rg"] else 0.0
                    mark = f"   saddle {d_sad:+.3f}  eig {d_eig:+.3f}  Rg {d_rg * 100:+.0f}%"
                    if floor:
                        sigma = abs(d_eig) / floor if floor > 0 else 0.0
                        mark += f"  ({sigma:.1f} sd eig)" + ("" if sigma >= 2.0 else " = noise")
                print(
                    f"  {name:<14}{row['saddle']:>9.3f}{row['eig']:>9.3f}{row['agree']:>8.3f}"
                    f"{row['rg']:>9.2f}{row['cv']:>9.3f}{row['overlap']:>9.3f}{mark}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<14}  ERROR: {type(e).__name__}: {e}")

        print(
            "\n  saddle is the A-A plus B-B contact enrichment over A-B, with bins sorted by the\n"
            "  input compartment track. 1.0 means no compartmentalization. It is the direct\n"
            "  measure of what the compartment term acts on, and unlike the eigenvector\n"
            "  correlation it does not depend on the region's own baseline architecture.\n"
            "  eig |r| is the absolute correlation of the structure's compartment eigenvector\n"
            "  with the experimental one; agree is the fraction of bins put in the same\n"
            "  compartment. A gain there only counts if Rg and bondCV hold: these terms are\n"
            "  attractive and can raise the score by collapsing the structure.\n"
            "  Pass --baseline-repeats to print the noise floor; an effect under 2 sd of it\n"
            "  is not a result."
        )


register(Epigenome())
