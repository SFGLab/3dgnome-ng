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

from validation.core import config as cfgmod
from validation.core import ensemble as ens_mod
from validation.metrics import hic as contacts
from validation.metrics import structure as smetrics
from validation.studies import Context, Study, register

# Each arm names the flags it turns on.  "off" is the baseline every other arm is
# compared against.  Excluded volume is on everywhere except the baseline, because
# the affinity terms are attractive and need something pushing back.
ARMS: dict[str, dict[str, object]] = {
    "off": {},
    "ev-only": {
        "use_excluded_volume": True,
        "exclusion_apply_to_smooth": True,
    },
    "compartments": {
        "use_excluded_volume": True,
        "exclusion_apply_to_smooth": True,
        "use_compartments": True,
    },
    "bridging": {
        "use_excluded_volume": True,
        "exclusion_apply_to_smooth": True,
        "use_bridging": True,
    },
    "fibre": {
        "use_excluded_volume": True,
        "exclusion_apply_to_smooth": True,
        "use_fibre_compaction": True,
    },
    "all": {
        "use_excluded_volume": True,
        "exclusion_apply_to_smooth": True,
        "use_compartments": True,
        "use_bridging": True,
        "use_fibre_compaction": True,
    },
}


def _track_paths(cell: str, data_root: str) -> tuple[str, str]:
    d = Path(data_root) / cell
    return (
        str(d / f"{cell}_compartments.bedGraph"),
        str(d / f"{cell}_atac.bedGraph"),
    )


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

        print(f"epigenome ablation  {ctx.cell}  {args.region}  n={ctx.n}")
        print(f"  compartments: {comp_path}")
        print(f"  accessibility: {acc_path}")
        print(f"  scored against {Path(ctx.hic).name} @ {args.binsize // 1000}kb\n")
        header = f"  {'arm':<14}{'eig |r|':>9}{'agree':>8}{'Rg':>9}{'bondCV':>9}{'overlap':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        base: dict[str, float] = {}
        for name in args.arms.split(","):
            name = name.strip()
            if name not in ARMS:
                print(f"  {name:<14}  unknown arm")
                continue
            flags = dict(ARMS[name])
            if flags.get("use_compartments"):
                flags["compartment_weight"] = args.compartment_weight
            if flags.get("use_bridging"):
                flags["bridging_weight"] = args.bridging_weight
            if flags.get("use_fibre_compaction"):
                flags["fibre_compaction"] = args.fibre
            # The tracks are always loaded; only the flags decide whether a term reads them.
            flags["data_compartments"] = comp_path
            flags["data_accessibility"] = acc_path

            try:
                s = cfgmod.settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
                s = cfgmod.apply_flags(s, flags)
                from gnome3d.data import ContactData

                data = ContactData.from_files(s, chrs, region)
                ens = ens_mod.run_ensemble(s, data, chrs, region, ctx.n)
                cl, ml = ens_mod.to_arrays_list(ens)

                radius = float(np.median(smetrics.bond_lengths(cl[0])))
                c_sim = np.zeros_like(c_obs)
                for coords, mids in zip(cl, ml, strict=True):
                    c_sim += contacts.simulated_contacts(
                        coords, mids, bin_starts, args.binsize, radius
                    )
                cc = contacts.compartment_correlation(c_sim, c_obs)

                rg = float(np.mean([smetrics.radius_of_gyration(c) for c in cl]))
                bl = smetrics.bond_lengths(cl[0])
                cv = float(bl.std() / bl.mean()) if bl.mean() > 0 else float("nan")
                ov = float(smetrics.overlap_fraction(cl[0]))

                row = {"eig": cc["eig_pearson_abs"], "agree": cc["agreement"], "rg": rg}
                if name == "off":
                    base = row
                mark = ""
                if base and name != "off":
                    d_eig = row["eig"] - base["eig"]
                    d_rg = (rg - base["rg"]) / base["rg"] if base["rg"] else 0.0
                    mark = f"   eig {d_eig:+.3f}  Rg {d_rg * 100:+.0f}%"
                print(
                    f"  {name:<14}{cc['eig_pearson_abs']:>9.3f}{cc['agreement']:>8.3f}"
                    f"{rg:>9.2f}{cv:>9.3f}{ov:>9.3f}{mark}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<14}  ERROR: {type(e).__name__}: {e}")

        print(
            "\n  eig |r| is the absolute correlation of the structure's compartment eigenvector\n"
            "  with the experimental one; agree is the fraction of bins put in the same\n"
            "  compartment. A gain there only counts if Rg and bondCV hold: these terms are\n"
            "  attractive and can raise the score by collapsing the structure."
        )


register(Epigenome())
