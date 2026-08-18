"""Prove a divergence makes sense by running flags off versus flags on over the same data, then
comparing. The headline test for excluded volume and confinement is that they reduce the
physically impossible overlaps the 2016 paper admitted, without degrading self-consistency or the
polymer scaling laws.

    python -m validation prove --cell GM12878 \\
        --region chr1:18288319-20307135 -n 5 --prove ev

--prove {ev, confinement, dynamic, all}

Settings are wired from canonical per-cell params by core.config.settings_for_cell, selected with
the shared --cell, --data-root and --quality.
"""

from __future__ import annotations

import argparse

import numpy as np

from gnome3d import log
from gnome3d.data import ContactData
from gnome3d.types import BeadOut
from validation import metrics
from validation.core.config import apply_flags, settings_for_cell
from validation.core.data import load_contacts
from validation.core.ensemble import run_ensemble, summarize
from validation.core.regions import parse_region_arg
from validation.core.report import print_comparison
from validation.studies import Context, Study, register

# Which settings attributes each --prove target toggles. These are smooth-stage gates, where the
# divergences actually act. Baseline forces them all False, treatment forces them True.
PROVE_FLAGS: dict[str, dict[str, bool]] = {
    "ev": {"use_excluded_volume": True, "exclusion_apply_to_smooth": True},
    "confinement": {"use_confinement": True, "confinement_apply_to_smooth": True},
    "dynamic": {"use_dynamic_loop_density": True},
}


def _radius_for(ensemble: list[list[BeadOut]], fixed: float | None) -> float:
    if fixed is not None:
        return fixed
    coords, _ = metrics.to_arrays(ensemble[0])
    return float(np.median(metrics.bond_lengths(coords)))


class Prove(Study):
    name = "prove"
    help = "run a divergence flags off against on and judge it"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", required=True, help="chr:start-end or chr name")
        p.add_argument(
            "--prove",
            required=True,
            choices=["ev", "confinement", "dynamic", "all"],
            help="comparison mode, run flags off against on and judge the divergence",
        )
        p.add_argument(
            "--contact-radius",
            type=float,
            default=None,
            help="overlap / contact-prob radius (default: median baseline bond length)",
        )
        p.add_argument(
            "--skip-neighbors",
            type=int,
            default=1,
            help="exclude |i-j|<=this as bonded (default 1)",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        chrs_list, region = parse_region_arg(args.region)

        base_settings = settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
        log.setup(base_settings.output_level, log_file=base_settings.log_file or None)

        # Load contacts once, independent of the toggled physics flags.
        data = ContactData.from_files(base_settings, chrs_list, region)
        contacts = load_contacts(base_settings, chrs_list, region)
        print(f"[prove] region={args.region}  ensemble={ctx.n}  contacts={len(contacts)}")

        targets = ["ev", "confinement", "dynamic"] if args.prove == "all" else [args.prove]
        # Baseline is all proved flags off. Build once and reuse its radius for both runs.
        off = {a: False for t in targets for a in PROVE_FLAGS[t]}
        base_s = apply_flags(base_settings, off)
        base_ens = run_ensemble(base_s, data, chrs_list, region, ctx.n)
        radius = _radius_for(base_ens, args.contact_radius)
        print(f"[prove] contact/overlap radius = {radius:.3f}")
        base_m = summarize(base_ens, contacts, radius, args.skip_neighbors)

        all_ok = True
        for target in targets:
            treat_s = apply_flags(base_s, PROVE_FLAGS[target])
            treat_ens = run_ensemble(treat_s, data, chrs_list, region, ctx.n)
            treat_m = summarize(treat_ens, contacts, radius, args.skip_neighbors)
            all_ok &= print_comparison(target, base_m, treat_m)

        raise SystemExit(0 if all_ok else 1)


register(Prove())
