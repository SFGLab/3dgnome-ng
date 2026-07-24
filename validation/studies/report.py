"""Report a single cell's ensemble.

Drives the public gnome3d modelling API, Settings and ContactData and simulate, exactly as a user
would, then scores the output ensemble with the resolution-independent metrics in
validation/metrics/structure.py. See docs/validation.md.

    python -m validation report --cell GM12878 \\
        --region chr1:18288319-20307135 -n 5

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
from validation.core.config import settings_for_cell
from validation.core.data import load_contacts
from validation.core.ensemble import run_ensemble, summarize
from validation.core.regions import parse_region_arg
from validation.core.report import print_single
from validation.studies import Context, Study, register


def _radius_for(ensemble: list[list[BeadOut]], fixed: float | None) -> float:
    if fixed is not None:
        return fixed
    coords, _ = metrics.to_arrays(ensemble[0])
    return float(np.median(metrics.bond_lengths(coords)))


class Report(Study):
    name = "report"
    help = "run one ensemble and report its metrics"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--region", required=True, help="chr:start-end or chr name")
        p.add_argument(
            "--contact-radius",
            type=float,
            default=None,
            help="overlap / contact-prob radius (default: median baseline bond length)",
        )
        p.add_argument(
            "--skip-neighbors", type=int, default=1, help="exclude |i-j|<=this as bonded (default 1)"
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        chrs_list, region = parse_region_arg(args.region)

        settings = settings_for_cell(ctx.cell, ctx.data_root, ctx.quality)
        log.setup(settings.output_level, log_file=settings.log_file or None)

        data = ContactData.from_files(settings, chrs_list, region)
        contacts = load_contacts(settings, chrs_list, region)
        print(f"[report] region={args.region}  ensemble={ctx.n}  contacts={len(contacts)}")

        ens = run_ensemble(settings, data, chrs_list, region, ctx.n)
        radius = _radius_for(ens, args.contact_radius)
        print(f"[report] contact/overlap radius = {radius:.3f}")
        print_single(ctx.cell, summarize(ens, contacts, radius, args.skip_neighbors))


register(Report())
