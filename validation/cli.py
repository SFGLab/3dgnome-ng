"""Command-line plumbing. Shared args, the Context builder, and the run battery."""

from __future__ import annotations

import importlib
from argparse import ArgumentParser, Namespace

from validation.studies import Context, all_studies

STUDY_MODULES = [
    "validation.studies.compare",
    "validation.studies.synthetic",
    "validation.studies.self_corr",
    "validation.studies.model_hic",
    "validation.studies.sweep",
    "validation.studies.tune",
    "validation.studies.prove",
    "validation.studies.report",
    "validation.studies.boundaries",
    "validation.studies.fetch",
]
BATTERY = ["compare", "synthetic", "self-corr", "model-hic"]

# Shared arg names copied from a battery invocation onto each study's own defaulted
# namespace in run_battery. Matches the dest names add_shared_args registers.
_SHARED_OVERRIDE_FIELDS = [
    "cell",
    "data_root",
    "hic",
    "quality",
    "n_structures",
    "py_arcs",
    "py_workers",
    "ref_workers",
    "fast",
]


def load_studies() -> None:
    for m in STUDY_MODULES:
        importlib.import_module(m)


def add_shared_args(p: ArgumentParser) -> None:
    p.add_argument("--cell", default="GM12878")
    p.add_argument("--data-root", default="data")
    p.add_argument("--hic", default=None)
    p.add_argument("--quality", default=None, choices=["fast", "balanced", "full"])
    p.add_argument("-n", "--n-structures", type=int, default=100)
    p.add_argument("--py-arcs", default="batch", choices=["batch", "threaded", "serial"])
    p.add_argument("--py-workers", type=int, default=0)
    p.add_argument("--ref-workers", type=int, default=0)
    p.add_argument("--fast", action="store_true")


def context_from(args: Namespace) -> Context:
    return Context(
        cell=args.cell,
        data_root=args.data_root,
        hic=getattr(args, "hic", None),
        quality=("fast" if args.fast else args.quality),
        n=args.n_structures,
        py_arcs=args.py_arcs,
        py_workers=args.py_workers,
        ref_workers=args.ref_workers,
        fast=args.fast,
    )


def run_battery(args: Namespace) -> None:
    """Run the standard battery, compare synthetic self-corr model-hic.

    Each battery study needs its own study-specific args, not just the shared
    ones the run subcommand exposes. For every study in BATTERY this builds a
    fresh parser with add_shared_args followed by that study's own add_args,
    parses an empty argument list to get a complete namespace of defaults,
    then overwrites the shared fields on that namespace with whatever the
    user actually passed to run. The result is a namespace each study can
    read exactly as if it had been invoked directly, with the user's shared
    overrides and its own defaults for everything else.
    """
    studies = all_studies()
    for name in BATTERY:
        print(f"\n===== {name} =====")
        study = studies[name]
        p = ArgumentParser()
        add_shared_args(p)
        study.add_args(p)
        study_args = p.parse_args([])
        for field in _SHARED_OVERRIDE_FIELDS:
            setattr(study_args, field, getattr(args, field))
        ctx = context_from(study_args)
        study.run(ctx, study_args)
