"""The single entry point for the validation harness. python -m validation <study> [opts]."""

from __future__ import annotations

import argparse

from validation.cli import add_shared_args, context_from, load_studies, run_battery
from validation.studies import all_studies


def main(argv: list[str] | None = None) -> None:
    load_studies()
    parser = argparse.ArgumentParser(prog="validation", description="3dgnome validation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    studies = all_studies()
    for name, study in studies.items():
        sp = sub.add_parser(name, help=study.help)
        add_shared_args(sp)
        study.add_args(sp)
    runp = sub.add_parser(
        "run", help="run the standard battery, compare synthetic self-corr model-hic"
    )
    add_shared_args(runp)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run_battery(args)
        return
    ctx = context_from(args)
    studies[args.cmd].run(ctx, args)


if __name__ == "__main__":
    main()
