"""Resolve one array index to `chrom sample first last total`, for the job script.

A separate entry point so slurm/ensemble/trio_ensemble.sh and playground/trio/trio_status.py share
one mapping instead of each carrying its own arithmetic. Exits 1 when the index is past the end.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--chroms", default="")
    ap.add_argument("--samples", default="")
    args = ap.parse_args()

    chroms = [c for c in args.chroms.split(",") if c] or list(trio_samples.CHROMS)
    samples = trio_samples.resolve(args.samples or None)
    got = trio_samples.shard(args.task, chroms, samples, args.n_models, args.per_task)
    total = trio_samples.array_length(chroms, samples, args.n_models, args.per_task)
    if got is None:
        print(f"task {args.task} past the end; submit --array=0-{total - 1}", file=sys.stderr)
        raise SystemExit(1)
    chrom, sample, first, last = got
    print(f"{chrom} {sample.name} {first} {last} {total}")


if __name__ == "__main__":
    main()
