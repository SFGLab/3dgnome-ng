"""Write Hi-C bin pairs as a singleton BEDPE the model can ingest.

Every pair with a count is written. The train/test split that
``validation.studies.self_corr`` applies is a validation device, and holding half the contacts
back would only weaken a production ensemble, so ``holdout=False`` here.

One chromosome at a time even when the whole genome is asked for. The underlying reader
materialises a dense matrix per region, and a genome-wide one at 25 kb would be far larger than
memory, while chr1 alone is about 0.8 GB and fits. Only intra-chromosomal pairs are written,
which is what the model consumes for per-chromosome reconstruction.

    python slurm/ensemble/prep_singletons.py \
        --mcool data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \
        --chroms chr1-chr22,chrX --binsize 25000 \
        --out data/GM12878/GM12878_hic_25kb_singletons.bedpe
"""

import argparse
import sys
import tempfile
from pathlib import Path

# pyproject installs gnome3d* only, so validation/ is importable from the repo root and nowhere
# else. Running this file by path puts its own directory on sys.path rather than the root, so
# the root goes on explicitly and the script works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gnome3d.io import parse_chrs_arg  # noqa: E402
from validation.studies.self_corr import hic_to_singleton_bedpe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcool", required=True)
    ap.add_argument("--region", help="single region, e.g. chr1 or chr1:0-40000000")
    ap.add_argument(
        "--chroms", help="chromosome set, e.g. chr1-chr22,chrX (same syntax as --region)"
    )
    ap.add_argument("--binsize", type=int, default=25000)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Drop pairs below this raw count. Raise it on a deep map to cut file size.",
    )
    args = ap.parse_args()

    if bool(args.region) == bool(args.chroms):
        sys.exit("give exactly one of --region or --chroms")

    out = Path(args.out)
    if out.exists():
        print(f"[prep] {out} exists, leaving it alone")
        return

    regions = [args.region] if args.region else parse_chrs_arg(args.chroms)[0]
    out.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file and moved into place at the end, so an interrupted run cannot
    # leave a truncated BEDPE that the job guard would happily accept.
    total = 0
    with tempfile.NamedTemporaryFile("w", dir=out.parent, suffix=".partial", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("w") as sink:
            for region in regions:
                part = tmp_path.with_suffix(f".{region}")
                try:
                    _bal, starts, _test, _train = hic_to_singleton_bedpe(
                        args.mcool,
                        region,
                        args.binsize,
                        part,
                        holdout=False,
                        min_count=args.min_count,
                    )
                except Exception as exc:  # noqa: BLE001 - a chromosome absent from the cooler
                    print(f"[prep] {region}: skipped ({exc})")
                    continue
                rows = 0
                with part.open() as fh:
                    for line in fh:
                        sink.write(line)
                        rows += 1
                part.unlink()
                total += rows
                print(
                    f"[prep] {region}: {rows} rows over {len(starts)} bins (running total {total})"
                )
        tmp_path.replace(out)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"[prep] {out}: {total} singleton rows across {len(regions)} regions")


if __name__ == "__main__":
    main()
