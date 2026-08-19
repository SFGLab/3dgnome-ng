"""Write a region's Hi-C bin pairs as a singleton BEDPE the model can ingest.

Every pair with a count is written. The train/test split that
``validation.studies.self_corr`` applies is a validation device, and holding half the contacts
back would only weaken a production ensemble, so ``holdout=False`` here.

    python slurm/ensemble/prep_singletons.py \
        --mcool data/_hic/GM12878/hic.4DNFIQ32RWCQ.mcool \
        --region chr1 --binsize 25000 \
        --out data/GM12878/GM12878_chr1_hic_25kb_singletons.bedpe
"""

import argparse
import sys
from pathlib import Path

# pyproject installs gnome3d* only, so validation/ is importable from the repo root and nowhere
# else. Running this file by path puts its own directory on sys.path rather than the root, so
# the root goes on explicitly and the script works from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.studies.self_corr import hic_to_singleton_bedpe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcool", required=True)
    ap.add_argument("--region", required=True, help="e.g. chr1 or chr1:0-40000000")
    ap.add_argument("--binsize", type=int, default=25000)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Drop pairs below this raw count. Raise it on a deep map to cut file size.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[prep] {out} exists, leaving it alone")
        return

    _bal, starts, _test, _train = hic_to_singleton_bedpe(
        args.mcool, args.region, args.binsize, out, holdout=False, min_count=args.min_count
    )
    rows = sum(1 for _ in out.open())
    print(f"[prep] {out}: {rows} singleton rows over {len(starts)} bins")


if __name__ == "__main__":
    main()
