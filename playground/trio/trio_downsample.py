"""Match the loop count of a family's three samples, so a trio comparison is not a density
comparison.

The providers' `downsampling` folder was meant to give every sample a similar PET3+ loop count.
It succeeded for YRI, spread 1.02, and PUR, 1.12, but not for CHS, where HG00514 carries 1.61
times its family. Matching on those counts is not enough on its own, because the CTCF site
filter that produces the modelled loop set keeps between 14% and 26% depending on the sample, so
equal input to that filter still gives unequal output. The target here is therefore the filtered
set itself, drawn down to each family's own minimum. That lands anchor counts within 1% inside
every family.

The draw is uniform and seeded, so it preserves the PET strength distribution's shape rather
than favouring strong loops the way a raised PET threshold would. Loop strength is one of the
axes a trio comparison reads, so it must not be disturbed.

Counting and sampling always read the fetched `<S>_hq.BE3`, never a previous result, so
rerunning cannot compound. Output goes beside it as `<S>_hq.matched.BE3`, which
trio_prepare.py prefers when present. Delete those files to undo.

    python playground/trio/trio_downsample.py --dry-run
    python playground/trio/trio_downsample.py

Caveat. Subsampling loops is not the same as subsampling PETs and calling loops again. The reads
are not here, so this is an approximation of the step the providers intended.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402

SEED = 20260824


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/_trio")
    ap.add_argument("--scope", choices=("family", "global"), default="family",
                    help="match within each family, or across all nine")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw)
    src = {s.name: raw / s.name / f"{s.name}_hq.BE3" for s in trio_samples.SAMPLES}
    n_hq = {k: sum(1 for _ in v.open()) for k, v in src.items()}

    groups = (
        [("all", list(trio_samples.SAMPLES))]
        if args.scope == "global"
        else [(p, trio_samples.family(p)) for p in ("CHS", "PUR", "YRI")]
    )
    for label, fam in groups:
        target = min(n_hq[s.name] for s in fam)
        print(f"[downsample] {label} target = {target} hq loops")
        for s in fam:
            n = n_hq[s.name]
            dest = raw / s.name / f"{s.name}_hq.matched.BE3"
            if n == target:
                if dest.is_file():
                    dest.unlink()
                print(f"    {s.name}: {n}, already at target")
                continue
            print(
                f"    {s.name}: {n} -> {target} (keep {100 * target / n:.1f}%)"
                f"{'  dry run' if args.dry_run else ''}"
            )
            if args.dry_run:
                continue
            lines = src[s.name].open().readlines()
            idx = sorted(random.Random(args.seed).sample(range(len(lines)), target))
            tmp = dest.with_suffix(".part")
            with tmp.open("w") as fh:
                fh.writelines(lines[i] for i in idx)
            tmp.replace(dest)


if __name__ == "__main__":
    main()
