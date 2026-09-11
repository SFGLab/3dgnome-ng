"""Blend ChIA-PET and Hi-C singletons into one file, balanced per cell line.

The two assays are wildly unequal in contact mass, and unequal by different factors per cell
line: Hi-C carries 9.1x the ChIA-PET mass in GM12878, 4.2x in H1ESC and 54.6x in HFFC6. Simply
concatenating would let Hi-C drive the distance targets, and would blend the three cell lines in
three different proportions, which is its own cross-line artefact.

So ChIA-PET is scaled up to a chosen mass ratio against Hi-C, rather than Hi-C being scaled down.
Hi-C scores are mostly 1, so scaling them down would round to 0, clamp back to 1, and silently
discard the balancing. Scaling the other way keeps every count an integer and loses nothing.

`--hic-weight 1.0` gives the two assays equal mass. Lower values favour ChIA-PET, higher favour
Hi-C. This assumes only the ratio matters, since the singleton heatmap is normalised downstream
before it becomes a distance target.

    python slurm/ensemble/combine_singletons.py --cell GM12878 \
        --chiapet data/GM12878/GM12878_singletons_lessthan3.bedpe \
        --hic     data/GM12878/GM12878_hic_25kb_singletons.bedpe \
        --out     data/GM12878/GM12878_combined_singletons.bedpe
"""

import argparse
import sys
import tempfile
from pathlib import Path


def mass(path: Path) -> tuple[int, int]:
    """Row count and summed score of a 7-column singleton BEDPE."""
    rows = total = 0
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            rows += 1
            total += int(parts[6])
    return rows, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--chiapet", required=True, type=Path)
    ap.add_argument("--hic", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--hic-weight",
        type=float,
        default=1.0,
        help="target Hi-C mass as a multiple of ChIA-PET mass after scaling; 1.0 is equal",
    )
    args = ap.parse_args()

    if args.out.exists() and args.out.stat().st_size > 0:
        print(f"[combine] {args.out} exists and is non-empty, leaving it alone", flush=True)
        return
    for p in (args.chiapet, args.hic):
        if not p.is_file() or p.stat().st_size == 0:
            sys.exit(f"[combine] missing or empty input: {p}")

    cp_rows, cp_mass = mass(args.chiapet)
    hic_rows, hic_mass = mass(args.hic)
    if cp_mass == 0:
        sys.exit(f"[combine] ChIA-PET mass is zero: {args.chiapet}")

    # Scale ChIA-PET so that hic_mass == hic_weight * scaled_chiapet_mass.
    scale = max(1, round(hic_mass / (cp_mass * args.hic_weight)))
    print(
        f"[combine] {args.cell}: ChIA-PET {cp_rows} rows mass {cp_mass}, "
        f"Hi-C {hic_rows} rows mass {hic_mass} "
        f"(ratio {hic_mass / cp_mass:.1f}x) -> scaling ChIA-PET by {scale}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=args.out.parent, suffix=".partial", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        written = 0
        with tmp_path.open("w") as sink:
            with args.chiapet.open() as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    parts[6] = str(int(parts[6]) * scale)
                    sink.write("\t".join(parts[:7]) + "\n")
                    written += 1
            with args.hic.open() as fh:
                for line in fh:
                    if line.strip():
                        sink.write(line if line.endswith("\n") else line + "\n")
                        written += 1
        if written == 0:
            sys.exit("[combine] wrote nothing; refusing to leave an empty file")
        tmp_path.replace(args.out)
    finally:
        tmp_path.unlink(missing_ok=True)

    final_cp_mass = cp_mass * scale
    print(
        f"[combine] {args.out}: {written} rows, "
        f"ChIA-PET mass {final_cp_mass} vs Hi-C mass {hic_mass} "
        f"(achieved {hic_mass / final_cp_mass:.2f}x, target {args.hic_weight:.2f}x)",
        flush=True,
    )


if __name__ == "__main__":
    main()
