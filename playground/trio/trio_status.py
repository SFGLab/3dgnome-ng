"""Report what is ingested and what is modelled, and name the array tasks still missing.

Everything here reads the filesystem, so it is accurate after a cancelled job, a dead node or a
requeue. The array indices it prints use the same mapping the job script uses, because both call
trio_samples.shard.

    python playground/trio/trio_status.py
    python playground/trio/trio_status.py --resubmit        # just the --array spec for the gaps
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402


def count_lines(p: Path) -> int:
    if not p.is_file():
        return 0
    with p.open("rb") as fh:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))


def balanced(mcool: Path, res: int) -> bool:
    if not mcool.is_file():
        return False
    try:
        import cooler

        return "weight" in cooler.Cooler(f"{mcool}::/resolutions/{res}").bins().columns
    except Exception:
        return False


def ingest_row(s: trio_samples.Sample, data: Path, raw: Path, check_balance: bool) -> list[str]:
    d = data / s.name
    mcool = data / "_hic" / s.name / f"{s.name}.mcool"
    # Count what was fetched. Files this pipeline derives, the depth matched loop set and any
    # interrupted download, are not part of the expected nine.
    n_raw = (
        sum(
            1
            for f in (raw / s.name).glob("*")
            if not f.name.endswith((".matched.BE3", ".part"))
        )
        if (raw / s.name).is_dir()
        else 0
    )
    return [
        s.name,
        f"{n_raw}/9",
        str(count_lines(d / f"{s.name}_clusters_3+.bedpe") or "-"),
        str(count_lines(d / f"{s.name}_anchors_3+_oriented.bed") or "-"),
        "ok" if mcool.is_file() else "-",
        ("ok" if balanced(mcool, 25000) else "-") if check_balance else "?",
        str(count_lines(d / f"{s.name}_tads.bed") or "-"),
        str(count_lines(d / f"{s.name}_blocks.bed") or "-"),
        str(count_lines(d / f"{s.name}_segments.bed") or "-"),
        str(count_lines(d / f"{s.name}_hic_25kb_singletons.bedpe") or "-"),
        "ok" if (Path("slurm/ensemble") / f"{s.name.lower()}_trio.ini").is_file() else "-",
    ]


def table(rows: list[list[str]], head: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths, strict=False)))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=False)))


def done_members(out: Path, sample: str, chrom: str) -> set[int]:
    """Member ids with a finished .cif. A .part or a zero byte file does not count."""
    d = out / sample / chrom
    if not d.is_dir():
        return set()
    found: set[int] = set()
    for f in d.glob(f"{chrom}_s*.cif"):
        if f.stat().st_size == 0:
            continue
        stem = f.stem.rsplit("_s", 1)[-1]
        if stem.isdigit():
            found.add(int(stem))
    return found


def compress(idx: list[int]) -> str:
    """Turn a sorted index list into the range spec sbatch --array accepts."""
    if not idx:
        return ""
    parts: list[str] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = i
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--raw", default="data/_trio")
    ap.add_argument("--out", default="out/trio")
    ap.add_argument("--n-models", type=int, default=100)
    ap.add_argument("--per-task", type=int, default=10)
    ap.add_argument("--chroms", default=",".join(trio_samples.CHROMS))
    ap.add_argument("--samples")
    ap.add_argument("--resubmit", action="store_true", help="print only the --array spec")
    ap.add_argument("--skip-balance-check", action="store_true", help="faster, needs no cooler")
    args = ap.parse_args()

    samples = trio_samples.resolve(args.samples)
    chroms = [c for c in args.chroms.split(",") if c]
    data, raw, out = Path(args.data_root), Path(args.raw), Path(args.out)

    missing: list[int] = []
    total_done = 0
    grid: dict[tuple[str, str], int] = {}
    for task in range(trio_samples.array_length(chroms, samples, args.n_models, args.per_task)):
        got = trio_samples.shard(task, chroms, samples, args.n_models, args.per_task)
        if got is None:
            continue
        chrom, sample, first, last = got
        have = done_members(out, sample.name, chrom)
        want = set(range(first, last + 1))
        grid[(sample.name, chrom)] = len(have & set(range(args.n_models)))
        total_done += len(want & have)
        if want - have:
            missing.append(task)

    if args.resubmit:
        print(compress(missing))
        return

    print("INGEST")
    table(
        [ingest_row(s, data, raw, not args.skip_balance_check) for s in samples],
        ["sample", "raw", "clusters", "anchors", "mcool", "bal25k", "tads", "blocks", "segs",
         "singletons", "config"],
    )

    want_total = len(chroms) * len(samples) * args.n_models
    print(f"\nMODELS  {out}  ({args.n_models} per sample per chromosome)")
    head = ["sample", *chroms, "total"]
    rows = []
    for s in samples:
        counts = [grid.get((s.name, c), 0) for c in chroms]
        rows.append([s.name, *[str(n) for n in counts], f"{sum(counts)}/{len(chroms) * args.n_models}"])
    table(rows, head)

    pct = 100 * total_done / max(want_total, 1)
    print(f"\n{total_done}/{want_total} members done ({pct:.1f}%), {len(missing)} array tasks outstanding")
    if missing:
        print(f"resubmit with:\n  sbatch --array={compress(missing)}%6 slurm/ensemble/trio_ensemble.sh")


if __name__ == "__main__":
    main()
