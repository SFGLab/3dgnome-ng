"""Spike. Download the trio ChIA-PET inputs from Drive into data/_trio/<SAMPLE>/.

Selects only the files the modelling pipeline needs, which is about 5 GB of the folder's 34 GB.
Selection reads the inventory JSON that gdrive_inventory.py writes, so a path that moves in
Drive shows up as a missing selection rather than a silent wrong file.

    python playground/trio/gdrive_inventory.py <folder-url> --json inventory.json
    python playground/trio/trio_fetch.py --inventory inventory.json --dry-run
    python playground/trio/trio_fetch.py --inventory inventory.json --samples HG00512

Downloads are resumable in the sense that a finished file is skipped when its md5 matches
Drive's, and an interrupted one leaves a .part file that is rewritten rather than appended to.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trio_samples  # noqa: E402
from gdrive_inventory import service  # noqa: E402
from googleapiclient.http import MediaIoBaseDownload  # noqa: E402

CHUNK = 32 * 1024 * 1024


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pick(rows: list[dict[str, Any]], want: str, exact: bool = True) -> list[dict[str, Any]]:
    if exact:
        return [r for r in rows if r["path"] == want]
    return [r for r in rows if r["path"].startswith(want)]


def select(
    rows: list[dict[str, Any]],
    sample: trio_samples.Sample,
    arm: str,
    factor: str,
    include_gz: bool,
) -> list[tuple[str, dict[str, Any]]]:
    """Choose this sample's files. Returns (local name, drive row) pairs.

    Raises when a required file is missing, because a sample silently ingested without its loops
    would reach the model as an empty structure rather than an error.
    """
    fl = trio_samples.FACTOR_LOOPS[factor]
    fd = trio_samples.FACTOR_DOWNSAMPLING[factor]
    ds_dir = f"/downsampling/{sample.downsampling_dir}/{sample.name}_{fd}/"
    merged_dir = f"/Loops/{sample.pop}/{fl}/{sample.name}/merged/"
    out: list[tuple[str, dict[str, Any]]] = []

    if arm == "downsampled":
        loops = pick(rows, ds_dir + "subsample_1.e500.clusters.cis.BE3")
        gz = pick(rows, ds_dir + "subsample_1.e500.clusters.cis.gz")
    else:
        loops = [r for r in pick(rows, merged_dir, exact=False) if r["name"].endswith(".BE3")]
        gz = [r for r in pick(rows, merged_dir, exact=False) if r["name"].endswith(".cis.gz")]
        # HG00732 and HG00514 ship an "alternativemerge" instead of, or alongside, the plain
        # merge. Prefer the plain one so the nine samples are treated the same way.
        plain = [r for r in loops if "alternativemerge" not in r["name"]]
        if plain and len(loops) > 1:
            print(f"[fetch:{sample.name}] {len(loops)} merged BE3 files, taking the plain merge")
            loops = plain
    if not loops:
        raise SystemExit(f"[fetch:{sample.name}] no loops BE3 found under {ds_dir or merged_dir}")
    out.append((f"{sample.name}_loops.BE3", loops[0]))
    if include_gz and gz:
        out.append((f"{sample.name}_loops.cis.gz", gz[0]))

    hic = pick(
        rows,
        f"/hic_files/pairs_merged_replicates_CTCF/ChIA-PET_hg38_{sample.name}_merged_allres.hic",
    )
    if not hic:
        raise SystemExit(f"[fetch:{sample.name}] no _allres.hic found")
    out.append((f"{sample.name}_allres.hic", hic[0]))

    peaks = [
        r
        for r in pick(rows, f"/Peaks/{fl}/{sample.name}_merged_replicates", exact=False)
        if r["name"].endswith(".broadPeak") and "alternativemerge" not in r["name"]
    ]
    if peaks:
        out.append((f"{sample.name}_peaks.broadPeak", peaks[0]))
    else:
        print(f"[fetch:{sample.name}] no merged broadPeak, orientation will use motifs only")

    stats = [
        r for r in pick(rows, merged_dir, exact=False) if r["name"].endswith("final_stats.tsv")
    ]
    if stats:
        out.append((f"{sample.name}_final_stats.tsv", stats[0]))

    # The high quality set keeps PET3+ loops whose anchors both overlap a CTCF site. It is
    # the modelling input, so it is always fetched. The phased splits sit beside it and are
    # small, and they are what a later haplotype arm would need.
    if True:
        for r in pick(rows, ds_dir, exact=False):
            if r["name"].startswith("wyniki_"):
                tag = r["name"].split(".BE3")[-1].lstrip(".") or "hq.BE3"
                out.append((f"{sample.name}_{tag}", r))
    return out


def fetch(svc: Any, row: dict[str, Any], dest: Path) -> str:
    if dest.is_file() and row.get("md5") and md5_of(dest) == row["md5"]:
        return "cached"
    part = dest.with_suffix(dest.suffix + ".part")
    request = svc.files().get_media(fileId=row["id"], supportsAllDrives=True)
    with part.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status and (row["size"] or 0) > 50e6:
                print(f"      {status.progress() * 100:5.1f}%", end="\r", flush=True)
    if row.get("md5") and md5_of(part) != row["md5"]:
        part.unlink()
        raise SystemExit(f"md5 mismatch on {row['path']}, refusing to keep a corrupt file")
    part.replace(dest)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, help="JSON written by gdrive_inventory.py")
    ap.add_argument("--out", default="data/_trio")
    ap.add_argument("--arm", choices=("downsampled", "merged"), default="downsampled")
    ap.add_argument("--factor", choices=("CTCF", "RNAPOL2"), default="CTCF")
    ap.add_argument("--samples", help="comma separated, default all nine")
    ap.add_argument("--include-gz", action="store_true", help="also the PET1+ loops, adds ~4 GB")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = json.loads(Path(args.inventory).read_text())
    samples = trio_samples.resolve(args.samples)
    plan = [
        (s, select(rows, s, args.arm, args.factor, args.include_gz))
        for s in samples
    ]
    total = sum(r["size"] or 0 for _, files in plan for _, r in files)
    print(f"{len(samples)} samples, arm={args.arm}, factor={args.factor}, {total / 1e9:.2f} GB\n")
    for s, files in plan:
        print(f"{s.name} ({s.role}, {s.pop})")
        for local, r in files:
            print(f"   {local:<44} {(r['size'] or 0) / 1e6:9.2f} MB  {r['path']}")
    if args.dry_run:
        return

    svc = service()
    for s, files in plan:
        root = Path(args.out) / s.name
        root.mkdir(parents=True, exist_ok=True)
        for local, r in files:
            print(f"[fetch:{s.name}] {local} ({(r['size'] or 0) / 1e6:.1f} MB)", flush=True)
            print(f"      {fetch(svc, r, root / local)}")


if __name__ == "__main__":
    main()
