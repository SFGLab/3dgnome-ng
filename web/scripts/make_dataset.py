r"""Writes the directory of one cell type for the site from the prepared inputs of this
repository and adds it to the catalog.

    python web/scripts/make_dataset.py <repo data dir> <site data dir> <CELL> \
        --gtf gencode.v40.basic.annotation.gtf --enhancers <CELL>_enhancers_hg38.bed \
        [--label "GM12878, lymphoblastoid"] [--lymphoblastoid] [--note "..."]

`--genes` takes the genes.tsv of a cell type that is built, in place of `--gtf`. Files are
copied. An existing cell type directory is refused.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def genes_from_gtf(gtf: Path, out: Path) -> int:
    """Returns gene_id, name, chrom, tss and strand of every gene line."""
    opener = gzip.open if str(gtf).endswith(".gz") else open
    n = 0
    with opener(gtf, "rt") as fh, out.open("w") as w:
        w.write("gene_id\tname\tchrom\ttss\tstrand\n")
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            attrs = dict(a.strip().split(" ", 1) for a in f[8].split(";") if a.strip())
            gid = attrs.get("gene_id", "").strip('"')
            name = attrs.get("gene_name", gid).strip('"')
            tss = int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1
            w.write(f"{gid}\t{name}\t{f[0]}\t{tss}\t{f[6]}\n")
            n += 1
    return n


# The site fills the stretches no loop covers with beads, which the production config leaves empty.
# Its maps are read by eye and by the rank agreement, see design/gap-fill.md.
HOLES = "\n[holes]\nuse_hole_beads = yes\nflank_beads = 5\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo_data")
    ap.add_argument("site_data")
    ap.add_argument("cell")
    ap.add_argument("--gtf", default="")
    ap.add_argument("--genes", default="", help="a genes.tsv already built from the annotation, in place of --gtf")
    ap.add_argument("--enhancers", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--lymphoblastoid", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--singletons", default="", help="the Hi-C singletons, when not under the repo data dir")
    ap.add_argument("--contact-map", default="", help="a cooler URI at 1 kb on the server, for the stretches without loops")
    a = ap.parse_args()
    if bool(a.gtf) == bool(a.genes):
        raise SystemExit("give one of --gtf and --genes")
    c = a.cell
    src = Path(a.repo_data) / c
    site = Path(a.site_data)
    dest = site / "cell_types" / c
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite it")
    dest.mkdir(parents=True)
    copies = {
        "anchors_ctcf.bed": src / f"{c}_anchors_3+_oriented.bed",
        "anchors_ctcf_rnapii.bed": src / f"{c}_xc_anchors_ctcf_rnapii.bed",
        "ctcf.bedpe": src / f"{c}_xc_ctcf_clusters_3+.bedpe",
        "rnapii.bedpe": src / f"{c}_xc_rnapii_clusters_3+.bedpe",
        "singletons.bedpe": Path(a.singletons) if a.singletons else src / f"{c}_hic_25kb_singletons.bedpe",
        "segments.bed": src / f"ccds_all_hg38_merged100k_{c}.breakpoints.bed",
        "centromeres.bed": src / "hg38_centromeres.bed",
        "config.ini": ROOT / "slurm" / "ensemble" / f"{c.lower()}_xc_C.ini",
        "enhancers.bed": Path(a.enhancers),
    }
    for name, p in copies.items():
        if not p.is_file():
            raise SystemExit(f"missing {p}")
        shutil.copyfile(p, dest / name)
    with (dest / "config.ini").open("a") as fh:
        fh.write(HOLES)
    if a.genes:
        shutil.copyfile(a.genes, dest / "genes.tsv")
        with (dest / "genes.tsv").open() as fh:
            n = sum(1 for _ in fh) - 1
    else:
        n = genes_from_gtf(Path(a.gtf), dest / "genes.tsv")
    cat_path = site / "catalog.json"
    cat = json.loads(cat_path.read_text()) if cat_path.is_file() else {"cell_types": {}, "sv_sets": {}}
    cat["cell_types"][c] = {
        "label": a.label or c,
        "lymphoblastoid": a.lymphoblastoid,
        "dir": f"cell_types/{c}",
        "anchors": {"CTCF": "anchors_ctcf.bed", "CTCF+RNAPII": "anchors_ctcf_rnapii.bed"},
        "clusters": {"CTCF": "ctcf.bedpe", "RNAPII": "rnapii.bedpe"},
        "singletons": "singletons.bedpe",
        "segments": "segments.bed",
        "centromeres": "centromeres.bed",
        "config": "config.ini",
        "genes": "genes.tsv",
        "enhancers": "enhancers.bed",
        "note": a.note,
    }
    if a.contact_map:
        cat["cell_types"][c]["contact_map"] = a.contact_map
    cat_path.write_text(json.dumps(cat, indent=2) + "\n")
    print(f"wrote {dest}, {n} genes and the catalog entry")


if __name__ == "__main__":
    main()
