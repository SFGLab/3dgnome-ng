"""Add an SV call set to the site's catalog.

    python web/scripts/add_sv_set.py <site data dir> <NAME> <calls.vcf.gz> [--label "..."]

The VCF is copied under `svs/`, indexed with tabix when tabix is installed and its samples are
read from the header. Records need SVTYPE of DEL, INS, INV or DUP; others are skipped at run time.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site_data")
    ap.add_argument("name")
    ap.add_argument("vcf")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    site = Path(a.site_data)
    (site / "svs").mkdir(parents=True, exist_ok=True)
    dest = site / "svs" / f"{a.name}.vcf.gz"
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite it")
    shutil.copyfile(a.vcf, dest)
    if shutil.which("tabix"):
        subprocess.run(["tabix", "-p", "vcf", str(dest)], check=False)
    samples: list[str] = []
    with gzip.open(dest, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                break
    cat_path = site / "catalog.json"
    cat = json.loads(cat_path.read_text()) if cat_path.is_file() else {"cell_types": {}, "sv_sets": {}}
    cat["sv_sets"][a.name] = {"label": a.label or a.name, "vcf": f"svs/{a.name}.vcf.gz", "samples": samples}
    cat_path.write_text(json.dumps(cat, indent=2) + "\n")
    print(f"added {a.name}, {len(samples)} samples")


if __name__ == "__main__":
    main()
