"""Adds each cell type's deep Hi-C map to a site's catalog.json as hic_url and hic_name, so the
contact map tab compares a job with the deep map rather than the thinned contacts the job read.

    python web/scripts/add_deep_maps.py DATA_DIR/catalog.json

The accession of each cell line comes from validation/manifests/<CELL>_hic.json and its file in
the 4DN bucket from validation/manifests/4dn_keys.tsv, design/4dn-data-access.md. A cell type
without a 4DN Hi-C manifest is left as it is. The file is rewritten in place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "https://4dn-open-data-public.s3.amazonaws.com"


def deep_maps() -> dict[str, tuple[str, str]]:
    """Cell line to the bucket url and the name of its deep 4DN Hi-C map."""
    keys = {}
    for line in (ROOT / "validation" / "manifests" / "4dn_keys.tsv").read_text().splitlines()[1:]:
        acc, key, _ = line.split("\t")
        keys[acc] = key
    out = {}
    for m in (ROOT / "validation" / "manifests").glob("*_hic.json"):
        spec = json.loads(m.read_text())
        for a in spec.get("assays", []):
            if a.get("source") == "4dn" and a["accession"] in keys:
                out[spec["cell_line"].upper()] = (f"{BUCKET}/{keys[a['accession']]}", f"4DN {a['accession']}")
    return out


def main(path: str) -> int:
    p = Path(path)
    cat = json.loads(p.read_text())
    maps = deep_maps()
    for name, ct in cat.get("cell_types", {}).items():
        hit = maps.get(name.upper())
        if hit:
            ct["hic_url"], ct["hic_name"] = hit
            print(f"{name}: {hit[1]}")
        else:
            print(f"{name}: no 4DN Hi-C manifest, left as it is")
    p.write_text(json.dumps(cat, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
