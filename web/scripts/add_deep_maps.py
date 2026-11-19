"""Adds the deep Hi-C map of each cell type to the catalog.json of a site as hic_url and hic_name.

    python web/scripts/add_deep_maps.py DATA_DIR/catalog.json

The accessions come from legacy/validation/manifests/<CELL>_hic.json and 4dn_keys.tsv, see
design/4dn-data-access.md. The file is rewritten in place.
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
    for line in (ROOT / "legacy" / "validation" / "manifests" / "4dn_keys.tsv").read_text().splitlines()[1:]:
        acc, key, _ = line.split("\t")
        keys[acc] = key
    out = {}
    for m in (ROOT / "legacy" / "validation" / "manifests").glob("*_hic.json"):
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
