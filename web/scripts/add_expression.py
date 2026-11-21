"""Puts expression tables into a site data directory and lists them in its catalog.

    python web/scripts/add_expression.py <site data dir> <tables dir>

The tables directory is the output of `make_expression_tables.py`. It has one table for each source
and `expression.json` with their catalog entries. A table goes to `expression/<NAME>.tsv`. The
catalog is first copied to `catalog.json.before-expression`. A table or an entry that exists
is refused. A cell type of an entry that the catalog does not hold is refused too.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site_data")
    ap.add_argument("tables")
    a = ap.parse_args()
    site, tables = Path(a.site_data), Path(a.tables)
    cat_path = site / "catalog.json"
    cat = json.loads(cat_path.read_text())
    entries = json.loads((tables / "expression.json").read_text())
    have = cat.setdefault("expression", {})
    for name, entry in entries.items():
        if name in have:
            raise SystemExit(f"{name} is in the catalog already; refusing to overwrite it")
        if (site / entry["file"]).exists():
            raise SystemExit(f"{site / entry['file']} exists; refusing to overwrite it")
        unknown = [c for c in entry.get("cell_types", {}) if c not in cat["cell_types"]]
        if unknown:
            raise SystemExit(f"{name} has the cell type {', '.join(unknown)}, which {cat_path} does not hold")
    (site / "expression").mkdir(exist_ok=True)
    for name, entry in entries.items():
        shutil.copyfile(tables / f"{name}.tsv", site / entry["file"])
        have[name] = entry
        print(f"wrote {site / entry['file']}")
    backup = site / "catalog.json.before-expression"
    if not backup.exists():
        shutil.copyfile(cat_path, backup)
    cat_path.write_text(json.dumps(cat, indent=2) + "\n")
    print(f"listed {', '.join(entries)} in {cat_path}")


if __name__ == "__main__":
    main()
