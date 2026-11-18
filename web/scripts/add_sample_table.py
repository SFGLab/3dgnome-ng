"""Give an SV call set of the site's catalog a table of its samples.

    python web/scripts/add_sample_table.py <site data dir> <NAME> <pedigree file>

The pedigree file is the one of the 1000 Genomes 30x panel,
`20130606_g1k_3202_samples_ped_population.txt`, with the columns FamilyID, SampleID, FatherID,
MotherID, Sex, Population and Superpopulation. The table is written to
`svs/<NAME>.samples.tsv` with the columns sample, sex, population, superpopulation, father and
mother, one row per sample of the call set. A parent outside the call set is left empty. The
catalog is copied to `catalog.json.before-sample-table-<NAME>` before its entry gets
`samples_table`. An existing table is refused.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SEX = {"1": "male", "2": "female"}


def read_pedigree(path: Path) -> dict[str, dict[str, str]]:
    """The pedigree's rows by sample."""
    rows: dict[str, dict[str, str]] = {}
    with path.open() as fh:
        header = fh.readline().split()
        col = {name.lower(): i for i, name in enumerate(header)}
        for line in fh:
            f = line.split()
            if len(f) < len(header):
                continue
            rows[f[col["sampleid"]]] = {
                "sex": SEX.get(f[col["sex"]], ""),
                "population": f[col["population"]],
                "superpopulation": f[col["superpopulation"]],
                "father": f[col["fatherid"]],
                "mother": f[col["motherid"]],
            }
    return rows


def table_rows(samples: list[str], pedigree: dict[str, dict[str, str]]) -> tuple[list[list[str]], dict[str, int]]:
    """One row per sample of the call set, and the counts the script prints."""
    known = set(samples)
    counts = {"samples": len(samples), "not in the pedigree": 0, "with a parent in the set": 0, "with both parents in the set": 0, "parents outside the set": 0}
    rows: list[list[str]] = []
    for s in samples:
        p = pedigree.get(s)
        if p is None:
            counts["not in the pedigree"] += 1
            rows.append([s, "", "", "", "", ""])
            continue
        parents = []
        for who in ("father", "mother"):
            name = p[who]
            if name in ("0", ""):
                parents.append("")
            elif name in known:
                parents.append(name)
            else:
                counts["parents outside the set"] += 1
                parents.append("")
        counts["with a parent in the set"] += any(parents)
        counts["with both parents in the set"] += all(parents)
        rows.append([s, p["sex"], p["population"], p["superpopulation"], *parents])
    return rows, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site_data")
    ap.add_argument("name")
    ap.add_argument("pedigree")
    a = ap.parse_args()
    site = Path(a.site_data)
    cat_path = site / "catalog.json"
    cat = json.loads(cat_path.read_text())
    if a.name not in cat["sv_sets"]:
        raise SystemExit(f"{a.name} is not a call set of {cat_path}")
    dest = site / "svs" / f"{a.name}.samples.tsv"
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite it")
    rows, counts = table_rows(list(cat["sv_sets"][a.name]["samples"]), read_pedigree(Path(a.pedigree)))
    dest.write_text("sample\tsex\tpopulation\tsuperpopulation\tfather\tmother\n" + "".join("\t".join(r) + "\n" for r in rows))
    backup = site / f"catalog.json.before-sample-table-{a.name}"
    if not backup.exists():
        shutil.copyfile(cat_path, backup)
    cat["sv_sets"][a.name]["samples_table"] = f"svs/{a.name}.samples.tsv"
    cat_path.write_text(json.dumps(cat, indent=2) + "\n")
    print(f"wrote {dest}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
