"""Unit checks for 4DN files fetched from their public bucket.

    python legacy/harness/test_fetch_4dn.py

The 4DN portal is gone and its files remain in the bucket 4dn-open-data-public, under a uuid the
accession does not give. A table beside the manifests maps each accession to its key; a missing
accession is found by listing the bucket and added to the table.
"""

from __future__ import annotations

import ssl
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legacy.validation.studies.fetch import DN_BUCKET, AssaySpec, dn_key, resolve_4dn  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "\033[32mPASS\033[0m" if cond else "\033[31mFAIL\033[0m"
    print(f"  {mark}  {name}   {detail if not cond else ''}")


def main() -> int:
    ctx = ssl.create_default_context()
    with tempfile.TemporaryDirectory() as d:
        table = Path(d) / "4dn_keys.tsv"
        table.write_text(
            "accession\tkey\tsize\n4DNFIAAAAAAA\tfourfront-webprod/wfoutput/u1/4DNFIAAAAAAA.hic\t100\n"
        )
        calls: list[set[str]] = []

        def lister(_ctx: ssl.SSLContext, want: set[str]) -> dict[str, tuple[str, int]]:
            calls.append(set(want))
            listed = {"4DNFIBBBBBBB": ("fourfront-webprod/wfoutput/u2/4DNFIBBBBBBB.mcool", 200)}
            return {a: k for a, k in listed.items() if a in want}

        key, size = dn_key("4DNFIAAAAAAA", ctx, table, lister)
        check(
            "an accession in the table gives its key without a listing",
            key.endswith("u1/4DNFIAAAAAAA.hic") and size == 100 and not calls,
        )
        key, size = dn_key("4DNFIBBBBBBB", ctx, table, lister)
        check(
            "a missing accession is found by listing the bucket",
            key.endswith("u2/4DNFIBBBBBBB.mcool") and calls == [{"4DNFIBBBBBBB"}],
        )
        check(
            "and added to the table",
            "4DNFIBBBBBBB\tfourfront-webprod/wfoutput/u2/4DNFIBBBBBBB.mcool\t200"
            in table.read_text(),
        )
        dn_key("4DNFIBBBBBBB", ctx, table, lister)
        check("a second request reads the table", len(calls) == 1)
        try:
            dn_key("4DNFICCCCCCC", ctx, table, lister)
            check("an accession absent from the bucket stops naming it", False)
        except ValueError as e:
            check(
                "an accession absent from the bucket stops naming it",
                "4DNFICCCCCCC" in str(e),
                str(e),
            )

        ref = resolve_4dn(
            AssaySpec(name="hic", source="4dn", accession="4DNFIBBBBBBB"), ctx, table, lister
        )
        check(
            "a 4DN spec resolves to its bucket url and format",
            ref.url == f"{DN_BUCKET}/fourfront-webprod/wfoutput/u2/4DNFIBBBBBBB.mcool"
            and ref.file_format == "mcool",
            str(ref),
        )

    real = ROOT / "legacy" / "validation" / "manifests" / "4dn_keys.tsv"
    rows = [ln.split("\t")[0] for ln in real.read_text().splitlines()[1:]] if real.exists() else []
    import json

    wanted = {
        a["accession"]
        for m in (ROOT / "legacy" / "validation" / "manifests").glob("*.json")
        for a in json.loads(m.read_text()).get("assays", [])
        if a.get("source") == "4dn"
    }
    check(
        "every 4DN accession of the manifests is in the table",
        wanted <= set(rows),
        str(sorted(wanted - set(rows))),
    )

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
