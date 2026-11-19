"""Experiment data loader. Fetch 4DN and ENCODE files from a declarative manifest.

Fetches two data families reproducibly.

  * epigenomic tracks such as CTCF, RAD21, ATAC/DNase, histone ChIP, RNA-seq, from ENCODE
  * contacts and imaging such as ChIA-PET, Hi-C, MERFISH, from 4DN

A manifest is one JSON per cell line, see legacy/validation/manifests/. It lists a {source, accession}
per assay. The loader resolves each accession to a concrete file URL and md5 via the portal REST
API, downloads it with caching, checksum verification, and skip-if-present, and writes a lockfile
pinning the resolved accession, md5, and url so a rebuild is byte-identical.

Binning bigWig signal onto a bead grid, the conditioning epitensor, is a downstream transform
that belongs with the model that reads it. That is out of scope
here. This module's job is reproducible acquisition.

    python -m legacy.validation fetch --manifest legacy/validation/manifests/GM12878.json --out data/_epigenome
    python -m legacy.validation fetch --manifest ... --dry-run     # resolve and print plan, no download

fetch does not reconstruct, so it ignores the shared reconstruction args and needs only --manifest.

stdlib only, urllib/json/hashlib. 4DN files behind access control need DN_KEY/DN_SECRET env vars
over HTTP basic auth. The 4DN portal TLS certificate is currently expired server-side, so a CA
bundle cannot fix it. 4DN requests therefore skip TLS verification automatically, with a warning,
while ENCODE stays verified. --insecure forces skip-verify for all sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from legacy.validation.studies import Context, Study, register

ENCODE_BASE = "https://www.encodeproject.org"


@dataclass
class AssaySpec:
    """One requested file. Which assay, which portal, which accession."""

    name: str  # logical track name, e.g. "CTCF", "H3K27ac", "ATAC", "gencode", "enhancers"
    source: str  # one of "encode", "4dn", "url"
    accession: str = ""  # ENCFF.../ENCSR... or 4DNFI..., empty for a "url" source
    output_type: str | None = None  # preferred ENCODE output_type, disambiguates a dataset
    url: str | None = None  # direct download URL for source="url", e.g. GENCODE, EnhancerAtlas
    md5: str | None = None  # optional expected md5 for a "url" source


@dataclass
class Manifest:
    cell_line: str
    assembly: str
    assays: list[AssaySpec]

    @classmethod
    def from_json(cls, path: str) -> Manifest:
        with open(path) as f:
            raw = json.load(f)
        return cls(
            cell_line=raw["cell_line"],
            assembly=raw["assembly"],
            assays=[AssaySpec(**a) for a in raw["assays"]],
        )


@dataclass
class FileRef:
    """A resolved, concrete downloadable file."""

    assay: str
    source: str
    accession: str
    url: str
    md5: str | None
    file_format: str | None
    output_type: str | None


def _get_json(url: str, ctx: ssl.SSLContext, auth: tuple[str, str] | None = None) -> dict:
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "3dgnome-validation"}
    )
    if auth:
        import base64

        token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return json.loads(r.read().decode())


def resolve_encode(spec: AssaySpec, ctx: ssl.SSLContext) -> FileRef:
    """Resolve an ENCODE accession to a concrete file.

    A file accession, ENCFF..., resolves directly. An experiment or dataset accession, ENCSR...,
    is searched for a released file matching output_type. The default is the first released
    bigWig or signal file.
    """
    acc = spec.accession
    if acc.startswith("ENCFF"):
        meta = _get_json(f"{ENCODE_BASE}/files/{acc}/?format=json", ctx)
        return FileRef(
            assay=spec.name,
            source="encode",
            accession=acc,
            url=ENCODE_BASE + meta["href"],
            md5=meta.get("md5sum"),
            file_format=meta.get("file_format"),
            output_type=meta.get("output_type"),
        )
    # Dataset. Pick a released file, preferring the requested output_type.
    meta = _get_json(f"{ENCODE_BASE}/experiments/{acc}/?format=json", ctx)
    files = [f for f in meta.get("files", []) if f.get("status") == "released"]
    want = spec.output_type or "fold change over control"
    chosen = next((f for f in files if f.get("output_type") == want), None)
    if chosen is None:
        chosen = next((f for f in files if f.get("file_format") == "bigWig"), None)
    if chosen is None:
        raise ValueError(f"ENCODE {acc}: no released bigWig / '{want}' file found")
    return FileRef(
        assay=spec.name,
        source="encode",
        accession=chosen["accession"],
        url=ENCODE_BASE + chosen["href"],
        md5=chosen.get("md5sum"),
        file_format=chosen.get("file_format"),
        output_type=chosen.get("output_type"),
    )


DN_BUCKET = "https://4dn-open-data-public.s3.amazonaws.com"
DN_PREFIX = "fourfront-webprod/wfoutput/"
DN_KEYS = Path(__file__).resolve().parents[1] / "manifests" / "4dn_keys.tsv"


def list_dn_bucket(ctx: ssl.SSLContext, want: set[str]) -> dict[str, tuple[str, int]]:
    """The bucket key and size of each wanted accession, found by listing the 4DN prefix page by
    page until every one is seen or the listing ends."""
    import re
    import urllib.parse

    found: dict[str, tuple[str, int]] = {}
    token = None
    while len(found) < len(want):
        q = {"list-type": "2", "prefix": DN_PREFIX, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(
            f"{DN_BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60, context=ctx
        ) as r:
            page = r.read().decode()
        for key, size in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", page):
            acc = key.rsplit("/", 1)[-1].split(".")[0]
            if acc in want:
                found[acc] = (key, int(size))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", page)
        if not m:
            break
        token = m.group(1)
    return found


def dn_key(
    accession: str, ctx: ssl.SSLContext, table: Path = DN_KEYS, lister=list_dn_bucket
) -> tuple[str, int]:  # noqa: ANN001
    """The bucket key and size of a 4DN accession, from the table beside the manifests, or found
    by listing the bucket and added to the table.

    @raises ValueError naming an accession the bucket does not hold
    """
    rows = {}
    if table.exists():
        for line in table.read_text().splitlines()[1:]:
            acc, key, size = line.split("\t")
            rows[acc] = (key, int(size))
    if accession not in rows:
        hit = lister(ctx, {accession})
        if accession not in hit:
            raise ValueError(f"4DN {accession} is not in the bucket {DN_BUCKET}/{DN_PREFIX}")
        if not table.exists():
            table.write_text("accession\tkey\tsize\n")
        with table.open("a") as f:
            f.write(f"{accession}\t{hit[accession][0]}\t{hit[accession][1]}\n")
        rows[accession] = hit[accession]
    return rows[accession]


def resolve_4dn(
    spec: AssaySpec, ctx: ssl.SSLContext, table: Path = DN_KEYS, lister=list_dn_bucket
) -> FileRef:  # noqa: ANN001
    """Resolve a 4DN accession, 4DNFI..., to its file in the public bucket. The 4DN portal that
    gave the file's metadata is gone, see design/4dn-data-access.md, so there is no md5 and no
    assay to check; the manifest's note says what each file holds."""
    key, _ = dn_key(spec.accession, ctx, table, lister)
    return FileRef(
        assay=spec.name,
        source="4dn",
        accession=spec.accession,
        url=f"{DN_BUCKET}/{key}",
        md5=None,
        file_format=key.rsplit(".", 1)[-1],
        output_type=spec.output_type,
    )


def resolve_url(spec: AssaySpec) -> FileRef:
    """A direct download URL for sources without an accession API, such as GENCODE gene
    annotation at ftp.ebi.ac.uk or EnhancerAtlas enhancer BEDs. The manifest gives the literal url."""
    if not spec.url:
        raise ValueError(f"url source {spec.name!r} needs a 'url' field")
    fmt = spec.url.split("?")[0].rstrip("/").rsplit(".", 1)[-1]
    return FileRef(
        assay=spec.name,
        source="url",
        accession=spec.accession or spec.name,
        url=spec.url,
        md5=spec.md5,
        file_format=fmt,
        output_type=None,
    )


def resolve(spec: AssaySpec, ctx: ssl.SSLContext) -> FileRef:
    if spec.source == "encode":
        return resolve_encode(spec, ctx)
    if spec.source == "4dn":
        return resolve_4dn(spec, ctx)
    if spec.source == "url":
        return resolve_url(spec)
    raise ValueError(f"unknown source {spec.source!r} (expected 'encode', '4dn', or 'url')")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(ref: FileRef, dest: Path, ctx: ssl.SSLContext) -> bool:
    """Download ref to dest, skipping if present and md5 matches. Returns True if a download
    happened, False if the cached file was reused. Verifies md5 when known."""
    ext = Path(ref.url.split("?")[0]).suffix
    out = dest / f"{ref.assay}.{ref.accession}{ext}"
    if out.exists() and (ref.md5 is None or _md5(out) == ref.md5):
        print(f"  [cached] {ref.assay} ({ref.accession}) -> {out.name}")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"  [fetch ] {ref.assay} ({ref.accession}) <- {ref.url}")
    req = urllib.request.Request(ref.url, headers={"User-Agent": "3dgnome-validation"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if ref.md5 is not None:
        got = _md5(tmp)
        if got != ref.md5:
            tmp.unlink(missing_ok=True)
            raise ValueError(f"{ref.assay}: md5 mismatch (got {got}, want {ref.md5})")
    tmp.replace(out)
    return True


def write_lockfile(manifest: Manifest, refs: list[FileRef], dest: Path) -> None:
    """Pin resolved accessions, md5, and url so a rebuild is byte-identical."""
    lock = {
        "cell_line": manifest.cell_line,
        "assembly": manifest.assembly,
        "resolved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [asdict(r) for r in refs],
    }
    path = dest / "manifest.lock.json"
    path.write_text(json.dumps(lock, indent=2))
    print(f"  [lock  ] {path}")


class Fetch(Study):
    name = "fetch"
    help = "fetch 4DN/ENCODE experiment files from a manifest"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--manifest", required=True, help="path to a cell-line manifest JSON")
        p.add_argument(
            "--out", default="data/_epigenome", help="output root (default data/_epigenome)"
        )
        p.add_argument(
            "--dry-run", action="store_true", help="resolve + print plan, do not download"
        )
        p.add_argument(
            "--insecure",
            action="store_true",
            help="skip TLS verification (opt-in; for portals with cert issues in your env)",
        )

    def run(self, ctx: Context, args: argparse.Namespace) -> None:
        if not Path(args.manifest).exists():
            sys.exit(f"[error] manifest not found: {args.manifest}")
        manifest = Manifest.from_json(args.manifest)
        dest = Path(args.out) / manifest.cell_line

        secure = ssl.create_default_context()
        insecure = ssl.create_default_context()
        insecure.check_hostname = False
        insecure.verify_mode = ssl.CERT_NONE
        if args.insecure:
            print("[fetch] WARNING: TLS verification disabled for ALL sources (--insecure)")

        def ctx_for(source: str) -> ssl.SSLContext:
            # 4DN files come from their S3 bucket, whose certificate verifies like ENCODE's.
            if args.insecure:
                return insecure
            return secure

        print(f"[fetch] {manifest.cell_line} ({manifest.assembly}): {len(manifest.assays)} assays")
        refs: list[FileRef] = []
        failures = 0
        for spec in manifest.assays:
            try:
                ref = resolve(spec, ctx_for(spec.source))
            except (HTTPError, URLError, ValueError, KeyError) as e:
                print(
                    f"  [ERROR ] {spec.name} ({spec.source}:{spec.accession}): {type(e).__name__}: {e}"
                )
                failures += 1
                continue
            refs.append(ref)
            md5s = (ref.md5 or "?")[:8]
            print(
                f"  [resolve] {ref.assay:<10} {ref.accession}  fmt={ref.file_format}  "
                f"assay={ref.output_type}  md5={md5s}"
            )
            if not args.dry_run:
                try:
                    download(ref, dest, ctx_for(ref.source))
                except (HTTPError, URLError, ValueError) as e:
                    print(f"  [ERROR ] download {ref.assay}: {type(e).__name__}: {e}")
                    failures += 1

        if refs and not args.dry_run:
            write_lockfile(manifest, refs, dest)
        if failures:
            sys.exit(f"[fetch] {failures} failure(s)")
        print("[fetch] done")


register(Fetch())
