"""Experiment data loader. Fetch 4DN and ENCODE files from a declarative manifest.

Sources the data families docs/epigenome-to-structure.md §7 needs, reproducibly.

  * epigenomic tracks such as CTCF, RAD21, ATAC/DNase, histone ChIP, RNA-seq, from ENCODE
  * contacts and imaging such as ChIA-PET, Hi-C, MERFISH, from 4DN

A manifest is one JSON per cell line, see validation/manifests/. It lists a {source, accession}
per assay. The loader resolves each accession to a concrete file URL and md5 via the portal REST
API, downloads it with caching, checksum verification, and skip-if-present, and writes a lockfile
pinning the resolved accession, md5, and url so a rebuild is byte-identical.

Binning bigWig signal onto a bead grid, the conditioning epitensor, is a downstream transform
that belongs with the ML model at epigenome-to-structure Phase 0 step 3. That is out of scope
here. This module's job is reproducible acquisition.

    python -m validation fetch --manifest validation/manifests/GM12878.json --out data/_epigenome
    python -m validation fetch --manifest ... --dry-run     # resolve and print plan, no download

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
import os
import ssl
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from validation.studies import Context, Study, register

ENCODE_BASE = "https://www.encodeproject.org"
DN_BASE = "https://data.4dnucleome.org"


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


def resolve_4dn(spec: AssaySpec, ctx: ssl.SSLContext) -> FileRef:
    """Resolve a 4DN accession, 4DNFI..., to a concrete file. Honors DN_KEY/DN_SECRET."""
    auth = None
    key, secret = os.environ.get("DN_KEY"), os.environ.get("DN_SECRET")
    if key and secret:
        auth = (key, secret)
    meta = _get_json(f"{DN_BASE}/{spec.accession}/?format=json", ctx, auth=auth)
    # Prefer open_data_url, the public 4dn-open-data S3 mirror with a valid cert and no auth. The
    # gated @@download href 403s for open files and the portal cert is expired. open_data_url avoids both.
    url = meta.get("open_data_url")
    if not url:
        href = meta.get("href")
        if not href:
            raise ValueError(f"4DN {spec.accession}: no open_data_url or href in metadata")
        url = href if href.startswith("http") else DN_BASE + href
    # 3dgnome models CTCF-mediated architecture, so the validation contact target must be
    # CTCF-relevant, plain Hi-C or CTCF ChIA-PET, not RNA Pol II ChIA-PET or HiChIP, whose
    # transcription-driven loops are the wrong ground truth. Flag a bad pick loudly.
    assay = (
        (meta.get("track_and_facet_info") or {}).get("assay_info") or meta.get("file_type") or ""
    )
    if any(t in assay for t in ("Pol II", "RNA Pol", "POLR2")):
        print(
            f"  [WARN ] {spec.name} ({spec.accession}) assay='{assay}' is RNA Pol II. "
            "wrong contact type for a CTCF-driven model. pick plain Hi-C or CTCF ChIA-PET."
        )
    return FileRef(
        assay=spec.name,
        source="4dn",
        accession=spec.accession,
        url=url,
        md5=meta.get("md5sum") or meta.get("content_md5sum"),
        file_format=(meta.get("file_format") or {}).get("display_title")
        if isinstance(meta.get("file_format"), dict)
        else meta.get("file_format"),
        output_type=assay or meta.get("file_type"),
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
        p.add_argument("--out", default="data/_epigenome", help="output root (default data/_epigenome)")
        p.add_argument("--dry-run", action="store_true", help="resolve + print plan, do not download")
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
        _warned_4dn = [False]

        def ctx_for(source: str) -> ssl.SSLContext:
            # 4DN's portal cert is expired server-side and verify would always fail, so we skip
            # verify for 4DN. ENCODE stays verified unless --insecure.
            if args.insecure:
                return insecure
            if source == "4dn":
                if not _warned_4dn[0]:
                    print("[fetch] NOTE: 4DN TLS cert is expired server-side; skipping verify for 4DN")
                    _warned_4dn[0] = True
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
