"""Contact maps by link. The server reads one window of a remote .hic or .mcool file over HTTP
range requests and keeps only its bin pairs at 25 kb, thinned to the density of the hosted maps.
The whole file never comes to the server.

A link must be http or https, end in .hic, .mcool or .cool, and resolve to a public address.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

RES = 25_000
READ_SECONDS = 120
HEAD_SECONDS = 20
KINDS = (".hic", ".mcool", ".cool")


class RemoteError(ValueError):
    """A link the server refuses, with the way to fix it."""

    def __init__(self, message: str, fix: str) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass
class Window:
    """The bin pairs of a window, upper triangle, bin starts in bp."""

    bin1: np.ndarray
    bin2: np.ndarray
    counts: np.ndarray
    resolutions: list[int]


def _host(url: str) -> str:
    return urlparse(url).hostname or url


def _public(url: str) -> None:
    """Refuses a link whose host resolves to a private, loopback or link local address."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RemoteError("The link must start with http or https.", "Give the web address of the file, for example a 4DN, ENCODE or GEO download link.")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    except OSError as e:
        raise RemoteError(f"{u.hostname} does not resolve.", "Check the address of the link.") from e
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise RemoteError(f"{u.hostname} is a private address.", "Give a link the public internet reaches.")


def check_link(url: str, allow_private: bool) -> str:
    """Refuses a link that is not http or https, not a contact map, or not public. Returns its kind.

    @raises RemoteError naming the problem
    """
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RemoteError("The link must start with http or https.", "Give the web address of the file, for example a 4DN, ENCODE or GEO download link.")
    kind = next((k for k in KINDS if u.path.lower().endswith(k)), None)
    if kind is None:
        raise RemoteError("The link must point to a .hic or .mcool file.", "A BEDPE of the region goes in through BEDPE of the region.")
    if not allow_private:
        _public(url)
    return kind


def probe(url: str, allow_private: bool, first_hop_checked: bool = False, timeout: float = HEAD_SECONDS) -> tuple[int, str]:
    """The size of a remote file whose server answers range requests, and the address it is served
    from after redirects. Each redirect is checked like the link itself, and the readers read the
    final address. The first address is checked here unless the caller already has.

    @raises RemoteError when a hop is private, or the server answers a web page or the whole file
    """
    if not allow_private and not first_hop_checked:
        _public(url)

    class Hops(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201, PLR0913
            if not allow_private:
                _public(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(Hops)
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "3D-GNOME-NG"})
    try:
        with opener.open(req, timeout=timeout) as r:
            final = r.geturl()
            kind = r.headers.get("Content-Type", "")
            if kind.startswith("text/html"):
                raise RemoteError(f"{_host(url)} answers with a web page, not a file.", "Give the direct link that starts the download, not the page that offers it.")
            span = r.headers.get("Content-Range", "")
            if r.status != 206 or "/" not in span:
                raise RemoteError(
                    f"{_host(url)} does not answer range requests",
                    "The server would have to download the whole file. Host the file where range requests work, or choose Local .hic file and the page reads your region in the browser.",
                )
            total = span.rsplit("/", 1)[1]
            if not total.isdigit():
                raise RemoteError(f"{_host(url)} does not give the size of the file.", "Host the file where range requests work.")
            return int(total), final
    except urllib.error.HTTPError as e:
        raise RemoteError(f"{_host(url)} answers {e.code} {e.reason}.", "Check that the link opens the file in a browser without a sign in.") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if isinstance(getattr(e, "reason", None), RemoteError):
            raise e.reason from e  # type: ignore[misc]
        raise RemoteError(f"{_host(url)} cannot be reached.", "Check the link and try again.") from e


def check_range(url: str, timeout: float = HEAD_SECONDS) -> int:
    """The size of a remote file whose server answers range requests."""
    return probe(url, allow_private=True, timeout=timeout)[0]


def _chrom_name(names: list[str], chrom: str) -> str:
    """The file's name for a chromosome, with or without the chr prefix."""
    bare = chrom.removeprefix("chr")
    for n in names:
        if n == chrom or n == bare or n == f"chr{bare}":
            return n
    raise RemoteError(f"The file has no chromosome {chrom}.", f"It names {', '.join(names[:5])}. Check the genome build of the file.")


def _no_res(have: list[int]) -> RemoteError:
    return RemoteError(
        "The file has no 25 kb resolution.",
        f"It has {', '.join(_kb(r) for r in have)}. Give a file binned at 25 kb, or extract a BEDPE of the region at 25 kb.",
    )


def read_mcool(url: str, chrom: str, start: int, end: int, res: int = RES) -> Window:
    """The window's pixels of a remote .mcool or .cool, read through h5py on an HTTP file."""
    import fsspec
    import h5py

    with fsspec.open(url, "rb", block_size=1 << 20) as f, h5py.File(f, "r") as h:
        if "resolutions" in h:
            have = sorted(int(k) for k in h["resolutions"])
            if res not in have:
                raise _no_res(have)
            g = h["resolutions"][str(res)]
        else:
            have = [int(h.attrs["bin-size"])]
            if have[0] != res:
                raise _no_res(have)
            g = h
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in g["chroms/name"][:]]
        ci = names.index(_chrom_name(names, chrom))
        offsets = g["indexes/chrom_offset"][:]
        first, last = int(offsets[ci]), int(offsets[ci + 1])
        lo = min(first + start // res, last)
        hi = min(first + -(-end // res), last)
        off = g["indexes/bin1_offset"][lo : hi + 1]
        p0, p1 = (int(off[0]), int(off[-1])) if hi > lo else (0, 0)
        b1 = g["pixels/bin1_id"][p0:p1]
        b2 = g["pixels/bin2_id"][p0:p1]
        cnt = g["pixels/count"][p0:p1]
    keep = b2 < hi
    return Window((b1[keep] - first) * res, (b2[keep] - first) * res, np.rint(cnt[keep]).astype(np.int64), have)


def read_hic(url: str, chrom: str, start: int, end: int, res: int = RES) -> Window:
    """The window's observed counts of a remote .hic, read with hic-straw."""
    import hicstraw

    h = hicstraw.HiCFile(url)
    have = sorted(int(r) for r in h.getResolutions())
    if res not in have:
        raise _no_res(have)
    names = [c.name for c in h.getChromosomes()]
    name = _chrom_name(names, chrom)
    m = h.getMatrixZoomData(name, name, "observed", "NONE", "BP", res)
    recs = m.getRecords(start, end - 1, start, end - 1)
    b1 = np.array([r.binX for r in recs], dtype=np.int64)
    b2 = np.array([r.binY for r in recs], dtype=np.int64)
    cnt = np.rint([r.counts for r in recs]).astype(np.int64)
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    return Window(lo, hi, cnt, have)


def read_window(url: str, kind: str, chrom: str, start: int, end: int, seconds: float = READ_SECONDS) -> Window:
    """The window of a remote map, given up after the time limit."""
    fn = read_hic if kind == ".hic" else read_mcool
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, url, chrom, start, end).result(timeout=seconds)
    except FutureTimeout as e:
        raise RemoteError(f"The read of {_host(url)} took longer than {int(seconds) // 60} minutes.", "Try again later, or choose Local .hic file.") from e
    except RemoteError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RemoteError(f"The file at {_host(url)} could not be read as {kind}.", f"The reader says {e}.") from e
    finally:
        pool.shutdown(wait=False)


def thin(counts: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Counts reduced to a target total by sampling contacts without replacement. A map at or under
    the target is kept whole."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() <= target:
        return counts
    return np.random.default_rng(seed).multivariate_hypergeometric(counts, target)


@lru_cache(maxsize=64)
def hosted_density(singletons: Path, chrom: str) -> float:
    """Contacts per Mb of a hosted map on one chromosome, its total count over its covered span."""
    t = pd.read_csv(singletons, sep=r"\s+", header=None, usecols=[0, 3, 5, 6], comment="#")
    t = t[(t[0] == chrom) & (t[3] == chrom)]
    if t.empty:
        t = pd.read_csv(singletons, sep=r"\s+", header=None, usecols=[0, 3, 5, 6], comment="#")
        t = t[t[0] == t[3]]
        span = t.groupby(0)[5].max().sum()
    else:
        span = t[5].max()
    return float(t[6].sum()) / max(span / 1e6, 1e-6)


def _kb(bp: int) -> str:
    return f"{bp / 1e6:g} Mb" if bp >= 1_000_000 else f"{bp / 1e3:g} kb"


def link_window(url: str, chrom: str, start: int, end: int, per_mb: float, density_from: str, allow_private: bool) -> tuple[bytes, int, str]:
    """Reads a remote map's window into BEDPE bytes at 25 kb, thinned to per_mb contacts per Mb.
    Returns the bytes, the file size and the summary.

    @param density_from the name of the hosted map the density comes from, for the summary
    @raises RemoteError naming the problem and its fix
    """
    kind = check_link(url, allow_private)
    size, final = probe(url, allow_private, first_hop_checked=True)
    w = read_window(final, kind, chrom, start, end)
    span = f"{chrom}:{start:,}-{end:,}"
    read = int(w.counts.sum())
    if read == 0:
        raise RemoteError(f"The file has no contacts in {span}.", "Check the genome build of the file and the region.")
    seed = int.from_bytes(hashlib.blake2b(f"{url} {span}".encode(), digest_size=8).digest(), "little")
    target = round(per_mb * (end - start) / 1e6)
    kept = thin(w.counts, target, seed)
    out = io.StringIO()
    for a, b, n in zip(w.bin1, w.bin2, kept, strict=True):
        if n > 0:
            out.write(f"{chrom}\t{a}\t{a + RES}\t{chrom}\t{b}\t{b + RES}\t{n}\n")
    res_words = f"Resolutions from {_kb(w.resolutions[0])} to {_kb(w.resolutions[-1])}, {_kb(RES)} is used."
    if kept.sum() < read:
        how = f"{read:,} contacts in {span} are read and {int(kept.sum()):,} are kept, {per_mb:,.0f} per Mb as in the {density_from} map."
    else:
        how = f"{read:,} contacts in {span} are read and all are kept, under the {per_mb:,.0f} per Mb of the {density_from} map."
    return out.getvalue().encode(), size, f"{res_words} {how}"
