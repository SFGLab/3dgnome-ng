"""Contact maps by link.

The server reads one window of a remote .hic or .mcool file over HTTP range requests and keeps
its bin pairs at 25 kb, thinned to the density of the hosted maps. A link must be http or
https, end in .hic, .mcool or .cool and resolve to a public address. Every connection goes to
the address that was checked, through a local range proxy.
"""

# h5py and hic-straw state the types of their files too loosely for these checks.
# pyright: reportGeneralTypeIssues=false, reportIndexIssue=false, reportArgumentType=false, reportOperatorIssue=false, reportCallIssue=false

from __future__ import annotations

import functools
import hashlib
import http.client
import io
import ipaddress
import multiprocessing
import secrets
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from .density import thin

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


def _resolve(host: str, port: int) -> list[str]:
    """Returns the addresses of a host, in the order the resolver gives them."""
    return [str(i[4][0]) for i in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def _checked(host: str, port: int) -> list[str]:
    """Returns the addresses of a host. Refuses a host with a private, loopback or link local address."""
    try:
        addrs = _resolve(host, port)
    except OSError as e:
        raise RemoteError(f"{host} does not resolve.", "Check the address of the link.") from e
    for a in addrs:
        if not ipaddress.ip_address(a).is_global:
            raise RemoteError(f"{host} is a private address.", "Give a link the public internet reaches.")
    return addrs


def _public(url: str) -> None:
    """Refuses a link whose host resolves to a private, loopback or link local address."""
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RemoteError("The link must start with http or https.", "Give the web address of the file, for example a 4DN, ENCODE or GEO download link.")
    _checked(u.hostname, u.port or (443 if u.scheme == "https" else 80))


def _pinned_connect(allow_private: bool):  # noqa: ANN202
    """Connects as socket.create_connection does. It resolves once and connects only to checked addresses."""

    def connect(address: tuple[str, int], timeout: float = HEAD_SECONDS, source: object = None) -> socket.socket:
        host, port = address
        addrs = _resolve(host, port) if allow_private else _checked(host, port)
        err: OSError | None = None
        for a in addrs:
            try:
                return socket.create_connection((a, port), timeout)
            except OSError as e:
                err = e
        raise err or OSError(f"no address for {host}")

    return connect


class _HTTP(http.client.HTTPConnection):
    def __init__(self, *a: object, allow_private: bool, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.allow_private = allow_private
        self._create_connection = _pinned_connect(allow_private)


class _HTTPS(http.client.HTTPSConnection):
    def __init__(self, *a: object, allow_private: bool, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.allow_private = allow_private
        self._create_connection = _pinned_connect(allow_private)


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self.allow = allow_private

    def http_open(self, req):  # noqa: ANN001, ANN201
        return self.do_open(functools.partial(_HTTP, allow_private=self.allow), req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, allow_private: bool) -> None:
        super().__init__()
        self.allow = allow_private

    def https_open(self, req):  # noqa: ANN001, ANN201
        return self.do_open(functools.partial(_HTTPS, allow_private=self.allow), req, context=self._context)  # type: ignore[attr-defined]


def _opener(allow_private: bool) -> urllib.request.OpenerDirector:
    """Returns an opener whose every connection, redirects included, goes to a checked address."""
    return urllib.request.build_opener(_HTTPHandler(allow_private), _HTTPSHandler(allow_private))


def check_link(url: str, allow_private: bool) -> str:
    """Refuses a link that is not http or https, not a contact map, or not public. Returns its kind.

    @raises RemoteError with the problem
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


def probe(url: str, allow_private: bool, timeout: float = HEAD_SECONDS) -> tuple[int, str]:
    """Returns the size of a remote file whose server answers range requests and the address that
    serves it after redirects.

    @raises RemoteError when a hop is private, or the server answers a web page or the whole file
    """
    opener = _opener(allow_private)
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


def _chrom_name(names: list[str], chrom: str) -> str:
    """Returns the name of a chromosome in the file, with or without the chr prefix."""
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


def read_mcool(url: str, chrom: str, start: int, end: int, res: int = RES, size: int | None = None) -> Window:
    """Reads the pixels of the window of a remote .mcool or .cool, through h5py on an HTTP file. A
    known size spares the reader its own request for it."""
    import fsspec
    import h5py

    with fsspec.filesystem("http").open(url, "rb", block_size=1 << 20, size=size) as f, h5py.File(f, "r") as h:
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
    """Reads the observed counts of the window of a remote .hic, with hic-straw."""
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


class _Proxy:
    """A range proxy on the loopback interface for the readers. hic-straw reads through curl and
    h5py through aiohttp. This module cannot make either connect to a checked address. Each reads
    http://127.0.0.1:<port>/<token>/<name> and the proxy fetches the range with the checked opener.
    A token is valid for one read."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[str, bool]] = {}
        self.idle: dict[tuple[str, str, int, bool], list[http.client.HTTPConnection]] = {}
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def _relay(self, head: bool) -> None:
                token = self.path.lstrip("/").split("/", 1)[0]
                route = proxy.routes.get(token)
                if route is None:
                    self.send_error(404)
                    return
                url, allow = route
                headers = {"User-Agent": "3D-GNOME-NG"}
                if self.headers.get("Range"):
                    headers["Range"] = self.headers["Range"]
                try:
                    conn, r = proxy.fetch(url, allow, "HEAD" if head else "GET", headers)
                except (RemoteError, OSError, http.client.HTTPException):
                    self.send_error(502)
                    return
                # A redirect would send the reader past the checked connection. The link was
                # followed to its final address before the read.
                status = 502 if 300 <= r.status < 400 else r.status
                self.send_response(status)
                for k in ("Content-Range", "Content-Length", "Accept-Ranges", "Content-Type", "ETag", "Last-Modified"):
                    if status != 502 and r.headers.get(k):
                        self.send_header(k, r.headers[k])
                self.end_headers()
                try:
                    while status != 502 and not head and (chunk := r.read(1 << 20)):
                        self.wfile.write(chunk)
                    r.read()
                    proxy.put_back(url, conn, r)
                except OSError:
                    conn.close()

            def do_GET(self) -> None:  # noqa: N802
                self._relay(False)

            def do_HEAD(self) -> None:  # noqa: N802
                self._relay(True)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def _key(url: str, allow: bool) -> tuple[str, str, int, bool]:
        u = urlparse(url)
        return (u.scheme, u.hostname or "", u.port or (443 if u.scheme == "https" else 80), allow)

    def fetch(self, url: str, allow: bool, method: str, headers: dict[str, str]) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        """Returns a response for one range, on a kept connection to the host when one is idle. Each
        new connection goes to a checked address."""
        key = self._key(url, allow)
        u = urlparse(url)
        path = u.path + (f"?{u.query}" if u.query else "")
        with self.lock:
            pool = self.idle.setdefault(key, [])
            conn = pool.pop() if pool else None
        for attempt in range(2):
            if conn is None:
                cls = _HTTPS if key[0] == "https" else _HTTP
                conn = cls(key[1], key[2], timeout=HEAD_SECONDS, allow_private=allow)
            try:
                conn.request(method, path, headers=headers)
                return conn, conn.getresponse()
            except (OSError, http.client.HTTPException):
                conn.close()
                conn = None
                if attempt:
                    raise
        raise OSError("unreachable")

    def put_back(self, url: str, conn: http.client.HTTPConnection, r: http.client.HTTPResponse) -> None:
        if r.will_close:
            conn.close()
            return
        with self.lock:
            self.idle.setdefault(self._key(url, getattr(conn, "allow_private", False)), []).append(conn)

    def add(self, url: str, allow_private: bool) -> tuple[str, str]:
        """Returns a local address that serves `url` and its token."""
        token = secrets.token_urlsafe(16)
        self.routes[token] = (url, allow_private)
        name = urlparse(url).path.rsplit("/", 1)[-1] or "file"
        return f"http://127.0.0.1:{self.server.server_address[1]}/{token}/{name}", token


_proxy_lock = threading.Lock()
_proxy: _Proxy | None = None


def _local(url: str, allow_private: bool) -> tuple[str, str]:
    global _proxy
    with _proxy_lock:
        if _proxy is None:
            _proxy = _Proxy()
    return _proxy.add(url, allow_private)


def _child(kind: str, url: str, chrom: str, start: int, end: int, size: int, out, res: int = RES) -> None:  # noqa: ANN001, PLR0913
    """Runs one reader in a child process and sends back its window or its refusal."""
    try:
        w = read_hic(url, chrom, start, end, res) if kind == ".hic" else read_mcool(url, chrom, start, end, res, size)
        out.send(("ok", w))
    except RemoteError as e:
        out.send(("refused", str(e), e.fix))
    except Exception as e:  # noqa: BLE001
        out.send(("failed", str(e)))


def read_window(url: str, kind: str, chrom: str, start: int, end: int, size: int, allow_private: bool, seconds: float = READ_SECONDS, res: int = RES) -> Window:
    """Reads the window of a remote map in a child process through the local proxy and stops it
    after the time limit. hic-straw holds the interpreter lock while it waits on the network. In
    this process it would stall the proxy it reads through."""
    local, token = _local(url, allow_private)
    ctx = multiprocessing.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(kind, local, chrom, start, end, size, send, res), daemon=True)
    proc.start()
    send.close()
    try:
        if not recv.poll(seconds):
            raise RemoteError(f"The read of {_host(url)} took longer than {int(seconds) // 60} minutes.", "Try again later, or choose Local .hic file.")
        try:
            answer = recv.recv()
        except EOFError as e:
            raise RemoteError(f"The file at {_host(url)} could not be read as {kind}.", "The reader stopped without an answer.") from e
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
        if _proxy is not None:
            _proxy.routes.pop(token, None)
    if answer[0] == "ok":
        return answer[1]
    if answer[0] == "refused":
        raise RemoteError(answer[1], answer[2])
    raise RemoteError(f"The file at {_host(url)} could not be read as {kind}.", f"The reader says {answer[1]}.")


def _kb(bp: int) -> str:
    return f"{bp / 1e6:g} Mb" if bp >= 1_000_000 else f"{bp / 1e3:g} kb"


def link_window(url: str, chrom: str, start: int, end: int, per_mb: float, density_from: str, allow_private: bool) -> tuple[bytes, int, str]:
    """Reads the window of a remote map into BEDPE bytes at 25 kb, thinned to per_mb contacts per
    Mb. Returns the bytes, the file size and the summary.

    @param url the link
    @param chrom the chromosome of the window
    @param start the start of the window in bp
    @param end the end of the window in bp
    @param per_mb the contacts per Mb to keep
    @param density_from the name of the hosted map that gives the density, for the summary
    @param allow_private true to read a private or loopback address, for tests
    @raises RemoteError with the problem and its fix
    """
    kind = check_link(url, allow_private)
    size, final = probe(url, allow_private)
    w = read_window(final, kind, chrom, start, end, size, allow_private)
    # The hosted maps hold no pair within one bin and the engine reads every map alike.
    off = w.bin1 != w.bin2
    w = Window(w.bin1[off], w.bin2[off], w.counts[off], w.resolutions)
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
