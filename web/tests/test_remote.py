"""Contact maps by link: range requests, the window of an .mcool read over HTTP, the refusals and
the thinning to the density of the hosted maps.

    python web/tests/test_remote.py

A .hic link is checked by hand against a public 4DN file. There is no writer for a
.hic fixture.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from _check import check, done
from fixture import make_data


def serve(root: Path, ranges: bool, html: bool = False, head: bool = True, slow: float = 0.0) -> tuple[ThreadingHTTPServer, str]:
    """A file server on a free port that answers range requests, or answers every request whole.
    Without head it refuses HEAD requests, as an S3 link signed for GET does."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _send(self, head: bool) -> None:
            time.sleep(slow)
            if html:
                body = b"<!doctype html><html><body>Sign in</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            p = root / self.path.lstrip("/")
            data = p.read_bytes()
            rng = self.headers.get("Range")
            if ranges and rng and rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                lo, hi = int(a), min(int(b) if b else len(data) - 1, len(data) - 1)
                part = data[lo : hi + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(data)}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                if not head:
                    self.wfile.write(part)
                return
            self.send_response(200)
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head:
                self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/go/"):
                self.send_response(302)
                self.send_header("Location", self.path[3:].replace("/to-", "http://", 1))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(False)

        def do_HEAD(self) -> None:  # noqa: N802
            if not head:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(True)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def write_mcool(singletons: Path, out: Path) -> None:
    """The fixture's contact pairs as a multi resolution cooler at 5 and 25 kb."""
    import cooler

    t = pd.read_csv(singletons, sep="\t", header=None)
    for res in (5_000, 25_000):
        bins = pd.DataFrame({"chrom": "chr1", "start": np.arange(0, 4_000_000, res)})
        bins["end"] = bins["start"] + res
        diag = pd.DataFrame([["chr1", 600_000, 600_000 + res, "chr1", 600_000, 600_000 + res, 40]])
        both = pd.concat([t, diag], ignore_index=True)
        b1, b2 = both[1].to_numpy() // res, both[4].to_numpy() // res
        pixels = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2), "count": both[6].to_numpy()})
        pixels = pixels.groupby(["bin1_id", "bin2_id"], as_index=False)["count"].sum().sort_values(["bin1_id", "bin2_id"])
        cooler.create_cooler(f"{out}::resolutions/{res}", bins, pixels, ordered=True, mode="a")


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = make_data(root / "data")
        files = root / "files"
        files.mkdir()
        write_mcool(data / "cell_types" / "GMX" / "singletons.bedpe", files / "test.mcool")
        (files / "notes.txt").write_text("not a map\n")
        size = (files / "test.mcool").stat().st_size
        with_ranges, url = serve(files, ranges=True)
        whole, url_whole = serve(files, ranges=False)
        page, url_page = serve(files, ranges=True, html=True)
        signed, url_signed = serve(files, ranges=True, head=False)
        slow, url_slow = serve(files, ranges=True, slow=1.5)

        from gnome3d_web import remote

        check("a server with range requests gives the file size", remote.check_range(f"{url}/test.mcool") == size)
        try:
            remote.check_range(f"{url_whole}/test.mcool")
            check("a server without range requests is refused", False)
        except remote.RemoteError as e:
            check("a server without range requests is refused", "does not answer range requests" in str(e) and "Local .hic file" in e.fix, f"{e} {e.fix}")

        hop = f"{url}/go/to-{url.removeprefix('http://')}/test.mcool"
        size_seen, final = remote.probe(hop, allow_private=True)
        check("a redirect is followed to the file it names", size_seen == size and final == f"{url}/test.mcool", final)
        seen: list[str] = []
        real_resolve = remote._resolve
        remote._resolve = lambda host, p: seen.append(host) or real_resolve(host, p)
        try:
            remote.probe(hop.replace("127.0.0.1", "localhost"), allow_private=True)
        finally:
            remote._resolve = real_resolve
        check("every connection, the redirect's too, resolves through the checked resolver", seen == ["localhost", "localhost"], str(seen))
        try:
            remote.probe(hop.replace("127.0.0.1", "localhost", 1), allow_private=False)
            check("a private address is refused at connect", False)
        except remote.RemoteError as e:
            check("a private address is refused at connect", "private address" in str(e), str(e))

        thinned = remote.thin(np.array([500, 300, 200, 0]), 100, seed=7)
        check("thinning keeps the target total, never above a count", thinned.sum() == 100 and bool((thinned <= [500, 300, 200, 0]).all()), str(thinned))
        check("thinning is the same for the same seed", (remote.thin(np.array([500, 300, 200]), 100, seed=7) == remote.thin(np.array([500, 300, 200]), 100, seed=7)).all())
        check("a map under the target keeps every contact", (remote.thin(np.array([3, 4]), 100, seed=1) == [3, 4]).all())

        os.environ.update(
            DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), LINK_ALLOW_PRIVATE="1",
            ENGINE_CMD=f"{Path(__file__).resolve().parents[2] / '.venv' / 'bin' / 'python'} {Path(__file__).resolve().parent / 'fake_engine.py'}",
        )  # fmt: skip
        from fastapi.testclient import TestClient
        from gnome3d_web.app import create_app

        c = TestClient(create_app())

        def link(u: str, region: str = "chr1:0-1000000", client: TestClient = c):  # noqa: ANN202
            return client.post("/api/uploads/link", data={"url": u, "cell": "GMX", "region": region})

        r = link(f"{url}/test.mcool")
        j = r.json()
        check("an .mcool link becomes a contacts upload", r.status_code == 201 and j["ok"] and j["kind"] == "contacts", r.text)
        check("the upload carries the file's name and size", j.get("name") == "test.mcool" and j.get("size") == size, str(j))
        check("the summary names the resolutions and the one used", j["summary"].startswith("Resolutions from 5 kb to 25 kb, 25 kb is used."), j["summary"])
        check("the summary counts the contacts read in the window", "30 contacts in chr1:0-1,000,000 are read" in j["summary"], j["summary"])
        stored = root / "results" / "uploads" / j["id"]
        from gnome3d_web import owndata

        pairs = pd.read_csv(owndata.data_file(stored), sep="\t", header=None)
        check("the window's off diagonal bin pairs are stored at 25 kb", len(pairs) == 10 and set(pairs[2] - pairs[1]) == {25_000} and int(pairs[6].sum()) == 30, pairs.to_string())

        r = link(f"{url_signed}/test.mcool")
        check("an .mcool link whose server refuses HEAD is read", r.status_code == 201 and r.json()["ok"], r.text)

        r = link(f"{url_whole}/test.mcool")
        check("a link without range requests is a 400 naming it", r.status_code == 400 and "does not answer range requests" in r.json()["error"] and r.json()["fix"], r.text)
        r = link(f"{url_page}/test.mcool")
        check("a link that answers a web page is a 400 naming it", r.status_code == 400 and "web page" in r.json()["error"], r.text)
        r = link(f"{url}/notes.txt")
        check("a link to a file that is not .hic or .mcool is refused", r.status_code == 400 and ".hic or .mcool" in r.json()["error"], r.text)
        r = link(f"{url}/test.mcool", region="chr2:0-1000000")
        check("a chromosome the file lacks is named", r.status_code == 400 and "chr2" in r.json()["error"], r.text)
        r = link("file:///etc/passwd")
        check("a link that is not http or https is refused", r.status_code == 400 and "http" in r.json()["error"], r.text)

        os.environ["LINK_READS"] = "1"
        limited = TestClient(create_app())
        os.environ.pop("LINK_READS")
        answers: list[tuple[int, float]] = []

        def timed() -> None:
            t = time.time()
            answers.append((link(f"{url_slow}/test.mcool", client=limited).status_code, time.time() - t))

        pair = [threading.Thread(target=timed) for _ in range(2)]
        for th in pair:
            th.start()
            time.sleep(0.3)
        for th in pair:
            th.join()
        busy = [a for a in answers if a[0] == 503]
        check("a link read over the limit is refused at once", sorted(a[0] for a in answers) == [201, 503] and busy[0][1] < 1.0, str(answers))

        os.environ.pop("LINK_ALLOW_PRIVATE")
        strict = TestClient(create_app())
        r = link(f"{url}/test.mcool", client=strict)
        check("a link to a private address is refused by default", r.status_code == 400 and "private" in r.json()["error"], r.text)

        port = url.rsplit(":", 1)[1]
        answers_dns = iter([["8.8.8.8"]])
        real = remote._resolve
        remote._resolve = lambda host, p: next(answers_dns, ["127.0.0.1"]) if host == "rebind.test" else real(host, p)
        try:
            r = link(f"http://rebind.test:{port}/test.mcool", client=strict)
        finally:
            remote._resolve = real
        check("a name that resolves public at the check and private at the read is refused", r.status_code == 400 and "private" in r.json()["error"], r.text)
        for srv in (with_ranges, whole, page, signed, slow):
            srv.shutdown()
    return done()


if __name__ == "__main__":
    sys.exit(main())
