"""Contact maps by link. The tests cover range requests, the window of an .mcool read over HTTP, the
refusals and the thinning to the density of the hosted maps.

A .hic link is checked by hand against a public 4DN file. There is no writer for a .hic fixture.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fixture import make_data


def serve(root: Path, ranges: bool, html: bool = False, head: bool = True, slow: float = 0.0) -> tuple[ThreadingHTTPServer, str]:
    """Starts a file server on a free port that answers range requests, or answers every request
    whole. Without head it refuses HEAD requests, as an S3 link signed for GET does."""

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
    """Writes the contact pairs of the fixture as a multi resolution cooler at 5 and 25 kb."""
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


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
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
    return SimpleNamespace(remote=remote, url=url, size=size, url_whole=url_whole, data=data, root=root, url_signed=url_signed, url_page=url_page, url_slow=url_slow, with_ranges=with_ranges, whole=whole, page=page, signed=signed, slow=slow)


def test_a_server_with_range_requests_gives_the_file_size(shared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    remote, url, size, url_whole = shared.remote, shared.url, shared.size, shared.url_whole
    assert remote.probe(f"{url}/test.mcool", allow_private=True)[0] == size, "a server with range requests gives the file size"
    try:
        remote.probe(f"{url_whole}/test.mcool", allow_private=True)
        assert False, "a server without range requests is refused"
    except remote.RemoteError as e:
        assert "does not answer range requests" in str(e) and "Local .hic file" in e.fix, "a server without range requests is refused: " + f"{e} {e.fix}"

    hop = f"{url}/go/to-{url.removeprefix('http://')}/test.mcool"
    size_seen, final = remote.probe(hop, allow_private=True)
    assert size_seen == size and final == f"{url}/test.mcool", "a redirect is followed to the file it names: " + str(final)
    seen: list[str] = []
    real_resolve = remote._resolve
    with monkeypatch.context() as m:
        m.setattr(remote, "_resolve", lambda host, p: seen.append(host) or real_resolve(host, p))
        remote.probe(hop.replace("127.0.0.1", "localhost"), allow_private=True)
    assert seen == ["localhost", "localhost"], "every connection, the redirect's too, resolves through the checked resolver: " + str(seen)
    try:
        remote.probe(hop.replace("127.0.0.1", "localhost", 1), allow_private=False)
        assert False, "a private address is refused at connect"
    except remote.RemoteError as e:
        assert "private address" in str(e), "a private address is refused at connect: " + str(e)


def test_thinning_keeps_the_target_total_never_above_a_count() -> None:
    from gnome3d_web.density import thin

    thinned = thin(np.array([500, 300, 200, 0]), 100, seed=7)
    assert thinned.sum() == 100 and bool((thinned <= [500, 300, 200, 0]).all()), "thinning keeps the target total, never above a count: " + str(thinned)
    assert (thin(np.array([500, 300, 200]), 100, seed=7) == thin(np.array([500, 300, 200]), 100, seed=7)).all(), "thinning is the same for the same seed"
    assert (thin(np.array([3, 4]), 100, seed=1) == [3, 4]).all(), "a map under the target keeps every contact"


def test_an_mcool_link_becomes_a_contacts_upload(shared: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    data, root, url, size, url_signed, url_whole, url_page, url_slow, remote, with_ranges, whole, page, signed, slow = shared.data, shared.root, shared.url, shared.size, shared.url_signed, shared.url_whole, shared.url_page, shared.url_slow, shared.remote, shared.with_ranges, shared.whole, shared.page, shared.signed, shared.slow
    os.environ.update(
        DATA_DIR=str(data), RESULTS_DIR=str(root / "results"), QUEUE="inline", FRONTEND_DIST=str(root / "missing"), LINK_ALLOW_PRIVATE="1",
        ENGINE_CMD=f"{sys.executable} {Path(__file__).resolve().parent / 'fake_engine.py'}",
    )  # fmt: skip
    from fastapi.testclient import TestClient

    from gnome3d_web.app import create_app

    c = TestClient(create_app())

    def link(u: str, region: str = "chr1:0-1000000", client: TestClient = c):  # noqa: ANN202
        return client.post("/api/uploads/link", data={"url": u, "cell": "GMX", "region": region})

    r = link(f"{url}/test.mcool")
    j = r.json()
    assert r.status_code == 201 and j["ok"] and j["kind"] == "contacts", "an .mcool link becomes a contacts upload: " + r.text
    assert j.get("name") == "test.mcool" and j.get("size") == size, "the upload carries the file's name and size: " + str(j)
    assert j["summary"].startswith("Resolutions from 5 kb to 25 kb, 25 kb is used."), "the summary names the resolutions and the one used: " + str(j["summary"])
    assert "30 contacts in chr1:0-1,000,000 are read" in j["summary"], "the summary counts the contacts read in the window: " + str(j["summary"])
    stored = root / "results" / "uploads" / j["id"]
    from gnome3d_web.uploads import store as upload_store

    pairs = pd.read_csv(upload_store.data_file(stored), sep="\t", header=None)
    assert len(pairs) == 10 and set(pairs[2] - pairs[1]) == {25_000} and int(pairs[6].sum()) == 30, "the window's off diagonal bin pairs are stored at 25 kb: " + str(pairs.to_string())

    r = link(f"{url_signed}/test.mcool")
    assert r.status_code == 201 and r.json()["ok"], "an .mcool link whose server refuses HEAD is read: " + r.text

    r = link(f"{url_whole}/test.mcool")
    assert r.status_code == 400 and "does not answer range requests" in r.json()["error"] and r.json()["fix"], "a link without range requests is a 400 naming it: " + r.text
    r = link(f"{url_page}/test.mcool")
    assert r.status_code == 400 and "web page" in r.json()["error"], "a link that answers a web page is a 400 naming it: " + r.text
    r = link(f"{url}/notes.txt")
    assert r.status_code == 400 and ".hic or .mcool" in r.json()["error"], "a link to a file that is not .hic or .mcool is refused: " + r.text
    r = link(f"{url}/test.mcool", region="chr2:0-1000000")
    assert r.status_code == 400 and "chr2" in r.json()["error"], "a chromosome the file lacks is named: " + r.text
    r = link("file:///etc/passwd")
    assert r.status_code == 400 and "http" in r.json()["error"], "a link that is not http or https is refused: " + r.text

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
    assert sorted(a[0] for a in answers) == [201, 503] and busy[0][1] < 1.0, "a link read over the limit is refused at once: " + str(answers)

    os.environ.pop("LINK_ALLOW_PRIVATE")
    strict = TestClient(create_app())
    r = link(f"{url}/test.mcool", client=strict)
    assert r.status_code == 400 and "private" in r.json()["error"], "a link to a private address is refused by default: " + r.text

    port = url.rsplit(":", 1)[1]
    answers_dns = iter([["8.8.8.8"]])
    real = remote._resolve
    with monkeypatch.context() as m:
        m.setattr(remote, "_resolve", lambda host, p: next(answers_dns, ["127.0.0.1"]) if host == "rebind.test" else real(host, p))
        r = link(f"http://rebind.test:{port}/test.mcool", client=strict)
    assert r.status_code == 400 and "private" in r.json()["error"], "a name that resolves public at the check and private at the read is refused: " + r.text
    for srv in (with_ranges, whole, page, signed, slow):
        srv.shutdown()
