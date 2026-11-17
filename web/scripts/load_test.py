"""Submit several jobs to a running site at once and report when each started and ended.

    python web/scripts/load_test.py http://127.0.0.1:8766 <results dir> GM12878 chr1:1000000-4000000 chr2:5000000-8000000 ...

Every region becomes one job with CTCF and RNAPII loops and 10 structures. The script waits for
all of them and prints one line per job with its times from the first start, then the total.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        return None


def submit(site: str, cell: str, region: str) -> str:
    """Posts one job and returns its job path, or the error page's status."""
    body = urllib.parse.urlencode({"cell_type": cell, "region": region, "factors": "CTCF+RNAPII", "n_models": "10"}).encode()
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(site + "/jobs", data=body, method="POST"))
    except urllib.error.HTTPError as e:
        if e.code == 303:
            return e.headers["Location"]
        return f"error {e.code}"
    return "error no redirect"


def main() -> int:
    site, results, cell, regions = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    t0 = time.time()
    with ThreadPoolExecutor(len(regions)) as ex:
        paths = list(ex.map(lambda r: submit(site, cell, r), regions))
    print("submitted:", paths, flush=True)
    dirs = [results / "jobs" / p.rsplit("/", 1)[-1] for p in paths if p.startswith("/jobs/")]
    while True:
        states = [json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else {"state": "queued"} for d in dirs]
        if all(s["state"] in ("done", "failed") for s in states):
            break
        time.sleep(5)
    first = min(s["started"] for s in states)
    for d, s in sorted(zip(dirs, states, strict=True), key=lambda x: x[1]["started"]):
        print(
            f"{s['state']:6} {s.get('region', d.name):28} start {s['started'] - first:5.0f} s, "
            f"end {s['finished'] - first:5.0f} s, {s['finished'] - s['started']:4.0f} s  {s.get('error', '')[:100]}"
        )
    print(f"{len(dirs)} jobs in {time.time() - t0:.0f} s, {3600 * len(dirs) / (time.time() - t0):.1f} jobs per hour")
    return 0


if __name__ == "__main__":
    sys.exit(main())
