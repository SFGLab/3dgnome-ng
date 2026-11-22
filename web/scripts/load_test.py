"""Submits several jobs to a running site at once and reports when each started and ended.

    python web/scripts/load_test.py http://127.0.0.1:8766 GM12878 chr1:1000000-4000000 chr2:5000000-8000000 ...

Every region becomes one job of its own with CTCF and RNAPII loops and 10 structures. The script
waits for all of them and prints one line per job with its times from the first start, then the
total. The site is the address of its root, with its base path when it has one.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def submit(site: str, cell: str, region: str) -> str:
    """Posts one job and returns its path, or the refusal of the site."""
    fields = {"cell_type": cell, "region": region, "factors": "CTCF+RNAPII", "n_models": "10", "fresh": "1"}
    try:
        with urllib.request.urlopen(urllib.request.Request(site + "/api/jobs", data=urllib.parse.urlencode(fields).encode(), method="POST")) as r:
            return json.load(r)["url"]
    except urllib.error.HTTPError as e:
        return f"error {e.code} {e.read().decode(errors='replace')[:200]}"


def summary(site: str, job: str) -> dict:
    with urllib.request.urlopen(f"{site}/api{job}/summary") as r:
        return json.load(r)


def main() -> int:
    site, cell, regions = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3:]
    t0 = time.time()
    with ThreadPoolExecutor(len(regions)) as ex:
        answers = list(ex.map(lambda r: submit(site, cell, r), regions))
    print("submitted:", answers, flush=True)
    jobs = [a for a in answers if a.startswith("/")]
    if not jobs:
        return 1
    while True:
        states = [summary(site, j) for j in jobs]
        if all(s["state"] in ("done", "failed") for s in states):
            break
        time.sleep(5)
    first = min(s["started"] for s in states)
    for j, s in sorted(zip(jobs, states, strict=True), key=lambda x: x[1]["started"]):
        print(
            f"{s['state']:6} {s.get('region') or j:28} start {s['started'] - first:5.0f} s, "
            f"end {s['finished'] - first:5.0f} s, {s['finished'] - s['started']:4.0f} s  {(s.get('error') or '')[:100]}"
        )
    took = time.time() - t0
    print(f"{len(jobs)} jobs in {took:.0f} s, {3600 * len(jobs) / took:.1f} jobs per hour")
    return 0 if len(jobs) == len(regions) and all(s["state"] == "done" for s in states) else 1


if __name__ == "__main__":
    sys.exit(main())
