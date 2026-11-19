"""The local queue: a fixed set of slots, one GPU per slot, jobs waiting for a free slot.

    python web/tests/test_queue.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from _check import check, done
from fixture import make_data

from gnome3d_web.jobs import JobRequest
from gnome3d_web.queue import LocalPool, resume_interrupted

FAKE = [str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"), str(Path(__file__).resolve().parent / "fake_engine.py")]


def most_at_once(spans: list[tuple[float, float, str]], gpu: str | None = None) -> int:
    """The largest number of spans running at one moment, on one GPU when given."""
    events = sorted([(a, 1) for a, _, g in spans if gpu in (None, g)] + [(b, -1) for _, b, g in spans if gpu in (None, g)])
    now = best = 0
    for _, step in events:
        now += step
        best = max(best, now)
    return best


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_data(root / "data")
        trace = root / "trace.txt"
        os.environ.update(DATA_DIR=str(root / "data"), RESULTS_DIR=str(root / "results"), FAKE_ENGINE_SLEEP="0.6", FAKE_ENGINE_TRACE=str(trace))
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        pool = LocalPool(["0", "0", "1"], engine=FAKE)
        jobs = []
        for i in range(7):
            req = JobRequest(cell_type="GMX", region=f"chr1:{i * 1000}-2000000", factors="CTCF", n_models=1)
            job = root / "results" / "jobs" / f"j{i}"
            pool.submit(asdict(req), str(job))
            jobs.append(job)
        pool.join()
        states = [json.loads((j / "status.json").read_text())["state"] for j in jobs]
        check("every job finishes", states == ["done"] * 7, str(states))
        spans = [(float(a), float(b), g) for a, b, g in (ln.split() for ln in trace.read_text().splitlines())]
        check("seven engines ran", len(spans) == 7, str(len(spans)))
        check("no more jobs at once than slots", most_at_once(spans) == 3, str(most_at_once(spans)))
        check("GPU 0 runs at most its two slots", most_at_once(spans, "0") <= 2, str(most_at_once(spans, "0")))
        check("GPU 1 runs at most its one slot", most_at_once(spans, "1") <= 1, str(most_at_once(spans, "1")))
        check("every engine sees one GPU of the slots", set(Counter(g for _, _, g in spans)) <= {"0", "1"}, str(Counter(g for _, _, g in spans)))
        busy = LocalPool(["0"], engine=FAKE)
        dirs = [str(root / "results" / "jobs" / f"q{i}") for i in range(3)]
        for i, jd in enumerate(dirs):
            busy.submit(asdict(JobRequest(cell_type="GMX", region=f"chr1:{i * 1000}-2000000", factors="CTCF", n_models=1)), jd)
        import time

        time.sleep(0.2)
        check("the queue gives the jobs in front of a waiting job", busy.position(dirs[2]) == 1 and busy.position(dirs[0]) is None, str([busy.position(x) for x in dirs]))
        busy.join()

        one = LocalPool(["0"], engine=FAKE)
        one.submit({"not_a_field": 1}, str(root / "results" / "jobs" / "bad"))
        good = root / "results" / "jobs" / "after_bad"
        one.submit(asdict(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=1)), str(good))
        one.join()
        after = json.loads((good / "status.json").read_text())["state"] if (good / "status.json").is_file() else "never ran"
        check("a slot keeps serving after a job that raises", after == "done", after)

        # A restart: one job was running, one waited with its request on disk, one waited without
        # it and one had finished.
        results = root / "restart"
        def left(name: str, state: str, request: bool) -> Path:
            job = results / "jobs" / name
            job.mkdir(parents=True)
            (job / "status.json").write_text(json.dumps({"state": state}))
            if request:
                (job / "request.json").write_text(json.dumps(asdict(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=1))))
            return job
        running, waiting, lost, finished = left("running", "running", True), left("waiting", "queued", True), left("lost", "queued", False), left("finished", "done", True)
        fresh = LocalPool(["0"], engine=FAKE)
        again = resume_interrupted(results, fresh)
        fresh.join()
        state = lambda job: json.loads((job / "status.json").read_text())  # noqa: E731
        check("a restart queues again the running and the waiting job", sorted(again) == ["running", "waiting"], str(again))
        check("both then finish", state(running)["state"] == "done" and state(waiting)["state"] == "done", str((state(running)["state"], state(waiting)["state"])))
        check("a waiting job without its request fails with the reason", state(lost)["state"] == "failed" and "Submit it again" in state(lost)["error"], str(state(lost)))
        check("a finished job is left as it is", state(finished) == {"state": "done"}, str(state(finished)))
    return done()


if __name__ == "__main__":
    sys.exit(main())
