"""The local queue: a fixed set of slots, one GPU per slot, jobs waiting for a free slot.

    pytest web/tests/test_queue.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixture import make_data

from gnome3d_web import siteconfig
from gnome3d_web.jobs import JobRequest, on_cores
from gnome3d_web.queue import LocalPool, resume_interrupted, server_cores, slot_cores

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_engine.py")]


def most_at_once(spans: list[tuple[float, float, str]], gpu: str | None = None) -> int:
    """The largest number of spans running at one moment, on one GPU when given."""
    events = sorted([(a, 1) for a, _, g in spans if gpu in (None, g)] + [(b, -1) for _, b, g in spans if gpu in (None, g)])
    now = best = 0
    for _, step in events:
        now += step
        best = max(best, now)
    return best


@pytest.fixture(scope="module", autouse=True)
def shared(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    d = str(tmp_path_factory.mktemp("work"))
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
    return SimpleNamespace(states=states, trace=trace, root=root, i=i)


def test_every_job_finishes(shared: SimpleNamespace) -> None:
    states, trace, root, i = shared.states, shared.trace, shared.root, shared.i
    assert states == ["done"] * 7, "every job finishes: " + str(states)
    spans = [(float(a), float(b), g) for a, b, g in (ln.split()[:3] for ln in trace.read_text().splitlines())]
    assert len(spans) == 7, "seven engines ran: " + str(len(spans))
    assert most_at_once(spans) == 3, "no more jobs at once than slots: " + str(most_at_once(spans))
    assert most_at_once(spans, "0") <= 2, "GPU 0 runs at most its two slots: " + str(most_at_once(spans, "0"))
    assert most_at_once(spans, "1") <= 1, "GPU 1 runs at most its one slot: " + str(most_at_once(spans, "1"))
    assert set(Counter(g for _, _, g in spans)) <= {"0", "1"}, "every engine sees one GPU of the slots: " + str(Counter(g for _, _, g in spans))
    busy = LocalPool(["0"], engine=FAKE)
    dirs = [str(root / "results" / "jobs" / f"q{i}") for i in range(3)]
    for i, jd in enumerate(dirs):
        busy.submit(asdict(JobRequest(cell_type="GMX", region=f"chr1:{i * 1000}-2000000", factors="CTCF", n_models=1)), jd)
    import time

    time.sleep(0.2)
    assert busy.position(dirs[2]) == 1 and busy.position(dirs[0]) is None, "the queue gives the jobs in front of a waiting job: " + str([busy.position(x) for x in dirs])
    busy.join()


def test_a_slot_keeps_serving_after_a_job_that_raises(shared: SimpleNamespace) -> None:
    root = shared.root
    one = LocalPool(["0"], engine=FAKE)
    one.submit({"not_a_field": 1}, str(root / "results" / "jobs" / "bad"))
    good = root / "results" / "jobs" / "after_bad"
    one.submit(asdict(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=1)), str(good))
    one.join()
    after = json.loads((good / "status.json").read_text())["state"] if (good / "status.json").is_file() else "never ran"
    assert after == "done", "a slot keeps serving after a job that raises: " + str(after)


def test_a_restart_queues_again_the_running_and_the_waiting_job(shared: SimpleNamespace) -> None:
    root = shared.root
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
    assert sorted(again) == ["running", "waiting"], "a restart queues again the running and the waiting job: " + str(again)
    assert state(running)["state"] == "done" and state(waiting)["state"] == "done", "both then finish: " + str((state(running)["state"], state(waiting)["state"]))
    assert state(lost)["state"] == "failed" and "Submit it again" in state(lost)["error"], "a waiting job without its request fails with the reason: " + str(state(lost))
    assert state(finished) == {"state": "done"}, "a finished job is left as it is: " + str(state(finished))


def test_each_slot_has_its_share_of_the_cores() -> None:
    cores = list(range(36))
    assert slot_cores(cores, 0, 2) == list(range(18)) and slot_cores(cores, 1, 2) == list(range(18, 36)), "two slots halve 36 cores"
    four = [slot_cores(cores, i, 4) for i in range(4)]
    assert [len(c) for c in four] == [9] * 4 and sorted(c for s in four for c in s) == cores, "four slots take a quarter each and no core twice"
    assert slot_cores(cores, 1, 2, 8) == list(range(8, 16)), "a count from the config file is the cores of one job: " + str(slot_cores(cores, 1, 2, 8))
    assert slot_cores(cores, 0, 2, 36) == cores and slot_cores(cores, 1, 2, 36) == cores, "a count of every core gives each job the whole server"
    assert slot_cores(cores, 1, 2, 100) == cores, "a count above the cores of the server is all of them"
    # The cores of a container are not always 0 to n - 1.
    assert slot_cores([4, 5, 6, 7, 12, 13], 1, 2) == [7, 12, 13], "the cores are the ones the server may use, in order"
    # Three slots of 24 cores on 36: the second slot starts at core 24 and goes on from core 0.
    assert slot_cores(cores, 1, 3, 24) == [*range(12), *range(24, 36)], "sets that do not fit beside each other overlap: " + str(slot_cores(cores, 1, 3, 24))
    assert slot_cores([0], 3, 4) == [0] and slot_cores([], 0, 2) == [], "one core serves every slot and no cores is no pinning"


def test_the_config_file_takes_the_cores_of_a_job() -> None:
    assert siteconfig.parse("{}").cores_per_job is None, "left out, the queue divides the cores by the slots"
    assert siteconfig.parse('{"limits": {"cores_per_job": 12}}').cores_per_job == 12, "the file sets the cores of one job"
    for bad in ("0", "-2", "2.5", "true", '"8"'):
        with pytest.raises(siteconfig.ConfigError, match="whole number of cores"):
            siteconfig.parse('{"limits": {"cores_per_job": %s}}' % bad)


def test_a_command_on_no_cores_is_the_command() -> None:
    assert on_cores(["engine", "--x"], None) == ["engine", "--x"] and on_cores(["engine"], []) == ["engine"], "no cores given is the command as it is"


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="this system cannot pin a process to cores")
def test_the_engine_of_a_slot_runs_on_the_cores_of_the_slot(shared: SimpleNamespace) -> None:
    mine = server_cores()
    if len(mine) < 2:
        pytest.skip("one core cannot be divided between two slots")
    root = shared.root
    trace = root / "cores_trace.txt"
    os.environ["FAKE_ENGINE_TRACE"] = str(trace)
    try:
        pool = LocalPool(["0", "1"], engine=FAKE)
        for k in range(4):
            pool.submit(asdict(JobRequest(cell_type="GMX", region=f"chr1:{k * 1000}-2000000", factors="CTCF", n_models=1)), str(root / "results" / "jobs" / f"cores{k}"))
        pool.join()
    finally:
        os.environ["FAKE_ENGINE_TRACE"] = str(shared.trace)
    seen = {(ln.split()[2], ln.split()[3]) for ln in trace.read_text().splitlines()}
    want = {(gpu, ",".join(str(c) for c in slot_cores(mine, i, 2))) for i, gpu in enumerate(["0", "1"])}
    assert seen == want, f"each engine has the cores of its slot and the GPU of its slot: {seen} against {want}"
    a, b = (set(c.split(",")) for _, c in sorted(want))
    assert not a & b, "the two slots share no core"
