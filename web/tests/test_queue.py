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

from gnome3d_web.jobs import JobRequest, on_cores
from gnome3d_web.queue import LocalPool, resume_interrupted, server_cores, slot_cores
from gnome3d_web import settings
from gnome3d_web.queue import claim_slot
from gnome3d_web.settings import Settings, SettingsError

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
    def left(name: str, state: str, request: bool, kind: str = "jobs") -> Path:
        job = results / kind / name
        job.mkdir(parents=True)
        (job / "status.json").write_text(json.dumps({"state": state}))
        if request:
            (job / "request.json").write_text(json.dumps(asdict(JobRequest(cell_type="GMX", region="chr1:0-2000000", factors="CTCF", n_models=1))))
        return job
    running, waiting, lost, finished = left("running", "running", True), left("waiting", "queued", True), left("lost", "queued", False), left("finished", "done", True)
    private = left("private_waiting", "queued", True, "private")
    fresh = LocalPool(["0"], engine=FAKE)
    again = resume_interrupted(results, fresh)
    fresh.join()
    assert "private_waiting" in again and json.loads((private / "status.json").read_text())["state"] == "done", "a waiting private job is queued again and finishes: " + str(again)
    again = [name for name in again if name != "private_waiting"]
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
    assert slot_cores(cores, 1, 2, 8) == list(range(8, 16)), "a count of WORKER_CORES is the cores of one job: " + str(slot_cores(cores, 1, 2, 8))
    assert slot_cores(cores, 0, 2, 36) == cores and slot_cores(cores, 1, 2, 36) == cores, "a count of every core gives each job the whole server"
    assert slot_cores(cores, 1, 2, 100) == cores, "a count above the cores of the server is all of them"
    # The cores of a container are not always 0 to n - 1.
    assert slot_cores([4, 5, 6, 7, 12, 13], 1, 2) == [7, 12, 13], "the cores are the ones the server may use, in order"
    # Three slots of 24 cores on 36: the second slot starts at core 24 and goes on from core 0.
    assert slot_cores(cores, 1, 3, 24) == [*range(12), *range(24, 36)], "sets that do not fit beside each other overlap: " + str(slot_cores(cores, 1, 3, 24))
    assert slot_cores([0], 3, 4) == [0] and slot_cores([], 0, 2) == [], "one core serves every slot and no cores is no pinning"
    # Eight slots on 36 cores: the division leaves four cores, one more for each of the first four slots.
    eight = [slot_cores(cores, i, 8) for i in range(8)]
    assert [len(c) for c in eight] == [5, 5, 5, 5, 4, 4, 4, 4] and sorted(c for s in eight for c in s) == cores, "no core is left without a slot and none is in two: " + str([len(c) for c in eight])


def test_workers_that_are_processes_of_their_own_take_a_free_slot_each(tmp_path: Path) -> None:
    with claim_slot(tmp_path / "slots", 2) as first, claim_slot(tmp_path / "slots", 2) as second, claim_slot(tmp_path / "slots", 2) as third:
        assert (first, second) == (0, 1), "two jobs at once take the two slots: " + str((first, second))
        assert third is None, "a job beyond the slots of the server has none"
    with claim_slot(tmp_path / "slots", 2) as again:
        assert again == 0, "a slot is free again after its job"


WORKER_VARIABLES = ("GPUS", "WORKERS_PER_GPU", "WORKER_CORES", "GPU_POOL_GIB", "GPU_SLOTS", "GNOME3D_GPU_POOL_GIB", "XLA_PYTHON_CLIENT_MEM_FRACTION")


def test_the_workers_of_the_site_are_four_variables_of_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in WORKER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    one = Settings.from_env()
    assert one.slots == ["0"] and one.worker_cores is None, "left out, the site runs one job at once on card 0 and divides the cores"
    assert one.engine_env("0") == {"CUDA_VISIBLE_DEVICES": "0"}, "one job on a card keeps the pool JAX takes"
    assert one.engine_env() == {}, "a worker without a slot has the cards of its process"
    monkeypatch.setenv("GPUS", "0, 1")
    monkeypatch.setenv("WORKERS_PER_GPU", "2")
    four = Settings.from_env()
    assert four.slots == ["0", "1", "0", "1"], "two cards with two workers each are four jobs at once, the first two on a card each"
    # With the size of the card each engine keeps 0.5 GiB beside its pool and the pools take nine tenths of the rest.
    monkeypatch.setattr(settings, "card_gib", lambda gpu: 8.0)
    assert four.engine_env("1") == {"CUDA_VISIBLE_DEVICES": "1", "GNOME3D_GPU_POOL_GIB": "3.15"}, "two jobs on a card of 8 GiB have 3.15 GiB each: " + str(four.engine_env("1"))
    monkeypatch.setenv("WORKERS_PER_GPU", "4")
    eight = Settings.from_env()
    pool = float(eight.engine_env("0")["GNOME3D_GPU_POOL_GIB"])
    assert pool == 1.35 and 4 * (pool + 0.5) < 8.0, "four engines with their pools fit a card of 8 GiB: " + str(pool)
    monkeypatch.setenv("WORKERS_PER_GPU", "2")
    # A card of unknown size: the share of JAX is divided.
    monkeypatch.setattr(settings, "card_gib", lambda gpu: None)
    assert four.engine_env("1") == {"CUDA_VISIBLE_DEVICES": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3750"}, "without the size of the card two jobs halve the pool JAX takes: " + str(four.engine_env("1"))
    monkeypatch.setenv("GPU_POOL_GIB", "1.5")
    monkeypatch.setenv("WORKER_CORES", "6")
    set_ = Settings.from_env()
    assert set_.worker_cores == 6 and set_.engine_env("0") == {"CUDA_VISIBLE_DEVICES": "0", "GNOME3D_GPU_POOL_GIB": "1.5"}, "a pool in GiB and a count of cores stand over the defaults"
    monkeypatch.setenv("GNOME3D_GPU_POOL_GIB", "2")
    assert Settings.from_env().engine_env("0") == {"CUDA_VISIBLE_DEVICES": "0"}, "a pool set for the process itself is left alone"


@pytest.mark.parametrize(("name", "value", "said"), [("WORKERS_PER_GPU", "0", "whole number"), ("WORKER_CORES", "2.5", "whole number"), ("GPU_POOL_GIB", "lots", "number of GiB"), ("GPU_SLOTS", "0,0,1,1", "GPUS=0,1 with WORKERS_PER_GPU=2")])
def test_a_worker_variable_the_site_cannot_use_stops_it_with_the_fix(monkeypatch: pytest.MonkeyPatch, name: str, value: str, said: str) -> None:
    for other in WORKER_VARIABLES:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingsError, match=said):
        Settings.from_env()


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
