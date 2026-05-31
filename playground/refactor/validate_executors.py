"""Validate the executors: SerialExecutor and BatchExecutor agree, batching
actually groups the ready set, and heterogeneous chains (some skipping HEAT_DIST)
schedule correctly.

Uses mock stages (a string-trace "kernel") so this tests the *scheduling*
(ready-set walk, (kind,bucket) grouping, dep ordering) in isolation from the MC
math — output is independent of drain policy, which is the property that lets
serial and batched paths produce the same beads.

    python playground/refactor/validate_executors.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, ".")
from gnome3d import pipeline as p  # noqa: E402
from gnome3d.pipeline.executor import BatchExecutor, SerialExecutor  # noqa: E402

_calls = {"serial": 0, "batch": 0}


@dataclass
class MockStage:
    kind: p.StageKind
    bkt: int

    def bucket(self, inputs):
        return self.bkt

    def to_problem(self, inputs):
        src = inputs[0]
        trace = src if isinstance(src, list) else [f"seed{src.seed}"]
        return {"trace": trace, "kind": self.kind.value}

    def apply(self, inputs, result):
        return result


def _serial_runner(prob):
    _calls["serial"] += 1
    return [*prob["trace"], prob["kind"]]


def _batch_runner(probs):
    _calls["batch"] += 1
    return [[*pr["trace"], pr["kind"]] for pr in probs]


def _register():
    for k in p.StageKind:
        # DENSIFY has no batch runner in production; mirror that here.
        p.register(k, serial=_serial_runner, batch=None if k is p.StageKind.DENSIFY else _batch_runner)


def _build_dag():
    """3 IBs: two full chains at buckets 256/512, one heat-SKIP chain at 256."""
    nodes, seeds = {}, {}

    def chain(ib, bkt, with_heat):
        kinds = [p.StageKind.ARCS, p.StageKind.DENSIFY]
        if with_heat:
            kinds.append(p.StageKind.HEAT_DIST)
        kinds.append(p.StageKind.SMOOTH)
        prev = None
        for kd in kinds:
            nid = f"{ib}:{kd.value}"
            nodes[nid] = p.Node(nid, MockStage(kd, bkt), () if prev is None else (prev,))
            if prev is None:
                seeds[nid] = p.Seeded(
                    settings=None, seed=ib, anchor_seed_pos=None, exp_dist=None,
                    orientations=None, anchor_neighbors=None, anchor_neighbor_weights=None,
                    subanchor_heat_raw=None, anchor_genomic=[], step_size_arcs=1.0,
                )
            prev = nid

    chain(0, 256, True)
    chain(1, 512, True)
    chain(2, 256, False)
    return p.Dag(nodes=nodes, seeds=seeds)


def main() -> int:
    _register()

    _calls.update(serial=0, batch=0)
    out_serial = SerialExecutor().run(_build_dag())
    serial_calls = _calls["serial"]

    _calls.update(serial=0, batch=0)
    out_batch = BatchExecutor().run(_build_dag())
    batch_calls, densify_calls = _calls["batch"], _calls["serial"]

    ok = True
    if out_serial != out_batch:
        print("FAIL: serial and batch outputs differ")
        ok = False
    if "heat_dist" in out_batch["2:smooth"]:
        print("FAIL: heat-skip chain ran HEAT_DIST")
        ok = False
    if not (batch_calls < serial_calls):
        print(f"FAIL: batching did not group (batch={batch_calls} !< serial={serial_calls})")
        ok = False

    print(f"  serial trace (IB0): {out_batch['0:smooth']}")
    print(f"  heat-skip   (IB2): {out_batch['2:smooth']}")
    print(f"  serial-exec runner calls: {serial_calls}")
    print(f"  batch-exec  batch calls: {batch_calls}, densify(serial) calls: {densify_calls}")
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
